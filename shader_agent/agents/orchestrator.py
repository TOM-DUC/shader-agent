"""Orchestrator：两个角色的串行调度器。

替代 MetaGPT 的 Environment。
支持三种组合任务：
  - analyze_only(code)              : 仅分析
  - generate_only(user_text)        : 仅生成
  - analyze_then_generate(code, ask): 先分析一段代码，再让 Generator 按"在此基础上 + ask"改写

每个组合任务返回的都是 dict（含两端 Memory 的关键产物 + 端到端总时间），
便于上层 UI/CLI 渲染。
"""
from __future__ import annotations

import time
from typing import Any

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.schemas import (
    AnalysisReport,
    GeneratedShader,
    GenerationSpec,
    Message,
)
from shader_agent.config.settings import settings
from shader_agent.observability import trace_span, update_current_trace
from shader_agent.utils.logger import logger


def _trace_attrs(task: str) -> dict:
    """构造 trace 级属性：服务名、环境、任务名、标签。"""
    obs = settings.observability
    tags = list(obs.tags or [])
    if task not in tags:
        tags.append(task)
    return {
        "name": f"orchestrator.{task}",
        "tags": tags,
        "metadata": {
            "service": obs.service_name,
            "environment": obs.environment,
            "framework": settings.orchestration.framework,
        },
    }


class Orchestrator:
    """两个 agent 的协作壳。"""

    def __init__(
        self,
        analyzer: ShaderAnalyzer | None = None,
        generator: ShaderGenerator | None = None,
    ) -> None:
        self.analyzer = analyzer or ShaderAnalyzer()
        self.generator = generator or ShaderGenerator()

    # ---------- 任务 1：仅分析 ----------
    def analyze_only(self, code: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        with trace_span("task.analyze_only", input={"code_len": len(code or "")}) as span:
            update_current_trace(**_trace_attrs("analyze_only"))
            in_msg = Message(role="user", content=code, payload={"code": code})
            out = self.analyzer.handle(in_msg)
            report = AnalysisReport(**out.payload) if out.payload_type == AnalysisReport.PAYLOAD_TYPE else None
            elapsed = (time.perf_counter() - t0) * 1000.0
            span.update(
                output={"techniques": (report.techniques if report else []),
                        "n_similar": len(report.similar_shaders) if report else 0},
                metadata={"elapsed_ms": round(elapsed, 1), "ok": report is not None},
            )
            return {
                "task": "analyze_only",
                "elapsed_ms": elapsed,
                "report": report,
                "messages": [in_msg, out],
            }

    # ---------- 任务 2：仅生成 ----------
    def generate_only(self, user_text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        with trace_span("task.generate_only", input={"prompt": (user_text or "")[:500]}) as span:
            update_current_trace(**_trace_attrs("generate_only"))
            in_msg = Message(role="user", content=user_text)
            out = self.generator.handle(in_msg)
            gen = GeneratedShader(**out.payload) if out.payload_type == GeneratedShader.PAYLOAD_TYPE else None
            elapsed = (time.perf_counter() - t0) * 1000.0
            span.update(
                output={
                    "compile_ok": bool(gen.compile_result.ok) if gen else False,
                    "iterations": gen.iterations if gen else 0,
                    "critique_score": gen.self_critique_score if gen else 0.0,
                },
                metadata={"elapsed_ms": round(elapsed, 1), "ok": gen is not None},
            )
            return {
                "task": "generate_only",
                "elapsed_ms": elapsed,
                "generated": gen,
                "messages": [in_msg, out],
            }

    # ---------- 任务 3：基于原代码改写（Remix） ----------
    def analyze_then_generate(
        self,
        code: str,
        ask: str,
        *,
        analyze_first: bool = True,
    ) -> dict[str, Any]:
        """改写任务：在**原始代码基础上**按 ask 做最小化修改。

        关键设计（针对"改写慢"和"分析太长被塞进提示词"两个问题）：
        - **默认不做单独的分析步骤**（analyze_first=False）。改写本身只需要原始
          代码——它会被直接作为 base_code 喂给 ShaderRemixer，模型读代码即可，
          不需要先花 20~40s 跑一次 explain，更不该把那段长摘要再塞进改写提示词
          （否则提示词膨胀、推理变慢、还更易触发多轮修正）。
        - Generator 走 rewrite_mode：基于 base_code 做最小化改动，不从零重写。
        - 若确实需要一份"原代码简析"展示给用户，UI 不再依赖独立分析，而是直接
          复用改写结果里的 explanation（与 Generator 解释同格式、同长度）。

        这样 Remixer 从原来的「分析(1次LLM) + 改写(1~N次LLM)」缩减为
        「改写(1~N次LLM)」，端到端时间大幅下降。
        """
        t0 = time.perf_counter()
        _span_cm = trace_span(
            "task.analyze_then_generate",
            input={"code_len": len(code or ""), "ask": (ask or "")[:300]},
        )
        _span = _span_cm.__enter__()
        update_current_trace(**_trace_attrs("analyze_then_generate"))

        report: AnalysisReport | None = None
        in1 = Message(role="user", content=code, payload={"code": code})
        out1 = in1  # 默认不分析时，parent 直接用输入消息
        if analyze_first:
            # 可选：仅当显式要求时才跑一次轻量分析（single 策略）
            prev = getattr(self.analyzer, "_strategy", None)
            try:
                if prev is not None:
                    self.analyzer._strategy = "single"
                out1 = self.analyzer.handle(in1)
                if out1.payload_type == AnalysisReport.PAYLOAD_TYPE:
                    report = AnalysisReport(**out1.payload)
            finally:
                if prev is not None:
                    self.analyzer._strategy = prev

        # 改写：带 base_code + rewrite_mode。注意不传 reference_report，
        # 让 DraftCodeAction 的改写分支只看原始代码，提示词保持精简。
        spec = GenerationSpec(
            description=ask,
            base_code=code,
            rewrite_mode=True,
        )
        in2 = spec.to_message(parent_id=out1.msg_id)
        out2 = self.generator.handle(in2)
        gen = GeneratedShader(**out2.payload) if out2.payload_type == GeneratedShader.PAYLOAD_TYPE else None

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[orch] remix done in {elapsed:.1f}ms "
            f"(analyze_first={analyze_first}, generator mem={len(self.generator.memory)})"
        )
        _span.update(
            output={
                "compile_ok": bool(gen.compile_result.ok) if gen else False,
                "iterations": gen.iterations if gen else 0,
            },
            metadata={"elapsed_ms": round(elapsed, 1),
                      "analyze_first": analyze_first, "ok": gen is not None},
        )
        _span_cm.__exit__(None, None, None)
        return {
            "task": "analyze_then_generate",
            "elapsed_ms": elapsed,
            "report": report,
            "generated": gen,
            "messages": [in1, out1, in2, out2],
        }
