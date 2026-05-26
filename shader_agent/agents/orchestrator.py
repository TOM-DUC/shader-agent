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
from shader_agent.utils.logger import logger


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
        in_msg = Message(role="user", content=code, payload={"code": code})
        out = self.analyzer.handle(in_msg)
        report = AnalysisReport(**out.payload) if out.payload_type == AnalysisReport.PAYLOAD_TYPE else None
        return {
            "task": "analyze_only",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "report": report,
            "messages": [in_msg, out],
        }

    # ---------- 任务 2：仅生成 ----------
    def generate_only(self, user_text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        in_msg = Message(role="user", content=user_text)
        out = self.generator.handle(in_msg)
        gen = GeneratedShader(**out.payload) if out.payload_type == GeneratedShader.PAYLOAD_TYPE else None
        return {
            "task": "generate_only",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "generated": gen,
            "messages": [in_msg, out],
        }

    # ---------- 任务 3：先分析后改写 ----------
    def analyze_then_generate(
        self,
        code: str,
        ask: str,
    ) -> dict[str, Any]:
        """先让 Analyzer 解读 code，再用其 AnalysisReport 作为 reference，
        让 Generator 在此基础上按 ask 的指令改写。"""
        t0 = time.perf_counter()

        # 第一段：analyze
        in1 = Message(role="user", content=code, payload={"code": code})
        out1 = self.analyzer.handle(in1)
        report: AnalysisReport | None = None
        if out1.payload_type == AnalysisReport.PAYLOAD_TYPE:
            report = AnalysisReport(**out1.payload)

        # 第二段：generator 吃 spec.reference_report
        spec = GenerationSpec(
            description=ask,
            reference_report=report,
        )
        in2 = spec.to_message(parent_id=out1.msg_id)
        out2 = self.generator.handle(in2)
        gen = GeneratedShader(**out2.payload) if out2.payload_type == GeneratedShader.PAYLOAD_TYPE else None

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[orch] analyze_then_generate done in {elapsed:.1f}ms "
            f"(analyzer mem={len(self.analyzer.memory)}, generator mem={len(self.generator.memory)})"
        )
        return {
            "task": "analyze_then_generate",
            "elapsed_ms": elapsed,
            "report": report,
            "generated": gen,
            "messages": [in1, out1, in2, out2],
        }
