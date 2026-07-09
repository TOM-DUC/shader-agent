"""评估执行器：把 goldens 跑过真实链路 → 计算指标 → 回流 Langfuse → 产出报告。

闭环设计（这是接入 langfuse + deepeval 的关键价值点）：

    golden ──► 真实 Orchestrator 链路 ──► 产物 + trace_id
                                            │
                                            ├─► deepeval 指标计算
                                            │
                                            └─► score_trace_by_id(trace_id, metric, score)

每条 golden 单独开一个 trace，评估分数通过 trace_id 精确挂回该次运行。
于是在 Langfuse 看板上，一次生成的「耗时 / token / 检索命中 / 质量分」在同一视图里，
可以直接回答"哪些 prompt 最贵""低分是否总伴随多轮修正""检索为空是否导致降分"。

离线降级：
  - deepeval 未安装 → 只跑确定性指标（纯函数），仍能产出完整量化报告；
  - langfuse 未配置 → 跳过分数回流，报告照常落盘；
  - DEEPSEEK_API_KEY 缺失 → LLM 裁判指标自动跳过。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from shader_agent.config.settings import settings
from shader_agent.eval.datasets import (
    ANALYSIS_GOLDENS,
    GENERATION_GOLDENS,
    RETRIEVAL_GOLDENS,
    AnalysisGolden,
    GenerationGolden,
    RetrievalGolden,
    resolve_analysis_code,
)
from shader_agent.eval.judge_model import build_judge_model
from shader_agent.eval.metrics import (
    build_analysis_metrics,
    build_generation_metrics,
    build_retrieval_metrics,
    score_retrieval_relevancy,
    score_shadertoy_convention,
)
from shader_agent.observability import (
    flush as lf_flush,
    get_current_trace_id,
    is_enabled as lf_enabled,
    score_trace_by_id,
    trace_span,
    update_current_trace,
)
from shader_agent.utils.logger import logger


# =====================================================================
# 结果容器
# =====================================================================

@dataclass
class MetricScore:
    name: str
    score: float
    threshold: float
    success: bool
    reason: str = ""
    is_llm_judge: bool = False


@dataclass
class CaseResult:
    case_id: str
    task: str                      # generation / analysis / retrieval
    ok: bool
    elapsed_ms: float
    trace_id: str = ""
    metrics: list[MetricScore] = field(default_factory=list)
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)

    @property
    def passed(self) -> bool:
        return self.ok and all(m.success for m in self.metrics)


@dataclass
class EvalReport:
    started_at: float
    elapsed_s: float
    cases: list[CaseResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # ---------- 聚合 ----------
    def aggregate(self) -> dict[str, Any]:
        by_metric: dict[str, list[float]] = {}
        for c in self.cases:
            for m in c.metrics:
                by_metric.setdefault(m.name, []).append(m.score)

        agg = {
            "n_cases": len(self.cases),
            "n_passed": sum(1 for c in self.cases if c.passed),
            "pass_rate": (
                sum(1 for c in self.cases if c.passed) / len(self.cases)
                if self.cases else 0.0
            ),
            "mean_score": (
                sum(c.mean_score for c in self.cases) / len(self.cases)
                if self.cases else 0.0
            ),
            "avg_latency_ms": (
                sum(c.elapsed_ms for c in self.cases) / len(self.cases)
                if self.cases else 0.0
            ),
            "by_metric": {
                k: {
                    "mean": round(sum(v) / len(v), 4),
                    "min": round(min(v), 4),
                    "max": round(max(v), 4),
                    "n": len(v),
                }
                for k, v in by_metric.items()
            },
        }
        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 2),
            "config": self.config,
            "aggregate": self.aggregate(),
            "cases": [asdict(c) for c in self.cases],
        }

    def to_markdown(self) -> str:
        agg = self.aggregate()
        lines = ["# Shader Agent 评估报告\n"]
        lines.append(
            f"- 用例数：**{agg['n_cases']}**　通过：**{agg['n_passed']}**　"
            f"通过率：**{agg['pass_rate']:.1%}**"
        )
        lines.append(f"- 平均分：**{agg['mean_score']:.3f}**")
        lines.append(f"- 平均端到端耗时：**{agg['avg_latency_ms']:.0f} ms**")
        lines.append(f"- 总耗时：{self.elapsed_s:.1f}s")
        lines.append(f"- 评审模型：`{self.config.get('judge_model', 'n/a')}`")
        lines.append(f"- Langfuse：{'已启用' if self.config.get('langfuse') else '未启用'}\n")

        lines.append("## 指标汇总\n")
        lines.append("| 指标 | 均值 | 最小 | 最大 | 样本数 |")
        lines.append("|---|---|---|---|---|")
        for k, v in agg["by_metric"].items():
            lines.append(f"| {k} | {v['mean']:.3f} | {v['min']:.3f} | {v['max']:.3f} | {v['n']} |")
        lines.append("")

        lines.append("## 用例明细\n")
        for c in self.cases:
            flag = "PASS" if c.passed else "FAIL"
            lines.append(f"### `{c.case_id}` · {c.task} · **{flag}** · {c.elapsed_ms:.0f}ms")
            if c.trace_id:
                lines.append(f"trace_id: `{c.trace_id}`")
            if c.error:
                lines.append(f"\n> 错误：{c.error}\n")
            if c.metrics:
                lines.append("")
                lines.append("| 指标 | 分数 | 阈值 | 结果 | 说明 |")
                lines.append("|---|---|---|---|---|")
                for m in c.metrics:
                    ok = "✓" if m.success else "✗"
                    judge = " (LLM)" if m.is_llm_judge else ""
                    reason = (m.reason or "").replace("\n", " ")[:110]
                    lines.append(
                        f"| {m.name}{judge} | {m.score:.3f} | {m.threshold:.2f} | {ok} | {reason} |"
                    )
            if c.extras:
                lines.append(f"\n<details><summary>extras</summary>\n\n```json\n"
                             f"{json.dumps(c.extras, ensure_ascii=False, indent=2)}\n```\n</details>")
            lines.append("")
        return "\n".join(lines)


# =====================================================================
# 测试用例构造（兼容 deepeval 未安装）
# =====================================================================

def _make_test_case(**kwargs: Any) -> Any:
    """构造 LLMTestCase；deepeval 缺失时用等价的轻量对象顶替。"""
    try:
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(**kwargs)
    except Exception:
        class _Shim:
            def __init__(self, **kw: Any) -> None:
                self.input = kw.get("input", "")
                self.actual_output = kw.get("actual_output", "")
                self.expected_output = kw.get("expected_output")
                self.context = kw.get("context")
                self.retrieval_context = kw.get("retrieval_context")
                self.additional_metadata = kw.get("additional_metadata") or {}
        return _Shim(**kwargs)


def _run_metrics(metrics: list[Any], test_case: Any) -> list[MetricScore]:
    """逐个跑指标。单个指标失败不影响其余指标。"""
    out: list[MetricScore] = []
    for m in metrics:
        name = getattr(m, "__name__", None) or getattr(m, "name", m.__class__.__name__)
        is_judge = "deterministic" not in str(getattr(m, "evaluation_model", "")).lower()
        try:
            m.measure(test_case)
            out.append(MetricScore(
                name=str(name),
                score=float(getattr(m, "score", 0.0) or 0.0),
                threshold=float(getattr(m, "threshold", 0.5) or 0.5),
                success=bool(getattr(m, "success", False)),
                reason=str(getattr(m, "reason", "") or ""),
                is_llm_judge=is_judge,
            ))
        except Exception as e:
            logger.warning(f"[eval] 指标 {name} 计算失败: {e}")
            out.append(MetricScore(
                name=str(name), score=0.0,
                threshold=float(getattr(m, "threshold", 0.5) or 0.5),
                success=False, reason=f"指标异常: {type(e).__name__}: {e}",
                is_llm_judge=is_judge,
            ))
    return out


def _push_scores(trace_id: str, metrics: list[MetricScore]) -> None:
    """把指标分数回流到对应 trace。"""
    if not trace_id or not settings.evaluation.push_scores_to_langfuse:
        return
    for m in metrics:
        score_trace_by_id(
            trace_id,
            name=f"eval.{m.name.replace(' ', '_').lower()}",
            value=float(m.score),
            comment=(m.reason or "")[:400],
        )


# =====================================================================
# 三类任务的执行
# =====================================================================

class EvalRunner:
    """评估编排器。持有一次装配好的 agent 与指标套件。"""

    def __init__(
        self,
        *,
        with_llm_judge: bool = True,
        render_backend: str = "auto",
        use_vector_store: str = "auto",
        max_fix_loops: int = 1,
        top_k: int = 3,
    ) -> None:
        self.with_llm_judge = with_llm_judge
        self.judge = build_judge_model() if with_llm_judge else None
        if with_llm_judge and self.judge is None:
            logger.warning("[eval] 评审模型不可用，本次只跑确定性指标")
            self.with_llm_judge = False

        # 复用 UI 的装配逻辑，确保"评估的就是线上跑的那套"
        from shader_agent.ui.runners import AssemblyOptions, get_assembly
        self._opts = AssemblyOptions(
            render_backend=render_backend,
            use_vector_store=use_vector_store,
            use_llm_cache=True,
            enable_self_critique=False,   # 自评与评估职责重叠，评估时关掉更干净
            max_fix_loops=max_fix_loops,
            top_k=top_k,
        )
        self._asm = get_assembly(self._opts)
        self.max_fix_loops = max_fix_loops
        logger.info(f"[eval] 装配完成: {'; '.join(self._asm.diagnostics)}")

    # ---------- 生成 ----------
    def run_generation_case(self, g: GenerationGolden) -> CaseResult:
        from shader_agent.agents.schemas import GeneratedShader, GenerationSpec

        t0 = time.perf_counter()
        trace_id = ""
        try:
            with trace_span("eval.generation", input={"case_id": g.case_id,
                                                      "prompt": g.prompt}) as span:
                update_current_trace(
                    name=f"eval.generation.{g.case_id}",
                    tags=list(settings.observability.tags or []) + ["eval", "generation"],
                    metadata={"case_id": g.case_id, "effect_type": g.effect_type},
                )
                trace_id = get_current_trace_id() or ""

                spec = GenerationSpec(
                    description=g.prompt,
                    effect_type=g.effect_type,
                    palette=g.palette,
                    dynamic=g.dynamic,
                    complexity=g.complexity,  # type: ignore[arg-type]
                )
                msg = self._asm.generator.handle(spec.to_message())
                if msg.payload_type != GeneratedShader.PAYLOAD_TYPE:
                    raise RuntimeError(f"Generator 返回非预期消息: {msg.content[:120]}")
                gen = GeneratedShader(**msg.payload)
                span.update(output={"compile_ok": bool(gen.compile_result.ok),
                                    "iterations": gen.iterations})

            elapsed = (time.perf_counter() - t0) * 1000.0

            hits = [
                {"shader_id": s.shader_id, "name": s.name,
                 "fused_score": max(0.0, 1.0 - float(s.distance or 0.0))}
                for s in (gen.references_used or [])
            ]
            spec_text = (
                f"description={g.prompt}; effect_type={g.effect_type}; "
                f"palette={g.palette}; dynamic={g.dynamic}; complexity={g.complexity}"
            )
            tc = _make_test_case(
                input=spec_text,
                actual_output=gen.code,
                context=[gen.code],
                retrieval_context=[
                    (s.reference_context or s.code_excerpt or "")[:2000]
                    for s in (gen.references_used or [])
                ] or None,
                additional_metadata={
                    "compile_ok": bool(gen.compile_result.ok),
                    "compile_errors": gen.compile_result.errors or "",
                    "iterations": gen.iterations,
                    "max_fix_loops": self.max_fix_loops,
                    "retrieval_hits": hits,
                },
            )
            metrics = _run_metrics(
                build_generation_metrics(self.judge, with_llm_judge=self.with_llm_judge),
                tc,
            )
            # 解释忠实度需要 actual_output=解释、context=代码，单独构一个用例
            if self.with_llm_judge and self.judge is not None and gen.explanation:
                from shader_agent.eval.metrics import explanation_faithfulness_metric
                m = explanation_faithfulness_metric(self.judge)
                if m is not None:
                    tc2 = _make_test_case(
                        input=spec_text,
                        actual_output=gen.explanation,
                        context=[gen.code],
                    )
                    # 覆盖掉套件里那条（套件里的 actual_output 是代码，语义不对）
                    metrics = [x for x in metrics if x.name != "Explanation Faithfulness"]
                    metrics.extend(_run_metrics([m], tc2))

            # 确定性硬断言：must_contain / must_not_contain
            extras = self._assert_contains(g, gen.code)
            extras["iterations"] = gen.iterations
            extras["compile_ok"] = bool(gen.compile_result.ok)
            extras["n_references"] = len(hits)

            _push_scores(trace_id, metrics)
            return CaseResult(case_id=g.case_id, task="generation", ok=True,
                              elapsed_ms=elapsed, trace_id=trace_id,
                              metrics=metrics, extras=extras)
        except Exception as e:
            logger.exception(f"[eval] generation case {g.case_id} 失败")
            return CaseResult(case_id=g.case_id, task="generation", ok=False,
                              elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                              trace_id=trace_id, error=f"{type(e).__name__}: {e}")

    @staticmethod
    def _assert_contains(g: GenerationGolden, code: str) -> dict[str, Any]:
        missing = [k for k in g.must_contain if k not in code]
        forbidden = [k for k in g.must_not_contain if k in code]
        conv_score, violations = score_shadertoy_convention(code)
        return {
            "must_contain_missing": missing,
            "must_not_contain_hit": forbidden,
            "convention_violations": violations,
            "convention_score": round(conv_score, 4),
        }

    # ---------- 分析 ----------
    def run_analysis_case(self, g: AnalysisGolden) -> CaseResult:
        t0 = time.perf_counter()
        trace_id = ""
        code = resolve_analysis_code(g)
        if not code.strip():
            return CaseResult(case_id=g.case_id, task="analysis", ok=False,
                              elapsed_ms=0.0, error=f"无法解析源码 (seed_id={g.seed_id})")
        try:
            with trace_span("eval.analysis", input={"case_id": g.case_id,
                                                    "code_len": len(code)}) as span:
                update_current_trace(
                    name=f"eval.analysis.{g.case_id}",
                    tags=list(settings.observability.tags or []) + ["eval", "analysis"],
                    metadata={"case_id": g.case_id, "seed_id": g.seed_id},
                )
                trace_id = get_current_trace_id() or ""
                result = self._asm.orchestrator.analyze_only(code)
                report = result.get("report")
                if report is None:
                    raise RuntimeError("Analyzer 未产出 report")
                span.update(output={"techniques": report.techniques})

            elapsed = (time.perf_counter() - t0) * 1000.0

            report_text = (
                f"算法摘要：{report.algorithm_summary}\n"
                f"技术标签：{', '.join(report.techniques)}\n"
                f"视觉效果：{report.visual_effect}"
            )
            hits = [
                {"shader_id": s.shader_id,
                 "fused_score": max(0.0, 1.0 - float(s.distance or 0.0))}
                for s in (report.similar_shaders or [])
            ]
            tc = _make_test_case(
                input=code[:6000],
                actual_output=report_text,
                retrieval_context=[
                    (s.reference_context or s.code_excerpt or "")[:1500]
                    for s in (report.similar_shaders or [])
                ] or None,
                additional_metadata={"retrieval_hits": hits},
            )
            metrics = _run_metrics(
                build_analysis_metrics(self.judge, with_llm_judge=self.with_llm_judge),
                tc,
            )
            extras: dict[str, Any] = {
                "techniques": report.techniques,
                "n_similar": len(hits),
            }
            if g.expected_techniques:
                got = set(report.techniques or [])
                want = set(g.expected_techniques)
                extras["technique_recall"] = round(
                    len(got & want) / len(want) if want else 0.0, 4
                )

            _push_scores(trace_id, metrics)
            return CaseResult(case_id=g.case_id, task="analysis", ok=True,
                              elapsed_ms=elapsed, trace_id=trace_id,
                              metrics=metrics, extras=extras)
        except Exception as e:
            logger.exception(f"[eval] analysis case {g.case_id} 失败")
            return CaseResult(case_id=g.case_id, task="analysis", ok=False,
                              elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                              trace_id=trace_id, error=f"{type(e).__name__}: {e}")

    # ---------- 检索 ----------
    def run_retrieval_case(self, g: RetrievalGolden) -> CaseResult:
        t0 = time.perf_counter()
        trace_id = ""
        retriever = self._asm.retriever
        if retriever is None:
            return CaseResult(case_id=g.case_id, task="retrieval", ok=False,
                              elapsed_ms=0.0, error="检索器未装配（需先建库）")
        try:
            with trace_span("eval.retrieval", input={"case_id": g.case_id,
                                                     "query": g.query}) as span:
                update_current_trace(
                    name=f"eval.retrieval.{g.case_id}",
                    tags=list(settings.observability.tags or []) + ["eval", "retrieval"],
                    metadata={"case_id": g.case_id},
                )
                trace_id = get_current_trace_id() or ""
                hits = retriever.retrieve(g.query, top_k=3, want_tags=g.want_tags)
                span.update(output={"n_hits": len(hits)})

            elapsed = (time.perf_counter() - t0) * 1000.0

            hit_dicts = [
                {"shader_id": h.shader_id, "name": h.name,
                 "fused_score": h.fused_score, "tags": h.tags_topic}
                for h in hits
            ]
            tc = _make_test_case(
                input=g.query,
                actual_output=f"检索到 {len(hits)} 条参考",
                retrieval_context=[h.build_reference_context(1500) for h in hits] or None,
                additional_metadata={"retrieval_hits": hit_dicts},
            )

            # 负样例（期望不召回）：翻转判定——不返回参考才是正确行为
            is_negative = (not g.want_tags) and (not g.expected_shader_ids) and \
                          g.case_id.endswith("irrelevant")
            if is_negative:
                score = 1.0 if not hits else 0.0
                metrics = [MetricScore(
                    name="Negative Rejection", score=score, threshold=1.0,
                    success=bool(score >= 1.0),
                    reason=("正确地未返回不相关参考（阈值生效）" if not hits
                            else f"错误召回了 {len(hits)} 条不相关参考"),
                )]
            else:
                metrics = _run_metrics(
                    build_retrieval_metrics(self.judge, with_llm_judge=self.with_llm_judge),
                    tc,
                )

            rel, detail = score_retrieval_relevancy(hit_dicts)
            extras: dict[str, Any] = {"hits": hit_dicts, "relevancy_detail": detail}
            if g.expected_shader_ids:
                got = {h["shader_id"] for h in hit_dicts}
                extras["expected_recall"] = round(
                    len(got & set(g.expected_shader_ids)) / len(g.expected_shader_ids), 4
                )

            _push_scores(trace_id, metrics)
            return CaseResult(case_id=g.case_id, task="retrieval", ok=True,
                              elapsed_ms=elapsed, trace_id=trace_id,
                              metrics=metrics, extras=extras)
        except Exception as e:
            logger.exception(f"[eval] retrieval case {g.case_id} 失败")
            return CaseResult(case_id=g.case_id, task="retrieval", ok=False,
                              elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                              trace_id=trace_id, error=f"{type(e).__name__}: {e}")

    # ---------- 总入口 ----------
    def run(
        self,
        *,
        tasks: tuple[str, ...] = ("retrieval", "generation", "analysis"),
        limit: int = 0,
    ) -> EvalReport:
        t0 = time.perf_counter()
        started = time.time()
        cases: list[CaseResult] = []

        if "retrieval" in tasks:
            goldens = RETRIEVAL_GOLDENS[:limit] if limit else RETRIEVAL_GOLDENS
            for g in goldens:
                logger.info(f"[eval] retrieval · {g.case_id}")
                cases.append(self.run_retrieval_case(g))

        if "generation" in tasks:
            goldens = GENERATION_GOLDENS[:limit] if limit else GENERATION_GOLDENS
            for g in goldens:
                logger.info(f"[eval] generation · {g.case_id}")
                cases.append(self.run_generation_case(g))

        if "analysis" in tasks:
            goldens = ANALYSIS_GOLDENS[:limit] if limit else ANALYSIS_GOLDENS
            for g in goldens:
                logger.info(f"[eval] analysis · {g.case_id}")
                cases.append(self.run_analysis_case(g))

        lf_flush()  # 确保短生命周期脚本退出前把 trace 与 score 都发出去

        return EvalReport(
            started_at=started,
            elapsed_s=time.perf_counter() - t0,
            cases=cases,
            config={
                "judge_model": (self.judge.get_model_name() if self.judge else "none"),
                "with_llm_judge": self.with_llm_judge,
                "langfuse": lf_enabled(),
                "max_fix_loops": self.max_fix_loops,
                "top_k": self._opts.top_k,
                "retrieval_min_score": settings.retrieval.min_score,
                "use_rerank": settings.retrieval.use_rerank,
            },
        )


# =====================================================================
# 落盘
# =====================================================================

def save_report(report: EvalReport, name: str = "eval") -> Path:
    """把报告写到 data/reports/eval_{ts}/ ，返回目录。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = settings.project_root / "data" / "reports" / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    logger.info(f"[eval] 报告已落盘: {out_dir}")
    return out_dir
