"""评估层的离线单测。

确定性指标是纯函数，必须在**不装 deepeval、不联网、不调 LLM**的情况下全部可测。
这也是把它们与 LLM 裁判分层的直接收益。
"""
from __future__ import annotations

import pytest


# =====================================================================
# Shadertoy 规范符合度
# =====================================================================

_GOOD = """
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));
    fragColor = vec4(col, 1.0);
}
"""

_BAD_TEXTURE = """
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = texture(iChannel0, uv);
}
"""

_BAD_ES3 = """
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    float v = round(fragCoord.x);
    fragColor = vec4(v, 0.0, 0.0, 1.0);
}
"""

_BAD_SIG = """
void mainImage(vec4 fragColor, vec2 fragCoord){
    fragColor = vec4(1.0);
}
"""

_BAD_BRACES = """
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    fragColor = vec4(1.0);
"""


def test_convention_good_code_scores_full():
    from shader_agent.eval.metrics import score_shadertoy_convention

    score, violations = score_shadertoy_convention(_GOOD)
    assert score == pytest.approx(1.0)
    assert violations == []


def test_convention_detects_external_texture():
    from shader_agent.eval.metrics import score_shadertoy_convention

    score, violations = score_shadertoy_convention(_BAD_TEXTURE)
    assert score < 1.0
    assert any("外部纹理" in v for v in violations)


def test_convention_detects_es3_function():
    from shader_agent.eval.metrics import score_shadertoy_convention

    score, violations = score_shadertoy_convention(_BAD_ES3)
    assert score < 1.0
    assert any("ES3.0" in v for v in violations)


def test_convention_detects_bad_signature():
    from shader_agent.eval.metrics import score_shadertoy_convention

    score, violations = score_shadertoy_convention(_BAD_SIG)
    assert score < 1.0
    assert any("mainImage" in v for v in violations)


def test_convention_detects_unbalanced_braces():
    from shader_agent.eval.metrics import score_shadertoy_convention

    score, violations = score_shadertoy_convention(_BAD_BRACES)
    assert any("括号" in v for v in violations)


def test_convention_ignores_comments():
    """注释里的 iChannel0 / round( 不应被误判。"""
    from shader_agent.eval.metrics import score_shadertoy_convention

    code = _GOOD + "\n// 这里本可以用 iChannel0 和 round( 但没有用\n"
    score, violations = score_shadertoy_convention(code)
    assert score == pytest.approx(1.0), violations


# =====================================================================
# 修正轮数效率
# =====================================================================

def test_fix_loop_efficiency_monotonic():
    from shader_agent.eval.metrics import score_fix_loop_efficiency

    s1 = score_fix_loop_efficiency(1, max_loops=3)
    s2 = score_fix_loop_efficiency(2, max_loops=3)
    s3 = score_fix_loop_efficiency(3, max_loops=3)
    assert s1 == pytest.approx(1.0)
    assert s1 > s2 > s3 >= 0.0


def test_fix_loop_efficiency_zero_iterations():
    from shader_agent.eval.metrics import score_fix_loop_efficiency

    assert score_fix_loop_efficiency(0) == 0.0


# =====================================================================
# 检索相关性
# =====================================================================

def test_retrieval_relevancy_empty():
    from shader_agent.eval.metrics import score_retrieval_relevancy

    score, detail = score_retrieval_relevancy([])
    assert score == 0.0
    assert detail["n_hits"] == 0


def test_retrieval_relevancy_all_above_threshold():
    from shader_agent.eval.metrics import score_retrieval_relevancy

    hits = [{"fused_score": 0.9}, {"fused_score": 0.8}]
    score, detail = score_retrieval_relevancy(hits, min_score=0.15)
    assert detail["hit_rate"] == pytest.approx(1.0)
    assert score > 0.8


def test_retrieval_relevancy_mixed():
    from shader_agent.eval.metrics import score_retrieval_relevancy

    hits = [{"fused_score": 0.9}, {"fused_score": 0.05}]
    score, detail = score_retrieval_relevancy(hits, min_score=0.15)
    assert detail["hit_rate"] == pytest.approx(0.5)
    assert 0.0 < score < 1.0


def test_retrieval_relevancy_accepts_distance_field():
    """兼容 SimilarShader（只有 distance，没有 fused_score）。"""
    from shader_agent.eval.metrics import score_retrieval_relevancy

    hits = [{"distance": 0.1}]   # → fused 0.9
    score, detail = score_retrieval_relevancy(hits, min_score=0.15)
    assert detail["avg_score"] == pytest.approx(0.9)
    assert score > 0.5


# =====================================================================
# 确定性 deepeval 指标（BaseMetric 子类，无 LLM）
# =====================================================================

class _TC:
    """最小 test case 替身。"""
    def __init__(self, actual_output="", additional_metadata=None):
        self.actual_output = actual_output
        self.additional_metadata = additional_metadata or {}


def test_compile_success_metric():
    from shader_agent.eval.metrics import CompileSuccessMetric

    m = CompileSuccessMetric()
    assert m.measure(_TC(additional_metadata={"compile_ok": True})) == 1.0
    assert m.is_successful() is True

    m2 = CompileSuccessMetric()
    assert m2.measure(_TC(additional_metadata={"compile_ok": False,
                                               "compile_errors": "syntax"})) == 0.0
    assert m2.is_successful() is False
    assert "syntax" in m2.reason


def test_shadertoy_convention_metric():
    from shader_agent.eval.metrics import ShadertoyConventionMetric

    m = ShadertoyConventionMetric(threshold=0.8)
    assert m.measure(_TC(actual_output=_GOOD)) == pytest.approx(1.0)
    assert m.is_successful()

    m2 = ShadertoyConventionMetric(threshold=0.8)
    s = m2.measure(_TC(actual_output=_BAD_TEXTURE))
    assert s < 1.0


def test_fix_loop_metric():
    from shader_agent.eval.metrics import FixLoopEfficiencyMetric

    m = FixLoopEfficiencyMetric()
    s = m.measure(_TC(additional_metadata={"iterations": 1, "max_fix_loops": 2}))
    assert s == pytest.approx(1.0)


def test_retrieval_metric_empty_gives_zero():
    from shader_agent.eval.metrics import RetrievalRelevancyMetric

    m = RetrievalRelevancyMetric()
    s = m.measure(_TC(additional_metadata={"retrieval_hits": []}))
    assert s == 0.0
    assert "宁缺毋滥" in m.reason


def test_deterministic_metrics_declare_zero_cost():
    """确定性指标必须显式声明零成本，便于报告里区分 LLM 花费。"""
    from shader_agent.eval.metrics import (
        CompileSuccessMetric,
        ShadertoyConventionMetric,
    )

    for cls in (CompileSuccessMetric, ShadertoyConventionMetric):
        m = cls()
        assert m.evaluation_cost == 0.0
        assert "deterministic" in m.evaluation_model.lower()


def test_build_metrics_without_judge_has_no_llm_metrics():
    from shader_agent.eval.metrics import build_generation_metrics

    metrics = build_generation_metrics(model=None, with_llm_judge=False)
    assert len(metrics) == 4
    for m in metrics:
        assert m.evaluation_cost == 0.0


# =====================================================================
# 数据集
# =====================================================================

def test_datasets_are_non_empty_and_well_formed():
    from shader_agent.eval.datasets import (
        ANALYSIS_GOLDENS,
        GENERATION_GOLDENS,
        RETRIEVAL_GOLDENS,
        summary,
    )

    assert len(GENERATION_GOLDENS) >= 5
    assert len(RETRIEVAL_GOLDENS) >= 4
    assert len(ANALYSIS_GOLDENS) >= 1

    ids = [g.case_id for g in GENERATION_GOLDENS]
    assert len(ids) == len(set(ids)), "case_id 必须唯一"

    for g in GENERATION_GOLDENS:
        assert g.prompt.strip()
        assert g.complexity in ("minimal", "simple", "moderate", "complex")

    s = summary()
    assert s["total"] == s["generation"] + s["analysis"] + s["retrieval"]


def test_generation_goldens_cover_effect_types():
    """评估集要覆盖能力边界，而不是同质样例堆砌。"""
    from shader_agent.eval.datasets import GENERATION_GOLDENS

    effects = {g.effect_type for g in GENERATION_GOLDENS}
    for want in ("raymarching", "noise", "fractal", "2d-pattern", "post-processing"):
        assert want in effects, f"缺少 effect_type 覆盖: {want}"

    # 必须有静态（dynamic=False）与硬约束的边界样例
    assert any(not g.dynamic for g in GENERATION_GOLDENS)
    assert any(g.must_not_contain for g in GENERATION_GOLDENS)


def test_retrieval_goldens_have_negative_case():
    """必须有负样例，用于检验『宁缺毋滥』的阈值策略。"""
    from shader_agent.eval.datasets import RETRIEVAL_GOLDENS

    negatives = [g for g in RETRIEVAL_GOLDENS if g.case_id.endswith("irrelevant")]
    assert negatives, "缺少负样例"
    assert negatives[0].expected_shader_ids == []


# =====================================================================
# 报告结构
# =====================================================================

def test_eval_report_aggregate_and_markdown():
    from shader_agent.eval.runner import CaseResult, EvalReport, MetricScore

    cases = [
        CaseResult(
            case_id="c1", task="generation", ok=True, elapsed_ms=1200.0,
            trace_id="abc",
            metrics=[
                MetricScore("Compile Success", 1.0, 1.0, True, "ok"),
                MetricScore("Spec Adherence", 0.7, 0.6, True, "good", is_llm_judge=True),
            ],
        ),
        CaseResult(
            case_id="c2", task="generation", ok=True, elapsed_ms=800.0,
            metrics=[MetricScore("Compile Success", 0.0, 1.0, False, "fail")],
        ),
    ]
    report = EvalReport(started_at=0.0, elapsed_s=2.0, cases=cases,
                        config={"judge_model": "x", "langfuse": False})

    agg = report.aggregate()
    assert agg["n_cases"] == 2
    assert agg["n_passed"] == 1
    assert agg["pass_rate"] == pytest.approx(0.5)
    assert agg["by_metric"]["Compile Success"]["mean"] == pytest.approx(0.5)
    assert agg["avg_latency_ms"] == pytest.approx(1000.0)

    md = report.to_markdown()
    assert "评估报告" in md
    assert "c1" in md and "c2" in md
    assert "PASS" in md and "FAIL" in md

    d = report.to_dict()
    assert d["aggregate"]["n_cases"] == 2


def test_case_result_passed_requires_all_metrics_success():
    from shader_agent.eval.runner import CaseResult, MetricScore

    c = CaseResult(case_id="x", task="t", ok=True, elapsed_ms=1.0,
                   metrics=[MetricScore("a", 1.0, 0.5, True),
                            MetricScore("b", 0.1, 0.5, False)])
    assert c.passed is False

    c2 = CaseResult(case_id="y", task="t", ok=False, elapsed_ms=1.0, metrics=[])
    assert c2.passed is False


def test_judge_model_returns_none_without_deepeval_or_key(monkeypatch):
    """无 deepeval 或无 key 时，评审模型构造应返回 None 而非抛错。"""
    import shader_agent.eval.judge_model as jm

    monkeypatch.setattr(jm, "_deepeval_available", lambda: False)
    assert jm.build_judge_model() is None
