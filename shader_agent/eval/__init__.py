"""离线评估子系统（DeepEval）。

两层指标：
  - 确定性指标（零 LLM 成本）：编译通过率、Shadertoy 规范符合度、修正轮数效率、
    检索相关性、负样例拒答率。可直接进 CI。
  - LLM-as-a-judge（GEval + DeepSeek 评审）：spec 遵循度、解释忠实度、
    分析忠实度、检索上下文相关性。

评估结果可回流到 Langfuse（按 trace_id 挂分），实现"可观测性 ↔ 评估"闭环。

用法：
    python -m scripts.run_eval --tasks retrieval,generation --no-judge
"""
from shader_agent.eval.datasets import (
    ANALYSIS_GOLDENS,
    GENERATION_GOLDENS,
    RETRIEVAL_GOLDENS,
    summary,
)
from shader_agent.eval.judge_model import DeepSeekJudge, build_judge_model
from shader_agent.eval.metrics import (
    CompileSuccessMetric,
    FixLoopEfficiencyMetric,
    RetrievalRelevancyMetric,
    ShadertoyConventionMetric,
    build_analysis_metrics,
    build_generation_metrics,
    build_retrieval_metrics,
    score_fix_loop_efficiency,
    score_retrieval_relevancy,
    score_shadertoy_convention,
)
from shader_agent.eval.runner import (
    CaseResult,
    EvalReport,
    EvalRunner,
    MetricScore,
    save_report,
)

__all__ = [
    "ANALYSIS_GOLDENS",
    "GENERATION_GOLDENS",
    "RETRIEVAL_GOLDENS",
    "summary",
    "DeepSeekJudge",
    "build_judge_model",
    "CompileSuccessMetric",
    "FixLoopEfficiencyMetric",
    "RetrievalRelevancyMetric",
    "ShadertoyConventionMetric",
    "build_analysis_metrics",
    "build_generation_metrics",
    "build_retrieval_metrics",
    "score_fix_loop_efficiency",
    "score_retrieval_relevancy",
    "score_shadertoy_convention",
    "CaseResult",
    "EvalReport",
    "EvalRunner",
    "MetricScore",
    "save_report",
]
