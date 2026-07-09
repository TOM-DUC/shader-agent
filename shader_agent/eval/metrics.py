"""评估指标集合。

分两层，刻意区分开——这是本项目评估设计的核心取舍：

**第一层：确定性指标（不调用 LLM，零成本、零方差、可进 CI）**
  - CompileSuccessMetric   : 生成的 GLSL 能否通过编译/静态校验（0/1）
  - ShadertoyConventionMetric : 是否遵守 Shadertoy 硬性约束（入口签名、
                             不引外部纹理、WebGL1 兼容），按条目加权得分
  - FixLoopEfficiencyMetric: 修正轮数效率——一次成功=1.0，轮数越多分越低
  - RetrievalRelevancyMetric : 检索命中的融合分是否越过阈值、topk 命中率

**第二层：LLM-as-a-judge 指标（GEval，用 DeepSeek 作评审）**
  - spec_adherence_metric()      : 代码是否落实了 spec（effect/palette/dynamic）
  - explanation_faithfulness_metric() : 解释是否忠实于代码（不臆造未实现的细节）
  - analysis_faithfulness_metric()    : 分析报告是否忠实于源码
  - retrieval_context_relevancy_metric() : 检索到的参考与需求的相关性

为什么 shader 场景下确定性指标是主角：着色器有**客观可验证的正确性**
（能不能编译、有没有 mainImage、有没有引用 iChannel），这类事实不该交给
LLM 去猜。LLM 裁判只用在"是否体现霓虹配色""解释是否忠实"这类主观维度上。

deepeval 未安装时，本模块仍可 import；确定性指标退化为可独立调用的纯函数
（见 `score_*` 函数），LLM 指标工厂返回 None。
"""
from __future__ import annotations

import re
from typing import Any, Optional

from shader_agent.config.settings import settings


def _deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
        return True
    except Exception:
        return False


# ---------------- deepeval 基类的可降级导入 ----------------

if _deepeval_available():
    from deepeval.metrics import BaseMetric as _BaseMetric
    from deepeval.test_case import LLMTestCase as _LLMTestCase
else:  # pragma: no cover
    class _BaseMetric:  # type: ignore[no-redef]
        threshold: float = 0.5
        score: float = 0.0
        reason: str = ""
        success: bool = False
        evaluation_model: str = "deterministic"
        strict_mode: bool = False
        async_mode: bool = False
        verbose_mode: bool = False

    class _LLMTestCase:  # type: ignore[no-redef]
        pass


def _params():
    """LLMTestCaseParams 在新版被重命名为 SingleTurnParams，做双向兼容。"""
    try:
        from deepeval.test_case import LLMTestCaseParams
        return LLMTestCaseParams
    except Exception:
        from deepeval.test_case import SingleTurnParams  # type: ignore
        return SingleTurnParams


# =====================================================================
# 纯函数打分器：不依赖 deepeval，可被单测/CI/离线脚本直接调用
# =====================================================================

_SIG_RE = re.compile(
    r"\bvoid\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*in\s+vec2\s+\w+\s*\)"
)
# WebGL 1.0（GLSL ES 1.0）不存在的 ES 3.0 函数
_ES3_ONLY = ["round", "roundEven", "trunc", "textureLod", "texelFetch"]


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*[\s\S]*?\*/", "", code or "")
    return re.sub(r"//[^\n]*", "", code)


def score_shadertoy_convention(code: str) -> tuple[float, list[str]]:
    """Shadertoy 规范符合度：五项检查，返回 (0~1 分数, 违规项列表)。"""
    stripped = _strip_comments(code)
    violations: list[str] = []
    checks = 0
    passed = 0

    checks += 1
    if _SIG_RE.search(stripped):
        passed += 1
    else:
        violations.append("缺少标准 mainImage(out vec4, in vec2) 入口签名")

    checks += 1
    if not re.search(r"\bsampler2D\b|\bsamplerCube\b|\biChannel[0-9]\b", stripped):
        passed += 1
    else:
        violations.append("引用了外部纹理 / iChannelN")

    checks += 1
    if not re.search(r"\buniform\s+(?!.*\b(iResolution|iTime|iMouse)\b)", stripped):
        passed += 1
    else:
        violations.append("声明了自定义 uniform")

    checks += 1
    es3_hit = [f for f in _ES3_ONLY if re.search(rf"\b{f}\s*\(", stripped)]
    if not es3_hit:
        passed += 1
    else:
        violations.append(f"使用了 WebGL1 不支持的 ES3.0 函数: {', '.join(es3_hit)}")

    checks += 1
    if stripped.count("{") == stripped.count("}") and stripped.count("(") == stripped.count(")"):
        passed += 1
    else:
        violations.append("括号不配平")

    return (passed / checks if checks else 0.0), violations


def score_fix_loop_efficiency(iterations: int, max_loops: int = 3) -> float:
    """修正轮数效率：1 轮=1.0，之后线性衰减，超过上限记 0。

    公式： \\( s = \\max(0,\\ 1 - \\dfrac{it - 1}{\\max\\_loops}) \\)
    """
    if iterations <= 0:
        return 0.0
    if max_loops <= 0:
        return 1.0 if iterations == 1 else 0.0
    return max(0.0, 1.0 - (iterations - 1) / float(max_loops))


def score_retrieval_relevancy(
    hits: list[dict[str, Any]] | list[Any],
    min_score: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """检索相关性：越过阈值的命中占比 × 平均融合分的几何折中。

    输入可以是 dict 列表，也可以是 RetrievalHit / SimilarShader 对象。
    返回 (0~1 分数, 明细)。
    """
    thr = settings.retrieval.min_score if min_score is None else min_score
    if not hits:
        return 0.0, {"n_hits": 0, "hit_rate": 0.0, "avg_score": 0.0}

    scores: list[float] = []
    for h in hits:
        if isinstance(h, dict):
            s = h.get("fused_score")
            if s is None and "distance" in h:
                s = 1.0 - float(h["distance"] or 0.0)
        else:
            s = getattr(h, "fused_score", None)
            if s is None:
                s = 1.0 - float(getattr(h, "distance", 0.0) or 0.0)
        scores.append(float(s or 0.0))

    above = [s for s in scores if s >= thr]
    hit_rate = len(above) / len(scores)
    avg = sum(scores) / len(scores)
    # 命中率与平均分各占一半，避免"只有一条高分"或"全是低分擦线"两种极端
    final = 0.5 * hit_rate + 0.5 * min(1.0, avg)
    return final, {
        "n_hits": len(scores),
        "hit_rate": round(hit_rate, 4),
        "avg_score": round(avg, 4),
        "threshold": thr,
    }


# =====================================================================
# 第一层：确定性 deepeval 指标（BaseMetric 子类，零 LLM 调用）
# =====================================================================

class _DeterministicMetric(_BaseMetric):
    """确定性指标公共壳：同步打分，无 LLM，async 直接复用同步结果。"""

    _name = "Deterministic"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.evaluation_model = "deterministic (no LLM)"
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.evaluation_cost = 0.0   # 显式声明零成本

    # 子类实现
    def _compute(self, test_case: Any) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        self.score, self.reason = self._compute(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # deepeval 用它做展示名
        return self._name


class ShadertoyConventionMetric(_DeterministicMetric):
    """生成代码是否遵守 Shadertoy / WebGL1 硬性约束。"""

    _name = "Shadertoy Convention"

    def _compute(self, test_case: Any) -> tuple[float, str]:
        code = getattr(test_case, "actual_output", "") or ""
        score, violations = score_shadertoy_convention(code)
        reason = "全部约束通过" if not violations else "违规：" + "；".join(violations)
        return score, reason


class CompileSuccessMetric(_DeterministicMetric):
    """编译/静态校验是否通过。0/1 指标。

    从 test_case.additional_metadata["compile_ok"] 读取（由 runner 注入）。
    """

    _name = "Compile Success"

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold=threshold)

    def _compute(self, test_case: Any) -> tuple[float, str]:
        meta = getattr(test_case, "additional_metadata", None) or {}
        ok = bool(meta.get("compile_ok", False))
        if ok:
            return 1.0, "编译/校验通过"
        err = str(meta.get("compile_errors", ""))[:200]
        return 0.0, f"编译失败：{err or '未知错误'}"


class FixLoopEfficiencyMetric(_DeterministicMetric):
    """修正循环效率：越少轮次拿到可编译代码越好。"""

    _name = "Fix Loop Efficiency"

    def _compute(self, test_case: Any) -> tuple[float, str]:
        meta = getattr(test_case, "additional_metadata", None) or {}
        it = int(meta.get("iterations", 0) or 0)
        max_loops = int(meta.get("max_fix_loops", 3) or 3) + 1
        s = score_fix_loop_efficiency(it, max_loops=max_loops)
        return s, f"迭代 {it} 轮（上限 {max_loops}）"


class RetrievalRelevancyMetric(_DeterministicMetric):
    """RAG 检索相关性（基于融合分与阈值，不调 LLM）。"""

    _name = "Retrieval Relevancy"

    def _compute(self, test_case: Any) -> tuple[float, str]:
        meta = getattr(test_case, "additional_metadata", None) or {}
        hits = meta.get("retrieval_hits") or []
        s, detail = score_retrieval_relevancy(hits)
        if not hits:
            return 0.0, "未检索到任何参考（可能低于融合分阈值，属'宁缺毋滥'策略）"
        return s, (
            f"命中 {detail['n_hits']} 条，越阈值占比 {detail['hit_rate']:.2f}，"
            f"平均融合分 {detail['avg_score']:.3f}（阈值 {detail['threshold']}）"
        )


# =====================================================================
# 第二层：LLM-as-a-judge（GEval）
# =====================================================================

def _geval(name: str, steps: list[str], params: list, threshold: float,
           model: Any) -> Optional[Any]:
    if not _deepeval_available() or model is None:
        return None
    from deepeval.metrics import GEval
    return GEval(
        name=name,
        evaluation_steps=steps,   # 给定 steps 比 criteria 更可控、方差更小
        evaluation_params=params,
        threshold=threshold,
        model=model,
        async_mode=False,          # 复用同步 DeepSeek 客户端与其缓存
    )


def spec_adherence_metric(model: Any, threshold: float | None = None) -> Optional[Any]:
    """代码是否落实了需求 spec（effect_type / palette / dynamic / complexity）。"""
    P = _params()
    return _geval(
        name="Spec Adherence",
        steps=[
            "阅读 input 中的需求 spec，提取 effect_type、palette、dynamic、complexity 四项要求。",
            "阅读 actual_output 中的 GLSL 代码，判断代码是否真正实现了 effect_type "
            "所指的算法（例如 raymarching 应有步进循环，noise 应有 hash/fbm）。",
            "判断代码的主色调计算是否体现了 palette 的倾向；dynamic=true 时代码是否使用 iTime。",
            "判断代码复杂度是否与 complexity 大致相符，不要因为代码优雅或冗长而额外加减分。",
            "只依据代码本身作判断，不要假设代码之外的运行时效果。四项中满足越多分越高。",
        ],
        params=[P.INPUT, P.ACTUAL_OUTPUT],
        threshold=threshold if threshold is not None else settings.evaluation.threshold_generation,
        model=model,
    )


def explanation_faithfulness_metric(model: Any, threshold: float | None = None) -> Optional[Any]:
    """解释是否忠实于代码：不得声称代码里根本没有的实现。"""
    P = _params()
    return _geval(
        name="Explanation Faithfulness",
        steps=[
            "context 中是生成的 GLSL 代码，actual_output 是对该代码的中文说明。",
            "逐句检查说明中提到的每个技术点（算法、配色、动画方式）是否能在代码中找到对应实现。",
            "若说明声称了代码中不存在的技术或效果，视为幻觉，严重扣分。",
            "若说明遗漏了代码中的重要技术点，轻微扣分；表述简洁不扣分。",
        ],
        params=[P.ACTUAL_OUTPUT, P.CONTEXT],
        threshold=threshold if threshold is not None else settings.evaluation.threshold_generation,
        model=model,
    )


def analysis_faithfulness_metric(model: Any, threshold: float | None = None) -> Optional[Any]:
    """分析报告是否忠实于被分析的源码（Analyzer 侧核心指标）。"""
    P = _params()
    return _geval(
        name="Analysis Faithfulness",
        steps=[
            "input 是一段 GLSL 源码，actual_output 是对它的分析报告（算法摘要、技术标签、视觉效果）。",
            "检查算法摘要描述的步骤是否确实出现在源码中。",
            "检查技术标签是否有源码依据（例如标注 raymarching 就应有步进循环）。",
            "报告若捏造源码中不存在的实现细节，严重扣分；保守但准确的描述应给高分。",
        ],
        params=[P.INPUT, P.ACTUAL_OUTPUT],
        threshold=threshold if threshold is not None else settings.evaluation.threshold_analysis,
        model=model,
    )


def retrieval_context_relevancy_metric(model: Any, threshold: float | None = None) -> Optional[Any]:
    """检索到的参考样本与用户需求的相关性（RAG 检索质量的 LLM 视角）。"""
    P = _params()
    return _geval(
        name="Retrieval Context Relevancy",
        steps=[
            "input 是用户的 shader 需求，retrieval_context 是混合检索返回的参考样本。",
            "判断这些参考样本在算法、视觉风格上对完成该需求是否真的有帮助。",
            "参考样本与需求主题无关时给低分；高度相关且能提供可复用技巧时给高分。",
            "参考为空时不要给满分，应视为检索未能提供帮助。",
        ],
        params=[P.INPUT, P.RETRIEVAL_CONTEXT],
        threshold=threshold if threshold is not None else settings.evaluation.threshold_retrieval,
        model=model,
    )


# =====================================================================
# 指标套件装配
# =====================================================================

def build_generation_metrics(model: Any = None, *, with_llm_judge: bool = True) -> list[Any]:
    """Generator 任务的指标套件。"""
    metrics: list[Any] = [
        CompileSuccessMetric(threshold=1.0),
        ShadertoyConventionMetric(threshold=0.8),
        FixLoopEfficiencyMetric(threshold=0.5),
        RetrievalRelevancyMetric(threshold=settings.evaluation.threshold_retrieval),
    ]
    if with_llm_judge and model is not None:
        for m in (spec_adherence_metric(model), explanation_faithfulness_metric(model)):
            if m is not None:
                metrics.append(m)
    return metrics


def build_analysis_metrics(model: Any = None, *, with_llm_judge: bool = True) -> list[Any]:
    """Analyzer 任务的指标套件。"""
    metrics: list[Any] = [
        RetrievalRelevancyMetric(threshold=settings.evaluation.threshold_retrieval),
    ]
    if with_llm_judge and model is not None:
        m = analysis_faithfulness_metric(model)
        if m is not None:
            metrics.append(m)
    return metrics


def build_retrieval_metrics(model: Any = None, *, with_llm_judge: bool = True) -> list[Any]:
    """纯检索链路的指标套件。"""
    metrics: list[Any] = [
        RetrievalRelevancyMetric(threshold=settings.evaluation.threshold_retrieval),
    ]
    if with_llm_judge and model is not None:
        m = retrieval_context_relevancy_metric(model)
        if m is not None:
            metrics.append(m)
    return metrics
