"""评估用黄金数据集（goldens）。

刻意保持"小而精"：评估集不是越大越好，而是要覆盖**能力边界**——
每条 golden 对应一类明确的失败模式，跑一遍就能定位是检索、生成还是解释出了问题。

三类：
  - GENERATION_GOLDENS : 覆盖 raymarching / noise / fractal / 2d-pattern /
                         post-processing 五种 effect_type，以及"静态"与"约束"两种边界。
  - ANALYSIS_GOLDENS   : 用内置 seed shader 作为源码，检验分析忠实度。
  - RETRIEVAL_GOLDENS  : 只考检索，带 expected_tags 用于计算标签命中。

数据集完全离线内置，不依赖网络与外部标注，`pytest` 与 CI 都能直接跑。
需要扩充时，把新 case 追加到列表即可，无需改动 runner。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationGolden:
    """一条生成任务的评估样例。"""
    case_id: str
    prompt: str                                  # 用户自然语言需求
    effect_type: str = ""                        # 期望的 effect_type
    palette: str = ""
    dynamic: bool = True
    complexity: str = "simple"
    expected_tags: list[str] = field(default_factory=list)
    # 期望代码中出现/不出现的标识（确定性断言，便于快速定位回归）
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class AnalysisGolden:
    """一条分析任务的评估样例。"""
    case_id: str
    seed_id: str = ""            # 从 seed_shaders 取源码
    code: str = ""               # 或直接给源码
    expected_techniques: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RetrievalGolden:
    """一条检索任务的评估样例。"""
    case_id: str
    query: str
    want_tags: list[str] = field(default_factory=list)
    # 期望能被召回的 shader_id（可为空；为空时只考融合分与阈值）
    expected_shader_ids: list[str] = field(default_factory=list)
    note: str = ""


# =====================================================================
# 生成任务 goldens
# =====================================================================

GENERATION_GOLDENS: list[GenerationGolden] = [
    GenerationGolden(
        case_id="gen_raymarch_sphere",
        prompt="用 raymarching 渲染一个球体，冷蓝色调，带缓慢的时间动画",
        effect_type="raymarching",
        palette="cool blue",
        dynamic=True,
        complexity="simple",
        expected_tags=["raymarching", "sdf"],
        must_contain=["mainImage", "iTime"],
        must_not_contain=["iChannel0", "sampler2D"],
        note="最基础的 raymarching 路径，检验步进循环与动画",
    ),
    GenerationGolden(
        case_id="gen_noise_water",
        prompt="用 fbm 噪声画一个水波纹效果，简单，蓝绿色",
        effect_type="noise",
        palette="cool blue",
        dynamic=True,
        complexity="simple",
        expected_tags=["noise"],
        must_contain=["mainImage", "iTime"],
        must_not_contain=["iChannel0"],
        note="检验噪声/hash 函数的自包含实现（不得依赖纹理查表）",
    ),
    GenerationGolden(
        case_id="gen_neon_kaleido",
        prompt="画一个程序生成的霓虹蓝紫万花筒，带时间动画，6 折对称",
        effect_type="2d-pattern",
        palette="neon",
        dynamic=True,
        complexity="moderate",
        expected_tags=["2d-pattern"],
        must_contain=["mainImage", "iTime"],
        note="检验极坐标域重复与配色遵循度",
    ),
    GenerationGolden(
        case_id="gen_mandelbrot_static",
        prompt="画一个静态的 Mandelbrot 分形，单色，不要动画",
        effect_type="fractal",
        palette="monochrome",
        dynamic=False,
        complexity="simple",
        expected_tags=["fractal"],
        must_contain=["mainImage"],
        note="边界样例：dynamic=false，代码不应依赖 iTime 产生主要效果",
    ),
    GenerationGolden(
        case_id="gen_post_vignette",
        prompt="做一个后处理风格的暗角 + 扫描线效果，暖色调，不要用纹理",
        effect_type="post-processing",
        palette="warm sunset",
        dynamic=True,
        complexity="minimal",
        expected_tags=["post-processing"],
        must_contain=["mainImage"],
        must_not_contain=["iChannel0", "sampler2D", "texture("],
        note="边界样例：显式硬约束『不要用纹理』，检验约束遵循",
    ),
]


# =====================================================================
# 分析任务 goldens（源码取自内置 seed shaders，完全离线）
# =====================================================================

ANALYSIS_GOLDENS: list[AnalysisGolden] = [
    AnalysisGolden(
        case_id="ana_seed_raymarch",
        seed_id="seed03",
        expected_techniques=["raymarching"],
        note="经典 raymarched sphere，分析应识别步进循环与法线求解",
    ),
    AnalysisGolden(
        case_id="ana_seed_first",
        seed_id="seed01",
        note="第一条 seed，作为基线；只考忠实度不考特定标签",
    ),
]


# =====================================================================
# 检索任务 goldens
# =====================================================================

RETRIEVAL_GOLDENS: list[RetrievalGolden] = [
    RetrievalGolden(
        case_id="ret_normal_calc",
        query="如何在 raymarching 中计算表面法线",
        want_tags=["raymarching"],
        note="子块级检索的典型收益场景：应命中 calcNormal 类函数块",
    ),
    RetrievalGolden(
        case_id="ret_sdf_smin",
        query="两个球体的平滑融合 smin 距离函数",
        want_tags=["sdf"],
        note="检验 GLSL 标识符分词（smin / sdSphere）的关键词召回",
    ),
    RetrievalGolden(
        case_id="ret_fbm_noise",
        query="fbm 分形布朗运动噪声实现",
        want_tags=["noise"],
        note="检验驼峰/缩写词的 BM25 命中",
    ),
    RetrievalGolden(
        case_id="ret_irrelevant",
        query="如何用 Python 读取 CSV 文件并做数据透视",
        want_tags=[],
        expected_shader_ids=[],
        note="负样例：与语料完全无关，理想行为是**不返回**参考（阈值生效）",
    ),
]


def resolve_analysis_code(golden: AnalysisGolden) -> str:
    """把 AnalysisGolden 解析成实际源码（seed_id 优先）。"""
    if golden.code.strip():
        return golden.code
    if not golden.seed_id:
        return ""
    try:
        from shader_agent.corpus.seed_shaders import get_seed_shaders
        for s in get_seed_shaders():
            if s.shader_id == golden.seed_id:
                return s.code_image or ""
    except Exception:
        pass
    return ""


def summary() -> dict[str, Any]:
    """数据集规模概览，供报告头部展示。"""
    return {
        "generation": len(GENERATION_GOLDENS),
        "analysis": len(ANALYSIS_GOLDENS),
        "retrieval": len(RETRIEVAL_GOLDENS),
        "total": len(GENERATION_GOLDENS) + len(ANALYSIS_GOLDENS) + len(RETRIEVAL_GOLDENS),
    }
