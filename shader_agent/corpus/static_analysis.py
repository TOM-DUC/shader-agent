"""静态分析与质量评分。

在入库前对每条 shader 做轻量静态分析，产出：

- key_functions：关键自定义函数名（用于父子分块与函数级检索）；
- algorithm_summary：基于结构特征的启发式算法摘要（无 LLM 也能用）；
- visual_features：从标签与代码特征推断的视觉关键词；
- quality_score：综合质量分，用于决定是否进入高质量参考库、以及检索融合排序。

设计成无外部依赖、毫秒级。真实编译/渲染验证由调用方在外部完成后回填
``compile_ok`` / ``render_ok``，本模块会把它们纳入质量分。
"""
from __future__ import annotations

import re

from shader_agent.corpus.chunker import _extract_functions
from shader_agent.corpus.models import ShaderRecord

# 视觉特征关键词命中表：标签或代码命中即附加
_VISUAL_HINTS: dict[str, list[str]] = {
    "glow": ["glow", "bloom", "neon"],
    "smooth": ["smoothstep", "smin", "smax"],
    "metallic": ["reflect", "fresnel", "specular"],
    "organic": ["fbm", "noise", "domain warp", "turbulence"],
    "geometric": ["sdbox", "sdsphere", "sdf", "kaleidoscope"],
    "volumetric": ["raymarch", "march", "density", "fog"],
}


def extract_key_functions(code: str, limit: int = 8) -> list[str]:
    """抽取关键自定义函数名（去掉 mainImage 入口本身）。"""
    names: list[str] = []
    for name, _sig, _body in _extract_functions(code):
        if name == "mainImage":
            continue
        if name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def infer_visual_features(rec: ShaderRecord) -> list[str]:
    """从标签 + 代码推断视觉特征关键词。"""
    haystack = (
        " ".join(rec.tags_topic or [])
        + " "
        + (rec.code_image or "")
    ).lower()
    feats: list[str] = []
    for feature, hints in _VISUAL_HINTS.items():
        if any(h in haystack for h in hints):
            feats.append(feature)
    return feats


def build_algorithm_summary(rec: ShaderRecord, key_funcs: list[str]) -> str:
    """基于结构特征拼一段简短算法摘要（启发式，无 LLM）。"""
    code = rec.code_image or ""
    loc = len(code.splitlines())
    tags = ", ".join(rec.tags_topic or []) or "未分类"
    func_part = (
        f"关键函数 {', '.join(key_funcs[:5])}" if key_funcs else "无显著自定义函数"
    )
    technique = "光线步进" if re.search(r"raymarch|march", code, re.I) else (
        "符号距离场" if re.search(r"\bsd[A-Z]", code) else "屏幕空间着色"
    )
    return (
        f"该 shader 约 {loc} 行，主题标签为 {tags}，主要采用{technique}思路，"
        f"{func_part}。"
    )


def _compile_any_entry(code: str) -> bool:
    """自适应检查 shader 是否含有可编译入口。

    兼容两种格式：
    - Shadertoy 风格：``mainImage(out vec4 fragColor, in vec2 fragCoord)``
    - GLSL 裸风格：``void main()`` + ``gl_FragColor``
    - 通用：只要存在 main 或 mainImage 且至少有颜色输出
    """
    has_main_image = "mainImage" in code and ("fragColor" in code or "gl_FragColor" in code)
    has_main = "void main" in code and ("gl_FragColor" in code or "fragColor" in code)
    return has_main_image or has_main


def score_quality(rec: ShaderRecord) -> float:
    """综合质量评分（0~1）。

    维度与权重：
    - 编译通过 0.40：兼容 Shadertoy ``fragColor`` 与 GLSL ``gl_FragColor`` 两种入口写法；
    - 渲染成功 0.15：真实渲染出非空帧；
    - 结构完整 0.10：有入口且至少一个自定义函数；
    - 受欢迎度 0.10：likes 经对数压缩（ISF/shaders21k 无 likes 时该项为 0）；
    - 标签覆盖 0.15：命中受控主题词表越多越利于检索；
    - 代码行数 0.10：过短或过长都扣分；
    - 干净度 0.10：``reference_only=False`` 且 ``has_external_assets=False`` 加分。
    """
    import math

    score = 0.0
    if rec.compile_ok:
        score += 0.40
    if rec.render_ok:
        score += 0.15

    code = rec.code_image or ""
    has_entry = _compile_any_entry(code)
    func_count = len(extract_key_functions(code))
    if has_entry and func_count >= 1:
        score += 0.10
    elif has_entry:
        score += 0.05

    # likes 对数压缩（ISF/s21k 可能无 likes）
    likes = max(int(rec.likes or 0), 0)
    pop = math.log10(likes + 1) / 5.0
    score += 0.10 * min(pop, 1.0)

    # 标签覆盖：v2 词表 21 类，max 取 4 类
    tag_cov = min(len(rec.tags_topic or []), 4) / 4.0
    score += 0.15 * tag_cov

    # 代码行数：50~500 行最佳，太短或太长扣分
    loc = len(code.splitlines())
    if 50 <= loc <= 500:
        score += 0.10
    elif 20 <= loc < 50 or 500 < loc <= 800:
        score += 0.05

    # 干净度：非参考 + 无外部资源
    if not rec.reference_only and not rec.has_external_assets:
        score += 0.10

    return round(min(score, 1.0), 4)


def analyze_record(rec: ShaderRecord) -> ShaderRecord:
    """对单条记录做静态分析并回填字段（in-place）。

    注意：``compile_ok`` / ``render_ok`` 若已被外部验证回填，则保留；否则按
    "代码含 mainImage 入口"做一个保守的静态可编译近似。
    """
    code = rec.code_image or ""
    rec.key_functions = extract_key_functions(code)
    rec.visual_features = infer_visual_features(rec)
    rec.algorithm_summary = build_algorithm_summary(rec, rec.key_functions)

    # 未做真实编译时，用格式自适应的入口检查作为保守近似
    if not rec.compile_ok:
        rec.compile_ok = _compile_any_entry(code)

    rec.quality_score = score_quality(rec)
    rec.mark_indexed()
    return rec


def analyze_records(records: list[ShaderRecord]) -> list[ShaderRecord]:
    """批量静态分析。"""
    for rec in records:
        analyze_record(rec)
    return records
