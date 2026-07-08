"""v2 受控词表（21 类）+ ISF 分类映射。

扩展自旧版 8 类，覆盖 ISF / shaders21k / PixelFlow 等来源的技法与效果。
受控词表同时供给 tagger（关键词正则）与 rebuild_corpus（外部分类映射优先）。
"""
from __future__ import annotations

# ========== v2 受控词表 ==========

TOPIC_VOCAB_V2: list[str] = [
    # 保留的旧 8 类
    "raymarching",
    "sdf",
    "noise",
    "fractal",
    "post-processing",
    "2d-pattern",
    "lighting",
    "animation",
    # 新增 12 类（效果/技法）
    "color-grading",
    "glitch",
    "distortion",
    "blur",
    "transition",
    "stylize",
    "kaleidoscope",
    "tiling",
    "feedback",
    "geometry",
    "masking",
    "audio-reactive",
    # 诚实兜底（不再污染 2d-pattern）
    "uncategorized",
]

# 旧版词表（向后兼容，用于旧库加载时检查）
TOPIC_VOCAB_OLD: list[str] = [
    "raymarching", "sdf", "noise", "fractal",
    "post-processing", "2d-pattern", "lighting", "animation",
]

# ========== ISF CATEGORIES 到 v2 词表的映射 ==========

# ISF 文件头里的 CATEGORIES 数组值 → 我们的 tags_topic
# 一条 ISF 可能命中多个映射
ISF_CAT_TO_TAG: dict[str, list[str]] = {
    # Color Effect / Adjustment → color-grading
    "color effect":          ["color-grading"],
    "color adjustment":      ["color-grading"],
    "color":                 ["color-grading"],
    "levels":                ["color-grading"],
    "brightness/contrast":   ["color-grading"],
    "hue/saturation":        ["color-grading"],
    "color look":            ["color-grading"],
    "monochrome":            ["color-grading"],
    "sepia":                 ["color-grading"],
    "color conversion":      ["color-grading"],
    "colorbars":             ["color-grading"],
    "colorbars ":            ["color-grading"],
    # Glitch → glitch
    "glitch":                ["glitch"],
    "bad tv":                ["glitch"],
    # Distortion → distortion
    "distortion effect":     ["distortion"],
    "warp":                  ["distortion"],
    "bulge":                 ["distortion"],
    "pinch":                 ["distortion"],
    "twirl":                 ["distortion"],
    "ripple":                ["distortion"],
    "water":                 ["distortion"],
    "waves":                 ["distortion"],
    "lens":                  ["distortion"],
    "magnifier":             ["distortion"],
    "mirror":                ["distortion"],
    "reflection":            ["distortion"],
    # Blur → blur
    "blur":                  ["blur"],
    "defocus":               ["blur"],
    "dof":                   ["blur"],
    "depth of field":        ["blur"],
    "zoom blur":             ["blur"],
    "motion blur":           ["blur"],
    # Wipe / Dissolve → transition
    "wipe":                  ["transition"],
    "dissolve":              ["transition"],
    "transition":            ["transition"],
    "directional wipe":      ["transition"],
    "radial wipe":           ["transition"],
    # Stylize → stylize
    "stylize":               ["stylize"],
    "halftone":              ["stylize"],
    "ascii art":             ["stylize"],
    "ascii":                 ["stylize"],
    "retro":                 ["stylize"],
    "film":                  ["stylize"],
    "film grain":            ["stylize"],
    "scanlines":             ["stylize"],
    "posterize":             ["stylize"],
    "cartoon":               ["stylize"],
    "edge":                  ["stylize"],
    "edge detect":           ["stylize"],
    "emboss":                ["stylize"],
    "pixelate":              ["stylize"],
    "mosaic":                ["stylize"],
    "pixellate":             ["stylize"],
    "crosshatch":            ["stylize"],
    # Kaleidoscope → kaleidoscope
    "kaleidoscope":          ["kaleidoscope"],
    # Tile / Pattern → tiling
    "tile":                  ["tiling"],
    "truchet":               ["tiling"],
    "pattern":               ["tiling"],
    "brick":                 ["tiling"],
    # Feedback → feedback
    "feedback":              ["feedback"],
    "delay":                 ["feedback"],
    # Geometry → geometry
    "geometry":              ["geometry"],
    "rotation":              ["geometry"],
    "transform":             ["geometry"],
    "flip":                  ["geometry"],
    "crop":                  ["geometry"],
    "resize":                ["geometry"],
    "scale":                 ["geometry"],
    # Mask / Overlay → masking
    "mask":                  ["masking"],
    "masking":               ["masking"],
    "overlay":               ["masking"],
    "blend":                 ["masking"],
    "composite":             ["masking"],
    "alpha":                 ["masking"],
    "opacity":               ["masking"],
    "keying":                ["masking"],
    "chroma key":            ["masking"],
    "luma key":              ["masking"],
    # Audio → audio-reactive
    "audio":                 ["audio-reactive"],
    "audio visualizer":      ["audio-reactive"],
    "spectrum":              ["audio-reactive"],
    "waveform":              ["audio-reactive"],
}

# ISF CATEGORIES → 常规 tag 的备选（映射到旧 8 类的 subset）
# 仅在 ISF_CAT_TO_TAG 未命中时尝试
ISF_CAT_FALLBACK_TAG: dict[str, list[str]] = {
    "generator":               ["2d-pattern"],
    "3d":                       ["raymarching"],
    "light":                    ["lighting"],
    "noise":                    ["noise"],
}

# ========== 辅助函数 ==========


def isf_categories_to_tags(isf_categories: list[str]) -> list[str]:
    """将 ISF 文件头的 CATEGORIES 数组映射为 v2 tags_topic。

    映射策略（强→弱）：
    1. 精确匹配 ISF_CAT_TO_TAG（大小写不敏感）；
    2. 备选 ISF_CAT_FALLBACK_TAG；
    3. 全无命中则返回空列表（交由 rule_tag 关键词正则兜底）。
    """
    tags: list[str] = []
    seen: set[str] = set()
    for cat_raw in isf_categories:
        cat = cat_raw.strip().lower()
        # 精确匹配
        if cat in ISF_CAT_TO_TAG:
            for t in ISF_CAT_TO_TAG[cat]:
                if t not in seen:
                    tags.append(t)
                    seen.add(t)
        # 备选匹配
        elif cat in ISF_CAT_FALLBACK_TAG:
            for t in ISF_CAT_FALLBACK_TAG[cat]:
                if t not in seen:
                    tags.append(t)
                    seen.add(t)
    return tags
