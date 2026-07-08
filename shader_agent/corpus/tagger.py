"""主题打标（v2）。

输出维度（多标签，可同时挂多个）—— 21 类受控词表：

旧 8 类：raymarching / sdf / noise / fractal / post-processing / 2d-pattern / lighting / animation
新增 12 类：color-grading / glitch / distortion / blur / transition / stylize / kaleidoscope / tiling / feedback / geometry / masking / audio-reactive
兜底：uncategorized

打标顺序（强→弱）：
1. ISF/s21k 外部分类映射（由 `isf_categories_to_tags` 在 load 阶段通过 categories 传入）；
2. 代码关键词正则；
3. ``2d-pattern`` 仅在有正向 2D 信号时打（不再用排除法兜底）；
4. 全空则落 ``uncategorized``。

设计：
- 规则版（默认）：基于关键词正则 + 启发式，覆盖 80%+ 案例，毫秒级；
- LLM 复核版（可选，settings.corpus.enable_llm_tagging=True）：
   对规则结果为空且 categories 也为空的样本，让 deepseek-chat 从受控词表选标签。
"""
from __future__ import annotations

import json
import re

from shader_agent.config.settings import settings
from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger


# v2 受控词表
TOPIC_VOCAB: list[str] = [
    "raymarching", "sdf", "noise", "fractal",
    "post-processing", "2d-pattern", "lighting", "animation",
    "color-grading", "glitch", "distortion", "blur",
    "transition", "stylize", "kaleidoscope", "tiling",
    "feedback", "geometry", "masking", "audio-reactive",
    "uncategorized",
]


# ---------------- 规则匹配 ----------------

_RE_RAYMARCH = re.compile(
    r"\b(raymarch|ray\s*march|marching|"
    r"(\bro\b\s*\+\s*\brd\b)|"  # ro + rd * t 典型句式
    r"\bt\s*\+=\s*[a-zA-Z_])",
    re.IGNORECASE,
)
_RE_SDF = re.compile(
    r"\b(sd[A-Z]\w*|signed\s*distance|distance\s+function|"
    r"smin|smax|opUnion|opSubtraction|opIntersection)",
)
_RE_NOISE = re.compile(
    r"\b(noise|snoise|cnoise|valueNoise|fbm|simplex|"
    r"hash\d?|fract\s*\(\s*sin)",
    re.IGNORECASE,
)
_RE_FRACTAL = re.compile(
    r"\b(mandelbrot|julia|fractal|"
    r"mengersponge|sierpinski|IFS\b|"
    r"orbit\s*trap|escape[-_ ]?time)",
    re.IGNORECASE,
)
# kaleidoscope 单独识别为 2d-pattern 的强信号
_RE_KALEIDO = re.compile(r"\bkaleidoscop", re.IGNORECASE)
_RE_POST_FX = re.compile(
    r"\b(bloom|vignette|chromatic|aberration|tonemap|fxaa|"
    r"blur|kernel|gamma\s*correct|postpro|post-pro)",
    re.IGNORECASE,
)
_RE_LIGHTING = re.compile(
    r"\b(diffuse|specular|phong|blinn|cook[-_ ]?torrance|"
    r"pbr|reflect|refract|shadow|ambient\s*occlusion|"
    r"calcNormal|normalize\s*\(\s*cross)",
    re.IGNORECASE,
)
_RE_ANIMATION = re.compile(r"\biTime\b")
# 2d-pattern 正向信号：平铺 / 棋盘 / 网格 / 显式图案关键词。
# 不再匹配裸 `uv` 或 `fragCoord/iResolution`——它们几乎出现在每个 shader 里，会把
# 2d-pattern 变成垃圾桶标签。
_RE_2D_PATTERN = re.compile(
    r"\b(tile|tiling|truchet|checker|checkerboard|grid|"
    r"voronoi|hex\s*grid|polar\s*coord|kaleidoscop)\b",
    re.IGNORECASE,
)


def rule_tag(rec: ShaderRecord) -> list[str]:
    """基于规则给 rec 打主题标签（不修改 rec）。

    打标优先顺序：
    1. 外部已映射的 tags_topic（来自 ISF/load 阶段的 categories 映射）——跳过关键词正则；
    2. 关键词正则追加；
    3. ``2d-pattern`` 仅在有正向 2D 信号时打（不再用排除法兜底）；
    4. 全空则落 ``uncategorized``。
    """
    text = "\n".join(
        [
            rec.name or "",
            rec.description or "",
            " ".join(rec.tags_raw or []),
            rec.code_image or "",
            rec.code_common or "",
        ]
    )

    tags: list[str] = []

    # 步骤 1：优先保留外部已映射的标签（来自 ISF / shaders21k 的 categories 映射）
    # 这些映射由 category_map.isf_categories_to_tags 在 load 阶段通过 tags_topic 传入
    if rec.tags_topic:
        # 去掉可能存在的旧 v1 标签（由更精确的映射取代）
        pass  # tags_topic 已由 load 函数写入，保留之
        tags.extend(rec.tags_topic)

    # 步骤 2：关键词正则追加（不重复添加已有标签）
    has_ray = bool(_RE_RAYMARCH.search(text))
    has_sdf = bool(_RE_SDF.search(text))
    has_noise = bool(_RE_NOISE.search(text))
    has_frac = bool(_RE_FRACTAL.search(text))
    has_post = bool(_RE_POST_FX.search(text))
    has_light = bool(_RE_LIGHTING.search(text))
    has_anim = bool(_RE_ANIMATION.search(text))
    has_kaleido = bool(_RE_KALEIDO.search(text))

    regex_map = {
        "raymarching": has_ray,
        "sdf": has_sdf,
        "noise": has_noise,
        "fractal": has_frac,
        "post-processing": has_post,
        "lighting": has_light,
        "animation": has_anim,
        "kaleidoscope": has_kaleido,
    }
    for tag_name, flag in regex_map.items():
        if flag and tag_name not in tags:
            tags.append(tag_name)

    # 步骤 3：2d-pattern —— 仅在有明确的平铺/图案信号时打。
    # 不使用"含 iResolution 且开头没有 vec3 就算 2D"这类排除法：它会把绝大多数
    # shader 误判为 2d-pattern，正是标签分布失衡的根源。
    has_2d_signal = bool(has_kaleido or _RE_2D_PATTERN.search(text))

    if has_2d_signal and "2d-pattern" not in tags:
        tags.append("2d-pattern")

    # 步骤 4：全空才落 uncategorized
    if not tags:
        tags.append("uncategorized")

    # 去重保序
    seen = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------- LLM 复核（可选） ----------------

_LLM_SYSTEM = (
    "You are a GLSL shader topic classifier. "
    "Given a shader's name, description, and code, choose 1~3 topics "
    "STRICTLY from this fixed vocabulary: "
    f"{TOPIC_VOCAB}. "
    "Output ONLY a JSON array of lowercase strings, e.g. "
    '["raymarching","sdf"]. No prose, no markdown.'
)


def llm_tag(rec: ShaderRecord) -> list[str]:
    """让 LLM 在受控词表中挑标签。失败时返回空列表，调用方决定是否回退。"""
    # 延迟导入：tagger 仅在 enable_llm_tagging 时依赖 LLM
    from shader_agent.llm.deepseek_client import deepseek

    user = (
        f"Name: {rec.name}\n"
        f"Description: {rec.description}\n"
        f"Raw tags: {rec.tags_raw}\n"
        f"Code (truncated):\n{(rec.code_image or '')[:1800]}"
    )
    try:
        text = deepseek.chat(
            [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=128,
        )
    except Exception as e:
        logger.warning(f"[tagger] llm_tag failed for {rec.shader_id}: {e}")
        return []

    text = text.strip()
    # 容错：去掉常见 markdown 包裹
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            return []
        result: list[str] = []
        for x in arr:
            if isinstance(x, str) and x.lower() in TOPIC_VOCAB:
                result.append(x.lower())
        return result
    except Exception as e:
        logger.warning(f"[tagger] llm parse failed for {rec.shader_id}: {e} raw={text!r}")
        return []


# ---------------- 入口 ----------------

def tag_records(
    records: list[ShaderRecord],
    *,
    use_llm: bool | None = None,
) -> list[ShaderRecord]:
    """给所有记录打标签（in-place 修改 tags_topic 字段）。

    v2 打标策略：
    1. 对已有外部分类映射的记录（ISF），保留映射标签并用关键词正则补充；
    2. 对无外部分类映射的记录，纯关键词正则；
    3. LLM 复核仅在规则结果全空（uncategorized）时可选触发。
    """
    use_llm = settings.corpus.enable_llm_tagging if use_llm is None else use_llm
    n_llm = 0
    for rec in records:
        # 如果记录已有 tags_topic（来自 isf_loader 的 categories 映射），作为基础
        # 否则用纯规则
        tags = rule_tag(rec)
        # LLM 复核：仅当规则落在 uncategorized 且启用时
        if use_llm and tags == ["uncategorized"]:
            llm_tags = llm_tag(rec)
            if llm_tags:
                tags = llm_tags
                n_llm += 1
        rec.tags_topic = tags

    if use_llm:
        logger.info(f"[tagger] LLM re-tag fired on {n_llm} records")
    logger.info(f"[tagger] tagged {len(records)} records")
    return records
