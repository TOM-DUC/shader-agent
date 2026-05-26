"""主题打标。

输出维度（多标签，可同时挂多个）：
  - raymarching
  - sdf
  - noise
  - fractal
  - post-processing
  - 2d-pattern
  - lighting
  - animation

设计：
- 规则版（默认）：基于关键词正则 + 启发式，覆盖 80%+ 案例，毫秒级；
- LLM 复核版（可选，settings.corpus.enable_llm_tagging=True）：
  对规则未能命中任何主题的样本，让 deepseek-chat 强制从受控词表里选 1~3 个。
"""
from __future__ import annotations

import json
import re

from shader_agent.config.settings import settings
from shader_agent.corpus.models import ShaderRecord
from shader_agent.utils.logger import logger


# 受控词表（也用于 LLM 复核）
TOPIC_VOCAB: list[str] = [
    "raymarching",
    "sdf",
    "noise",
    "fractal",
    "post-processing",
    "2d-pattern",
    "lighting",
    "animation",
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


def rule_tag(rec: ShaderRecord) -> list[str]:
    """基于规则给 rec 打主题标签（不修改 rec）。"""
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

    has_ray = bool(_RE_RAYMARCH.search(text))
    has_sdf = bool(_RE_SDF.search(text))
    has_noise = bool(_RE_NOISE.search(text))
    has_frac = bool(_RE_FRACTAL.search(text))
    has_post = bool(_RE_POST_FX.search(text))
    has_light = bool(_RE_LIGHTING.search(text))
    has_anim = bool(_RE_ANIMATION.search(text))
    has_kaleido = bool(_RE_KALEIDO.search(text))

    if has_ray:
        tags.append("raymarching")
    if has_sdf:
        tags.append("sdf")
    if has_noise:
        tags.append("noise")
    if has_frac:
        tags.append("fractal")
    if has_post:
        tags.append("post-processing")
    if has_light:
        tags.append("lighting")
    if has_anim:
        tags.append("animation")

    # 启发式：既不是 raymarching/fractal 又匹配"2D 坐标变换 + 没有 vec3 几何"则归为 2d-pattern
    # kaleidoscope 是 2d-pattern 的明确信号
    if has_kaleido or (not (has_ray or has_frac)):
        if has_kaleido or ("iResolution" in text and "vec3" not in (rec.code_image or "")[:300]):
            tags.append("2d-pattern")

    # 兜底，避免空
    if not tags:
        tags.append("2d-pattern")

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
    """给所有记录打标签（in-place 修改 tags_topic 字段）。"""
    use_llm = settings.corpus.enable_llm_tagging if use_llm is None else use_llm
    n_llm = 0
    for rec in records:
        tags = rule_tag(rec)
        # 若开启 LLM 复核且规则结果只有兜底 2d-pattern，调用 LLM 二审
        if use_llm and tags == ["2d-pattern"]:
            llm_tags = llm_tag(rec)
            if llm_tags:
                tags = llm_tags
                n_llm += 1
        rec.tags_topic = tags

    if use_llm:
        logger.info(f"[tagger] LLM re-tag fired on {n_llm} records")
    logger.info(f"[tagger] tagged {len(records)} records")
    return records
