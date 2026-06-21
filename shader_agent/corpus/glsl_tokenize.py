"""面向 GLSL 的分词器。

通用英文分词器把 ``sdSphere``、``calcNormal``、``iResolution`` 当成单个不可分的
词，导致关键词检索几乎无法命中函数名片段。这里针对 GLSL 标识符做拆分：

- 驼峰拆分：``sdSphere`` -> ``sd`` ``sphere``，``calcNormal`` -> ``calc`` ``normal``；
- 下划线拆分：``map_scene`` -> ``map`` ``scene``；
- 同时保留原始标识符（``sdsphere``），让"整词命中"与"片段命中"都能加分；
- 保留 Shadertoy 内置变量（``iresolution`` / ``itime`` 等）作为强信号词；
- 过滤 GLSL 关键字与极短噪声 token。

输出全部小写，供 BM25 直接消费。
"""
from __future__ import annotations

import re

# GLSL 关键字与基础类型，作为停用词过滤掉（检索价值低、出现频率高）
_GLSL_STOPWORDS: frozenset[str] = frozenset(
    {
        "void", "float", "int", "bool", "vec2", "vec3", "vec4",
        "mat2", "mat3", "mat4", "ivec2", "ivec3", "ivec4",
        "if", "else", "for", "while", "do", "return", "break", "continue",
        "in", "out", "inout", "const", "uniform", "varying", "attribute",
        "true", "false", "discard", "struct", "precision", "highp", "mediump",
        "lowp", "sampler2d", "samplercube",
    }
)

# 匹配一个完整标识符（字母/下划线开头）
_RE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# 驼峰边界：小写/数字 后接 大写，或 连续大写后接 大写+小写
_RE_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifier(token: str) -> list[str]:
    """把一个标识符拆成子词，并保留原词。"""
    parts: list[str] = []
    # 先按下划线拆
    for chunk in token.split("_"):
        if not chunk:
            continue
        # 再按驼峰拆
        for piece in _RE_CAMEL.split(chunk):
            if piece:
                parts.append(piece.lower())
    out: list[str] = []
    # 原始整词（小写）也作为一个 token，保证整词检索仍然有效
    whole = token.replace("_", "").lower()
    if whole:
        out.append(whole)
    out.extend(parts)
    return out


def tokenize_glsl(text: str) -> list[str]:
    """把 GLSL 代码 / 自然语言文本切成检索 token 列表。"""
    if not text:
        return []
    tokens: list[str] = []
    for m in _RE_IDENT.finditer(text):
        ident = m.group(0)
        for tok in _split_identifier(ident):
            if len(tok) < 2:
                continue
            if tok in _GLSL_STOPWORDS:
                continue
            tokens.append(tok)
    return tokens
