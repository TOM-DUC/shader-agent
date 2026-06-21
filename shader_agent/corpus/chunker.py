"""GLSL 分块器：把一条 shader 切成可独立检索的父子知识块。

动机：一个 shader 整体向量化（且只取代码前若干字符）会丢失大量结构信息——
查询"如何求法线"时，命中的是整条 shader 的平均语义，而不是其中的 ``calcNormal``
函数。把 shader 拆成函数级子块后，检索可以精确命中"具体怎么实现某个能力"。

父子关系：
- 子块（child）用于检索，粒度细，每块语义聚焦：overview / structure / algorithm /
  每个自定义函数 / 完整代码摘要；
- 父文档（parent）即原 shader，命中任一子块后可回溯到完整代码、来源、质量等。

切分策略（纯静态，无 LLM、无外部依赖）：
1. 用花括号配对扫描出顶层函数体（处理嵌套大括号）；
2. 抽取函数签名（返回类型 + 函数名 + 形参）与函数体；
3. 额外生成 overview / structure 两类元信息块，承载名称、标签、内置变量等。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from shader_agent.corpus.models import ShaderRecord

# 匹配函数头："<ret_type> <name> ( ... )" 紧跟 "{"
_RE_FUNC_HEAD = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{]*)\)\s*\{"
)
# 非函数定义的控制关键字，避免把 if/for 误判为函数
_NON_FUNC_HEADS = {"if", "for", "while", "switch", "return", "else"}


@dataclass
class ShaderChunk:
    """一个可检索的子块。

    chunk_id 形如 ``<shader_id>::<kind>`` 或 ``<shader_id>::fn::<func_name>``。
    parent_id 始终指向所属 shader 的 shader_id。
    """

    chunk_id: str
    parent_id: str
    kind: str  # overview / structure / algorithm / function / full_code
    title: str  # 函数名或块标题
    text: str  # 喂给嵌入与 BM25 的检索文本
    meta: dict = field(default_factory=dict)


def _extract_functions(code: str) -> list[tuple[str, str, str]]:
    """返回 [(func_name, signature, body_with_head), ...]。

    用括号配对处理嵌套，保证函数体完整。
    """
    funcs: list[tuple[str, str, str]] = []
    for m in _RE_FUNC_HEAD.finditer(code):
        ret_t, name = m.group(1), m.group(2)
        if ret_t in _NON_FUNC_HEADS or name in _NON_FUNC_HEADS:
            continue
        params = (m.group(3) or "").strip()
        brace_start = m.end() - 1  # 指向 "{"
        depth = 0
        i = brace_start
        while i < len(code):
            ch = code[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = code[m.start(): i + 1]
        signature = f"{ret_t} {name}({params})"
        funcs.append((name, signature, body))
    return funcs


def chunk_shader(rec: ShaderRecord) -> list[ShaderChunk]:
    """把一条 ShaderRecord 切成父子知识块列表。"""
    sid = rec.shader_id
    code = rec.code_image or ""
    chunks: list[ShaderChunk] = []

    tags = list(getattr(rec, "tags_topic", []) or [])
    visual = ", ".join(getattr(rec, "visual_features", []) or [])
    algo = getattr(rec, "algorithm_summary", "") or ""
    key_funcs = list(getattr(rec, "key_functions", []) or [])

    # overview：名称 / 作者 / 描述 / 标签 / 视觉特征，回答"这是个什么效果"
    overview_parts = [f"Name: {rec.name}"]
    if rec.username:
        overview_parts.append(f"Author: {rec.username}")
    if rec.description:
        overview_parts.append(f"Description: {rec.description}")
    if tags:
        overview_parts.append(f"Topics: {', '.join(tags)}")
    if visual:
        overview_parts.append(f"Visual features: {visual}")
    chunks.append(
        ShaderChunk(
            chunk_id=f"{sid}::overview",
            parent_id=sid,
            kind="overview",
            title="overview",
            text="\n".join(overview_parts),
            meta={"tags_topic": ",".join(tags)},
        )
    )

    # structure：入口 / 自定义函数清单 / 内置变量，回答"代码怎么组织"
    funcs = _extract_functions(code)
    func_names = [f[0] for f in funcs]
    structure_lines = [
        f"Entry: {'mainImage present' if 'mainImage' in code else 'unknown'}",
        f"Functions: {', '.join(func_names) if func_names else 'none'}",
    ]
    builtins = [b for b in ("iTime", "iResolution", "iMouse", "iFrame") if b in code]
    if builtins:
        structure_lines.append(f"Builtins: {', '.join(builtins)}")
    chunks.append(
        ShaderChunk(
            chunk_id=f"{sid}::structure",
            parent_id=sid,
            kind="structure",
            title="structure",
            text="\n".join(structure_lines),
            meta={"function_count": len(func_names)},
        )
    )

    # algorithm：算法摘要（建库时静态/可选 LLM 产出），回答"用了什么算法"
    if algo:
        chunks.append(
            ShaderChunk(
                chunk_id=f"{sid}::algorithm",
                parent_id=sid,
                kind="algorithm",
                title="algorithm",
                text=algo,
                meta={"tags_topic": ",".join(tags)},
            )
        )

    # function：每个自定义函数一个子块，回答"某个能力怎么实现"
    seen_names: dict[str, int] = {}
    for name, signature, body in funcs:
        cnt = seen_names.get(name, 0)
        seen_names[name] = cnt + 1
        suffix = f"_{cnt}" if cnt else ""
        # 函数体过长则截断，保留签名 + 主体前若干行
        snippet = body if len(body) <= 1200 else (body[:1200] + "\n// ...")
        text = f"Function {signature}\n{snippet}"
        chunks.append(
            ShaderChunk(
                chunk_id=f"{sid}::fn::{name}{suffix}",
                parent_id=sid,
                kind="function",
                title=name,
                text=text,
                meta={
                    "function_name": name,
                    "is_key": name in key_funcs,
                },
            )
        )

    # full_code：整段代码的一个摘要块（截断），作为最后兜底召回
    full_excerpt = code[:2000]
    if full_excerpt.strip():
        chunks.append(
            ShaderChunk(
                chunk_id=f"{sid}::full_code",
                parent_id=sid,
                kind="full_code",
                title="full_code",
                text=full_excerpt,
                meta={"code_chars": len(code)},
            )
        )

    return chunks
