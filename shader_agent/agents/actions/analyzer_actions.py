"""Analyzer 的 Action 集合。

工作流采用"解析 → 检索 → 讲解 → 汇总"四段：

  1. ParseShaderAction       — 静态解析 GLSL（无 LLM）：识别 mainImage、uniforms、关键函数
  2. RetrieveSimilarAction   — 用混合检索器（或向量库）检索相似 shader（无 LLM）
  3. ExplainShaderAction     — 调 LLM：分段讲解 + 算法摘要 + 关键变量
  4. SynthesizeReportAction  — 把前三步合并成 AnalysisReport（无 LLM，纯组装）

ExplainShaderAction 的 llm_fn 可注入，便于单测用 stub 或替换为不同模型。
"""
from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from shader_agent.agents.actions.base import Action
from shader_agent.agents.schemas import AnalysisReport, SimilarShader


# =====================================================================
# 1. ParseShaderAction
# =====================================================================

class ParseShaderIn(BaseModel):
    code: str


class ParseShaderOut(BaseModel):
    """静态解析结果。"""
    has_main_image: bool = False
    uniforms: list[str] = Field(default_factory=list)
    custom_functions: list[str] = Field(default_factory=list)
    used_builtins: list[str] = Field(default_factory=list)  # iTime / iResolution / iMouse ...
    loc: int = 0  # 行数（粗略）


_RE_FUNC = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;{]*\)\s*\{",
)
# 提取 uniform 声明（旧式 + ES 风格）
_RE_UNIFORM = re.compile(
    r"\buniform\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;"
)
_SHADERTOY_BUILTINS = [
    "iResolution", "iTime", "iTimeDelta", "iFrame", "iChannelTime",
    "iMouse", "iDate", "iSampleRate", "iChannelResolution",
    "iChannel0", "iChannel1", "iChannel2", "iChannel3",
]


class ParseShaderAction(Action[ParseShaderIn, ParseShaderOut]):
    name = "parse_shader"
    input_schema = ParseShaderIn
    output_schema = ParseShaderOut

    def _run(self, inp: ParseShaderIn) -> ParseShaderOut:
        code = inp.code or ""
        funcs = []
        for m in _RE_FUNC.finditer(code):
            ret_t, fname = m.group(1), m.group(2)
            # 过滤明显的关键字
            if ret_t in {"if", "for", "while", "switch", "return"}:
                continue
            funcs.append(fname)

        uniforms = [name for _t, name in _RE_UNIFORM.findall(code)]

        builtins_used = [b for b in _SHADERTOY_BUILTINS if b in code]
        return ParseShaderOut(
            has_main_image="mainImage" in code,
            uniforms=uniforms,
            custom_functions=funcs,
            used_builtins=builtins_used,
            loc=len(code.splitlines()),
        )


# =====================================================================
# 2. RetrieveSimilarAction
# =====================================================================

class RetrieveSimilarIn(BaseModel):
    code: str
    top_k: int = 3
    where: dict[str, Any] | None = None
    want_tags: list[str] = Field(default_factory=list)


class RetrieveSimilarOut(BaseModel):
    items: list[SimilarShader] = Field(default_factory=list)


class RetrieveSimilarAction(Action[RetrieveSimilarIn, RetrieveSimilarOut]):
    """检索与输入代码相似的高质量参考样本。

    依赖（按优先级）：
      - retriever: HybridRetriever（向量 + BM25 + 标签/质量过滤 + 重排 + 阈值），
        通过 __init__(retriever=...) 注入，命中父子分块时检索精度最高；
      - vector_store: ShaderVectorStore，作为没有 retriever 时的兼容回退。
    两者都缺失时返回空（不报错，便于单测）。
    """
    name = "retrieve_similar"
    input_schema = RetrieveSimilarIn
    output_schema = RetrieveSimilarOut

    def _run(self, inp: RetrieveSimilarIn) -> RetrieveSimilarOut:
        query = (inp.code or "")[:1500]
        if not query.strip():
            return RetrieveSimilarOut(items=[])

        # 优先走混合检索器
        retriever = self.dep("retriever")
        if retriever is not None:
            hits = retriever.retrieve(query, top_k=inp.top_k, want_tags=inp.want_tags)
            items = [SimilarShader(**h.to_similar_payload()) for h in hits]
            return RetrieveSimilarOut(items=items)

        # 回退：旧的 shader 级向量检索
        vstore = self.dep("vector_store")
        if vstore is None:
            return RetrieveSimilarOut(items=[])
        hits = vstore.query_by_text(query, top_k=inp.top_k, where=inp.where)
        items = []
        for h in hits:
            md = h.get("metadata") or {}
            tags = (md.get("tags_topic") or "")
            items.append(SimilarShader(
                shader_id=h.get("shader_id", ""),
                name=md.get("name", ""),
                distance=float(h.get("distance") or 0.0),
                tags_topic=[t for t in tags.split(",") if t],
                code_excerpt=(h.get("document") or "")[:600],
            ))
        return RetrieveSimilarOut(items=items)


# =====================================================================
# 3. ExplainShaderAction
# =====================================================================

class ExplainShaderIn(BaseModel):
    code: str
    parse_result: ParseShaderOut
    similar: list[SimilarShader] = Field(default_factory=list)


class ExplainShaderOut(BaseModel):
    algorithm_summary: str
    key_variables: dict[str, str] = Field(default_factory=dict)
    techniques: list[str] = Field(default_factory=list)
    visual_effect: str = ""
    section_walkthrough: dict[str, str] = Field(default_factory=dict)


# LLM 调用签名：messages -> 文本回复
LLMFn = Callable[[list[dict[str, str]]], str]


_SYSTEM_PROMPT_EXPLAIN = (
    "你是一名资深图形学讲师，擅长用准确的术语讲解 GLSL / Shadertoy 代码。\n"
    "你的任务：阅读一段 Shadertoy fragment shader，并产出一份结构化讲解。\n"
    "严格按以下 JSON Schema 输出，不要任何 markdown 标记、不要多余文字：\n"
    "{\n"
    '  "algorithm_summary": "<200~400 字的中文摘要：先说做什么再说怎么做>",\n'
    '  "key_variables": { "<var>": "<用途>" , ... },\n'
    '  "techniques": ["raymarching"|"sdf"|"noise"|"fractal"|'
    '"post-processing"|"2d-pattern"|"lighting"|"animation"],\n'
    '  "visual_effect": "<一句中文描述视觉效果>",\n'
    '  "section_walkthrough": { "<段标题>": "<中文解释>", ... }\n'
    "}\n"
    "techniques 只能从给定词表里选。"
)


class ExplainShaderAction(Action[ExplainShaderIn, ExplainShaderOut]):
    """调用 LLM 生成讲解。

    依赖：
      - llm_fn: 由 Role 在构造时通过 __init__ 注入；签名见 LLMFn。
        单测时传 stub；接到 DeepSeek。
    """
    name = "explain_shader"
    input_schema = ExplainShaderIn
    output_schema = ExplainShaderOut

    def _build_messages(self, inp: ExplainShaderIn) -> list[dict[str, str]]:
        ref_text = ""
        if inp.similar:
            ref_lines = [
                f"- {s.name} (id={s.shader_id}, tags={','.join(s.tags_topic)})"
                for s in inp.similar[:3]
            ]
            ref_text = "参考的相似样本（可对照）：\n" + "\n".join(ref_lines)

        parse_text = (
            f"静态解析：\n"
            f"- has_main_image = {inp.parse_result.has_main_image}\n"
            f"- custom_functions = {inp.parse_result.custom_functions}\n"
            f"- used_builtins = {inp.parse_result.used_builtins}\n"
            f"- LOC = {inp.parse_result.loc}\n"
        )

        user = (
            parse_text + "\n" +
            (ref_text + "\n\n" if ref_text else "") +
            "需要讲解的 shader：\n```glsl\n" + (inp.code or "") + "\n```"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT_EXPLAIN},
            {"role": "user", "content": user},
        ]

    def _run(self, inp: ExplainShaderIn) -> ExplainShaderOut:
        llm_fn: LLMFn | None = self.dep("llm_fn")
        if llm_fn is None:
            # 占位：无 LLM 时返回基于解析结果的最小可用 explain
            return self._fallback(inp)
        messages = self._build_messages(inp)
        text = llm_fn(messages)
        return self._parse_llm_json(text, fallback_input=inp)

    @staticmethod
    def _parse_llm_json(text: str, fallback_input: ExplainShaderIn) -> ExplainShaderOut:
        import json
        s = (text or "").strip()
        # 容错：去 markdown 包裹
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        try:
            obj = json.loads(s)
            return ExplainShaderOut(
                algorithm_summary=str(obj.get("algorithm_summary", "")),
                key_variables=dict(obj.get("key_variables", {}) or {}),
                techniques=list(obj.get("techniques", []) or []),
                visual_effect=str(obj.get("visual_effect", "")),
                section_walkthrough=dict(obj.get("section_walkthrough", {}) or {}),
            )
        except Exception:
            # 解析失败 → 走 fallback
            return ExplainShaderAction._build_fallback(fallback_input,
                note=f"LLM 输出非合法 JSON，已回退到静态解析。原文片段: {s[:120]}")

    @staticmethod
    def _build_fallback(inp: ExplainShaderIn, note: str = "") -> ExplainShaderOut:
        techniques: list[str] = []
        funcs = " ".join(inp.parse_result.custom_functions).lower()
        code_l = (inp.code or "").lower()
        if "raymarch" in code_l or any("ro" in f or "rd" in f for f in [funcs]):
            techniques.append("raymarching")
        if any(f.startswith("sd") for f in inp.parse_result.custom_functions):
            techniques.append("sdf")
        if "noise" in funcs or "hash" in funcs:
            techniques.append("noise")
        if "mandelbrot" in code_l or "fractal" in code_l:
            techniques.append("fractal")
        if not techniques:
            techniques.append("2d-pattern")
        summary = (
            f"该 shader 有 {inp.parse_result.loc} 行，定义了 "
            f"{len(inp.parse_result.custom_functions)} 个自定义函数 "
            f"({', '.join(inp.parse_result.custom_functions[:5])} ...)，"
            f"使用了内置变量 {inp.parse_result.used_builtins}。"
        )
        if note:
            summary += "\n[note] " + note
        return ExplainShaderOut(
            algorithm_summary=summary,
            techniques=techniques,
            visual_effect="（占位，需 LLM 在线时填充）",
        )

    def _fallback(self, inp: ExplainShaderIn) -> ExplainShaderOut:
        return self._build_fallback(inp, note="未注入 llm_fn，使用静态回退。")


# =====================================================================
# 4. SynthesizeReportAction
# =====================================================================

class SynthesizeReportIn(BaseModel):
    code: str
    explain: ExplainShaderOut
    similar: list[SimilarShader] = Field(default_factory=list)
    model_used: str = ""


class SynthesizeReportAction(Action[SynthesizeReportIn, AnalysisReport]):
    name = "synthesize_report"
    input_schema = SynthesizeReportIn
    output_schema = AnalysisReport

    def _run(self, inp: SynthesizeReportIn) -> AnalysisReport:
        return AnalysisReport(
            source_code=inp.code,
            algorithm_summary=inp.explain.algorithm_summary,
            key_variables=inp.explain.key_variables,
            techniques=inp.explain.techniques,
            visual_effect=inp.explain.visual_effect,
            section_walkthrough=inp.explain.section_walkthrough,
            similar_shaders=list(inp.similar),
            model_used=inp.model_used,
        )
