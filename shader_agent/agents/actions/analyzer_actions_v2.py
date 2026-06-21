"""四段式 Analyzer Actions。

设计原则（来自 GPT-Researcher 的经验）：
- 把"巨型 prompt"拆成 4 段，每段独立可测、可重试；
- 每段都有结构化输出 schema；若 JSON 解析失败，仅回退该段，其他段保留。

四个 Action：
  1. WalkthroughAction      — 代码段逐块讲解
  2. SummaryAction          — 算法摘要
  3. EffectInferAction      — 视觉效果推断
  4. CompareAction          — 与参考样本的异同

每个 Action 都接受相同的输入维度（code + parse_result + similar），
但 prompt 与输出 schema 不同，确保模型在每段都"必须完成"。
"""
from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import BaseModel, Field

from shader_agent.agents.actions.analyzer_actions import ParseShaderOut
from shader_agent.agents.actions.base import Action
from shader_agent.agents.schemas import SimilarShader


# 受控词表（与 corpus.tagger.TOPIC_VOCAB 一致，避免循环依赖直接在此重复）
TECHNIQUE_VOCAB: list[str] = [
    "raymarching", "sdf", "noise", "fractal",
    "post-processing", "2d-pattern", "lighting", "animation",
]


# =====================================================================
# 公共：把 GLSL 代码按"自定义函数 + mainImage"切段
# =====================================================================

def split_code_into_sections(code: str, parse: ParseShaderOut) -> dict[str, str]:
    """把 GLSL 源码切成 { 段标题: 代码 } 字典。

    策略：
    - 每个自定义函数一段；
    - mainImage 单独一段；
    - 函数体之外的全局声明（uniform / const / #define）归到 "globals" 段。

    用法：给 LLM 时按段喂，避免一次性塞 10KB 让模型忽略中段。
    """
    if not code:
        return {}

    # 用 brace-matching 切函数，不能用 split：括号嵌套会乱。
    sections: dict[str, str] = {}
    n = len(code)
    i = 0
    last_global_end = 0

    # 收集 "(type ret) name (args) {" 的位置
    func_re = re.compile(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^;{]*?)\)\s*\{",
    )

    for m in func_re.finditer(code):
        ret_t, fname = m.group(1), m.group(2)
        if ret_t in {"if", "for", "while", "switch", "return", "else"}:
            continue
        start = m.start()
        # 找匹配的 '}'
        depth = 0
        j = m.end() - 1  # 指向开 '{'
        end = -1
        while j < n:
            c = code[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
            j += 1
        if end == -1:
            continue
        # globals 段：上一函数末尾到本函数开头
        if start > last_global_end:
            chunk = code[last_global_end:start].strip()
            if chunk:
                sections.setdefault("globals", "")
                sections["globals"] += ("\n" if sections["globals"] else "") + chunk
        sections[fname] = code[start:end].strip()
        last_global_end = end

    # 尾部 globals
    tail = code[last_global_end:].strip()
    if tail:
        sections.setdefault("globals", "")
        sections["globals"] += ("\n" if sections["globals"] else "") + tail

    return sections


# =====================================================================
# Action 1: WalkthroughAction — 逐段讲解
# =====================================================================

class WalkthroughIn(BaseModel):
    code: str
    parse_result: ParseShaderOut


class WalkthroughOut(BaseModel):
    walkthrough: dict[str, str] = Field(default_factory=dict)
    key_variables: dict[str, str] = Field(default_factory=dict)


_SYSTEM_WALKTHROUGH = (
    "你是资深图形学讲师。你的任务是对一段 Shadertoy GLSL 代码做"
    "**逐段讲解**，并提取关键变量。\n"
    "严格输出以下 JSON 结构，不要任何 markdown 包裹、不要多余文字：\n"
    "{\n"
    '  "walkthrough": { "<段名>": "<2~4 句简体中文解释这段做什么、关键技巧>", ... },\n'
    '  "key_variables": { "<变量名>": "<1 句简体中文说明用途/含义>", ... }\n'
    "}\n"
    "讲解约束：\n"
    "1. 只讲代码里真实出现的内容，不要捏造；\n"
    "2. 每段 60~150 字之间，不要过长；\n"
    "3. key_variables 至少 3 项，最多 10 项；包含 uv / 时间 / 关键参数；\n"
    "4. **所有解释文字必须使用简体中文**，禁止整句英文（GLSL 标识符、函数名、"
    "数学符号可保留原文，但描述性语句一律中文）。\n"
    "5. 即使代码较长，也要覆盖主要函数与 mainImage，保持中文、保持简洁。"
)


class WalkthroughAction(Action[WalkthroughIn, WalkthroughOut]):
    name = "walkthrough"
    input_schema = WalkthroughIn
    output_schema = WalkthroughOut

    def _run(self, inp: WalkthroughIn) -> WalkthroughOut:
        llm_fn: Callable | None = self.dep("llm_fn")
        sections = split_code_into_sections(inp.code, inp.parse_result)
        if not llm_fn:
            return self._fallback(sections)

        # prompt 拼接：把所有 section 一起喂，每段加显式标题。
        #
        # 关于"限流"：绝大多数 Shadertoy shader 都在 5KB 以内，DeepSeek 完全能
        # 一次性吃下，不需要截断。只有极少数超大 shader（>16KB）才有必要限流，
        # 以免 prompt 过大拖慢推理。因此这里把阈值放得很宽：
        #   - 总代码不超过 _SOFT_LIMIT 时：原样全量喂入，不做任何截断；
        #   - 超过时：才按段截断，且单段保留额度也更大，尽量少丢信息。
        # 这样既保证常见 shader 的解释完整、速度不受影响，又能兜住极端超大输入。
        _SOFT_LIMIT = 16000   # 总代码低于此值则完全不截断（约 400~500 行）
        _TOTAL_CAP = 16000    # 触发截断后，喂入的合计上限
        _PER_SECTION = 4000   # 触发截断后，单段最多保留的字符数

        total_len = sum(len(b) for b in sections.values())
        section_text = ""
        if total_len <= _SOFT_LIMIT:
            # 常见情况：全量喂入，不丢任何内容
            for title, body in sections.items():
                section_text += f"\n--- section: {title} ---\n{body}\n"
        else:
            # 超大 shader：按段限流，保留每段开头（含函数签名与主体逻辑）
            used = 0
            for title, body in sections.items():
                if used >= _TOTAL_CAP:
                    section_text += (
                        f"\n--- section: {title} ---\n"
                        f"（代码超长，本段已省略；请根据函数名 `{title}` "
                        f"与整体结构推断其作用）\n"
                    )
                    continue
                snippet = body
                if len(snippet) > _PER_SECTION:
                    snippet = snippet[:_PER_SECTION] + "\n/* …本段过长，仅保留前半部分… */"
                section_text += f"\n--- section: {title} ---\n{snippet}\n"
                used += len(snippet)

        user = (
            f"静态解析：custom_functions={inp.parse_result.custom_functions} "
            f"used_builtins={inp.parse_result.used_builtins} LOC={inp.parse_result.loc}\n"
            f"按以下分段对代码做讲解：\n{section_text}"
        )
        raw = llm_fn([
            {"role": "system", "content": _SYSTEM_WALKTHROUGH},
            {"role": "user", "content": user},
        ])
        try:
            obj = _safe_json(raw)
            return WalkthroughOut(
                walkthrough=dict(obj.get("walkthrough") or {}),
                key_variables=dict(obj.get("key_variables") or {}),
            )
        except Exception:
            return self._fallback(sections, note=f"LLM JSON parse failed: {raw[:80]!r}")

    @staticmethod
    def _fallback(sections: dict[str, str], note: str = "") -> WalkthroughOut:
        wk: dict[str, str] = {}
        for title in sections:
            wk[title] = (
                f"（占位）这段名为 {title}，"
                f"约 {len(sections[title].splitlines())} 行。需 LLM 在线时填充详细讲解。"
            )
        kv: dict[str, str] = {
            "uv": "标准化坐标（[0,1] 或居中后的 [-aspect,aspect]）",
            "iTime": "Shadertoy 内置时间变量（秒）",
            "iResolution": "视口分辨率，xy 为像素宽高",
        }
        if note:
            wk["_note"] = note
        return WalkthroughOut(walkthrough=wk, key_variables=kv)


# =====================================================================
# Action 2: SummaryAction — 算法摘要
# =====================================================================

class SummaryIn(BaseModel):
    code: str
    parse_result: ParseShaderOut
    walkthrough: dict[str, str] = Field(default_factory=dict)


class SummaryOut(BaseModel):
    algorithm_summary: str = ""
    techniques: list[str] = Field(default_factory=list)


_SYSTEM_SUMMARY = (
    "你是资深图形学讲师。你已经看过这段 Shadertoy GLSL 代码的逐段讲解，"
    "现在请产出**算法摘要**与**技术标签**。\n"
    "严格输出以下 JSON，不要任何 markdown 包裹：\n"
    "{\n"
    '  "algorithm_summary": "<150~300 字的简体中文摘要：先说做什么再说怎么做，'
    '强调算法主干，不要复述每行代码>",\n'
    '  "techniques": [<从受控词表里挑 1~4 个>]\n'
    "}\n"
    "**算法摘要必须使用简体中文**，禁止整句英文。\n"
    f"受控词表（只能从这里选）：{TECHNIQUE_VOCAB}"
)


class SummaryAction(Action[SummaryIn, SummaryOut]):
    name = "summary"
    input_schema = SummaryIn
    output_schema = SummaryOut

    def _run(self, inp: SummaryIn) -> SummaryOut:
        llm_fn: Callable | None = self.dep("llm_fn")
        if not llm_fn:
            return self._fallback(inp)

        wk_text = "\n".join(f"[{k}] {v}" for k, v in (inp.walkthrough or {}).items())
        user = (
            f"代码：\n```glsl\n{inp.code[:6000]}\n```\n\n"
            f"已知分段讲解：\n{wk_text or '(无)'}\n\n"
            f"请输出摘要 JSON。"
        )
        raw = llm_fn([
            {"role": "system", "content": _SYSTEM_SUMMARY},
            {"role": "user", "content": user},
        ])
        try:
            obj = _safe_json(raw)
            techs = [t for t in (obj.get("techniques") or []) if t in TECHNIQUE_VOCAB]
            return SummaryOut(
                algorithm_summary=str(obj.get("algorithm_summary") or "").strip(),
                techniques=techs,
            )
        except Exception:
            return self._fallback(inp, note=f"LLM JSON parse failed: {raw[:80]!r}")

    @staticmethod
    def _fallback(inp: SummaryIn, note: str = "") -> SummaryOut:
        funcs = inp.parse_result.custom_functions
        summary = (
            f"该 shader 有 {inp.parse_result.loc} 行，定义了 {len(funcs)} 个自定义函数 "
            f"（{', '.join(funcs[:4])}...）。"
        )
        if note:
            summary += " [note] " + note
        techs: list[str] = []
        joined = " ".join(funcs).lower() + " " + (inp.code or "").lower()
        if "raymarch" in joined or any(f.startswith("sd") for f in funcs):
            if "raymarch" in joined: techs.append("raymarching")
            if any(f.startswith("sd") for f in funcs): techs.append("sdf")
        if "noise" in joined or "hash" in joined:
            techs.append("noise")
        if "mandelbrot" in joined or "fractal" in joined:
            techs.append("fractal")
        if not techs:
            techs.append("2d-pattern")
        return SummaryOut(algorithm_summary=summary, techniques=techs)


# =====================================================================
# Action 3: EffectInferAction — 视觉效果推断
# =====================================================================

class EffectInferIn(BaseModel):
    code: str
    parse_result: ParseShaderOut
    summary: str = ""


class EffectInferOut(BaseModel):
    visual_effect: str = ""


_SYSTEM_EFFECT = (
    "你是图形学专家。看一段 Shadertoy GLSL 代码，从代码推断其**视觉效果**："
    "颜色、形状、运动感、构图特征。\n"
    "**只输出一段 60~150 字的中文描述**，不要 JSON、不要 markdown、不要标题。"
)


class EffectInferAction(Action[EffectInferIn, EffectInferOut]):
    name = "effect_infer"
    input_schema = EffectInferIn
    output_schema = EffectInferOut

    def _run(self, inp: EffectInferIn) -> EffectInferOut:
        llm_fn: Callable | None = self.dep("llm_fn")
        if not llm_fn:
            return EffectInferOut(visual_effect="(占位，需 LLM 在线时填充)")
        user = (
            f"算法摘要（仅供参考，可信度有限）：{inp.summary or '(无)'}\n\n"
            f"代码：\n```glsl\n{inp.code[:5000]}\n```"
        )
        text = llm_fn([
            {"role": "system", "content": _SYSTEM_EFFECT},
            {"role": "user", "content": user},
        ])
        # 容错：去掉可能的 markdown 围栏
        t = (text or "").strip()
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
        return EffectInferOut(visual_effect=t)


# =====================================================================
# Action 4: CompareAction — 与参考样本的异同
# =====================================================================

class CompareIn(BaseModel):
    code: str
    summary: str = ""
    similar: list[SimilarShader] = Field(default_factory=list)


class CompareOut(BaseModel):
    comparison: str = ""              # 给人看的对照分析
    section_walkthrough_extra: dict[str, str] = Field(default_factory=dict)


_SYSTEM_COMPARE = (
    "你是资深图形学讲师。看一段 Shadertoy GLSL 代码与若干个相似样本的元信息，"
    "做**对照分析**：本段代码相对于参考样本，在算法选型、参数取值、性能/视觉权衡上"
    "有什么相同与不同。\n"
    "输出 200~350 字的中文段落，**只输出一段文字**，不要 JSON、不要列表、不要 markdown。"
    "若参考样本为空，则一句话说明并退出。"
)


class CompareAction(Action[CompareIn, CompareOut]):
    name = "compare"
    input_schema = CompareIn
    output_schema = CompareOut

    def _run(self, inp: CompareIn) -> CompareOut:
        if not inp.similar:
            return CompareOut(comparison="（无相似样本可对照——向量库可能为空或检索未命中。）")
        llm_fn: Callable | None = self.dep("llm_fn")
        if not llm_fn:
            names = ", ".join(f"{s.name}({s.shader_id})" for s in inp.similar[:3])
            return CompareOut(comparison=f"（占位）检索到 {len(inp.similar)} 个相似样本：{names}。需 LLM 在线时填充对照分析。")

        ref_text = "\n".join(
            f"[{i+1}] {s.name} (id={s.shader_id}, tags={','.join(s.tags_topic)}, "
            f"distance={s.distance:.3f})\n  excerpt: {s.code_excerpt[:300]}"
            for i, s in enumerate(inp.similar[:3])
        )
        user = (
            f"算法摘要：{inp.summary or '(无)'}\n\n"
            f"待分析代码：\n```glsl\n{inp.code[:4500]}\n```\n\n"
            f"参考样本：\n{ref_text}"
        )
        text = llm_fn([
            {"role": "system", "content": _SYSTEM_COMPARE},
            {"role": "user", "content": user},
        ])
        t = (text or "").strip()
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
        return CompareOut(comparison=t)


# =====================================================================
# 公共工具
# =====================================================================

def _safe_json(text: str) -> dict:
    """容错 JSON 解析：去 markdown 围栏，再 json.loads。"""
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)
