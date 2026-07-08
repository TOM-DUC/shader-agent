"""Generator 的 Action 集合（4 个）。

工作流仿 MetaGPT 的 PRD → Design → Code：

  1. ParseSpecAction      — 解析用户自由文本 → GenerationSpec（无 LLM，规则关键词）
  2. RetrieveExamplesAction — 检索风格相似的 in-context 例子（无 LLM）
  3. DraftCodeAction      — 调 LLM 生成 GLSL 草稿
  4. ValidateCodeAction   — 编译验证（静态规则校验 + 可选真实编译）
  + 修正循环由 Role.handle() 编排，不单独做成一个 Action（流程性强、状态多）

DraftCodeAction 留可注入 llm_fn；ValidateCodeAction 给桩。
"""
from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from shader_agent.agents.actions.base import Action
from shader_agent.agents.schemas import (
    CompileResult,
    GenerationSpec,
    SimilarShader,
)


# =====================================================================
# 1. ParseSpecAction
# =====================================================================

class ParseSpecIn(BaseModel):
    user_text: str
    inherit_from: GenerationSpec | None = None


class ParseSpecOut(BaseModel):
    spec: GenerationSpec


_EFFECT_KW: dict[str, list[str]] = {
    "raymarching": ["raymarch", "ray march", "光线步进", "ray-march"],
    "sdf": ["sdf", "signed distance", "符号距离"],
    "noise": ["noise", "噪声", "fbm", "perlin", "simplex"],
    "fractal": ["fractal", "分形", "mandelbrot", "julia"],
    "post-processing": ["post", "后处理", "bloom", "vignette", "blur"],
    "2d-pattern": ["pattern", "图案", "kaleidoscope", "万花筒", "checker"],
    "lighting": ["light", "光照", "phong", "shading"],
    "animation": ["anim", "动画", "时间", "动态"],
}
_PALETTE_KW: dict[str, list[str]] = {
    "neon": ["neon", "霓虹"],
    "warm sunset": ["sunset", "warm", "暖色", "落日"],
    "monochrome": ["mono", "黑白", "单色"],
    "cool blue": ["cool", "blue", "蓝", "冷色"],
    "pastel": ["pastel", "粉彩"],
    "vibrant": ["vibrant", "鲜艳", "饱和"],
}
_COMPLEXITY_KW: dict[str, list[str]] = {
    "minimal": ["最简", "minimal", "最小"],
    "simple": ["简单", "simple", "basic"],
    "moderate": ["中等", "moderate", "适中"],
    "complex": ["复杂", "complex", "advanced"],
}


class ParseSpecAction(Action[ParseSpecIn, ParseSpecOut]):
    name = "parse_spec"
    input_schema = ParseSpecIn
    output_schema = ParseSpecOut

    def _run(self, inp: ParseSpecIn) -> ParseSpecOut:
        text = (inp.user_text or "").strip()
        lower = text.lower()

        effect_type = ""
        for k, kws in _EFFECT_KW.items():
            if any(w in lower for w in kws):
                effect_type = k
                break

        palette = ""
        for k, kws in _PALETTE_KW.items():
            if any(w in lower for w in kws):
                palette = k
                break

        complexity = "simple"
        for k, kws in _COMPLEXITY_KW.items():
            if any(w in lower for w in kws):
                complexity = k
                break

        dynamic = not any(w in lower for w in ["static", "静态", "不动"])

        # 抽取硬约束（"不要用 xxx" / "<= n 行"）
        constraints: list[str] = []
        for m in re.finditer(r"不要用([\u4e00-\u9fa5A-Za-z0-9 ]{2,30})", text):
            constraints.append(f"avoid {m.group(1).strip()}")
        for m in re.finditer(r"<=?\s*(\d+)\s*行", text):
            constraints.append(f"<= {m.group(1)} lines")
        if "no texture" in lower or "无纹理" in text or "不要用纹理" in text:
            constraints.append("no external textures")

        base = inp.inherit_from
        spec = GenerationSpec(
            description=text,
            effect_type=effect_type or (base.effect_type if base else ""),
            palette=palette or (base.palette if base else ""),
            dynamic=dynamic,
            complexity=complexity,  # type: ignore[arg-type]
            constraints=constraints or (list(base.constraints) if base else []),
            reference_report=base.reference_report if base else None,
        )
        return ParseSpecOut(spec=spec)


# =====================================================================
# 2. RetrieveExamplesAction
# =====================================================================

class RetrieveExamplesIn(BaseModel):
    spec: GenerationSpec
    top_k: int = 3


class RetrieveExamplesOut(BaseModel):
    items: list[SimilarShader] = Field(default_factory=list)


class RetrieveExamplesAction(Action[RetrieveExamplesIn, RetrieveExamplesOut]):
    """检索与 spec 相符的 in-context 例子。

    依赖（按优先级）：
      - retriever: HybridRetriever，命中父子分块时按效果类型/配色/复杂度检索最准；
      - vector_store: ShaderVectorStore，作为兼容回退。
    """
    name = "retrieve_examples"
    input_schema = RetrieveExamplesIn
    output_schema = RetrieveExamplesOut

    def _run(self, inp: RetrieveExamplesIn) -> RetrieveExamplesOut:
        spec = inp.spec
        # query 拼接：description + effect_type + palette
        query_parts = [spec.description]
        if spec.effect_type:
            query_parts.append(spec.effect_type)
        if spec.palette:
            query_parts.append(spec.palette)
        q = " ".join(p for p in query_parts if p).strip()
        if not q:
            return RetrieveExamplesOut(items=[])

        want_tags = [spec.effect_type] if spec.effect_type else []

        retriever = self.dep("retriever")
        if retriever is not None:
            hits = retriever.retrieve(q, top_k=inp.top_k, want_tags=want_tags)
            items = [SimilarShader(**h.to_similar_payload()) for h in hits]
            return RetrieveExamplesOut(items=items)

        vstore = self.dep("vector_store")
        if vstore is None:
            return RetrieveExamplesOut(items=[])
        hits = vstore.query_by_text(q, top_k=inp.top_k)
        items = []
        for h in hits:
            md = h.get("metadata") or {}
            tags = (md.get("tags_topic") or "")
            items.append(SimilarShader(
                shader_id=h.get("shader_id", ""),
                name=md.get("name", ""),
                distance=float(h.get("distance") or 0.0),
                tags_topic=[t for t in tags.split(",") if t],
                code_excerpt=(h.get("document") or "")[:800],
            ))
        return RetrieveExamplesOut(items=items)


# =====================================================================
# 3. DraftCodeAction
# =====================================================================

class DraftCodeIn(BaseModel):
    spec: GenerationSpec
    examples: list[SimilarShader] = Field(default_factory=list)
    prev_code: str = ""
    prev_errors: str = ""


class DraftCodeOut(BaseModel):
    code: str
    explanation: str = ""


_SYSTEM_PROMPT_GENERATE = (
    "你是 ShaderGenerator。你的任务是按照用户需求生成一段 Shadertoy 风格的 "
    "GLSL fragment shader。\n"
     "硬性约束：\n"
    "1. 必须实现 `void mainImage(out vec4 fragColor, in vec2 fragCoord)` 入口；\n"
    "2. 仅使用 Shadertoy 默认的内置 uniform（iResolution / iTime / iMouse），"
    "不引用外部纹理或额外 pass（不要 `iChannel0`、`sampler2D`、自定义 uniform）；\n"
    "3. 代码自包含、能直接在 Shadertoy 编辑器中编译运行；\n"
    "4. WebGL 1.0 兼容：避免使用 GLSL ES 3.0 特有函数（如 `round()`、`roundEven()`、"
    "`trunc()` 等），改用 `floor(x + 0.5)` 等兼容写法；\n"
    "5. 必须严格遵循需求中的 palette（调色板）：让主色调明显体现该倾向；"
    "必须遵循 complexity（复杂度）：minimal=几行极简，simple=单一效果，"
    "moderate=多技巧组合，complex=可炫技；dynamic=true 时必须用 iTime 做动画。\n"
    "输出格式（务必遵守）：\n"
    "先输出一段以 `// EXPLAIN:` 开头的**中文**说明块，可跨多行（每行都以 `//` 开头），"
    "需说明清楚：① 使用的主要算法/技术（如 raymarching、SDF、值噪声、域重复等）；"
     "② 如何体现指定的调色板与动态；③ 大致的实现思路。说明控制在 **3~6 句**。\n"
    "说明块之后紧跟纯 GLSL 代码，不要 markdown 代码围栏。\n"
    "全程用简体中文书写说明，不要使用英文整句。"
)

# 修正轮：只修编译错误，但解释仍面向"最终成品的设计思路"（与一次成功时一致）。
_SYSTEM_PROMPT_FIX = (
    "你是 ShaderGenerator 的修正模式。上一轮生成的 GLSL 代码未通过编译/校验。\n"
    "本轮目标：**最小改动** 修复指出的问题，**保留原有算法与结构**，不要重写。\n"
    "硬性约束（与首轮一致）：\n"
    "1. 入口 `void mainImage(out vec4 fragColor, in vec2 fragCoord)`；\n"
    "2. 仅使用 iResolution/iTime/iMouse；不引外部纹理；\n"
    "3. WebGL 1.0 兼容：避免 `round()`、`roundEven()`、`trunc()` 等 ES 3.0 函数，改用 `floor(x + 0.5)`。\n"
    "修复策略：仔细读编译错误定位到具体行/函数；只改报错相关部分；"
    "不要因修复而引入新特性或改变视觉效果。\n"
    "输出格式：先输出以 `// EXPLAIN:` 开头的**中文**多行说明块（每行以 `//` 开头），"
    "**说明这段 shader 最终成品的设计思路与主要算法、如何体现配色与动态**"
    "（3~5 句，面向成品，不要写成『本次修复改了什么』）；说明之后紧跟完整的"
    "修复后代码，不要 markdown 围栏。全程用简体中文。"
)

# 改写模式：基于原始代码做**最小化**修改，而非从零重写。
# 注意：改写模式**不注入分析报告**，只给原始代码——避免提示词过长拖慢推理。
_SYSTEM_PROMPT_REWRITE = (
    "你是 ShaderRemixer。给你一段**已有的** Shadertoy GLSL 代码和一条改写指令，"
    "请在**保留原代码整体结构与算法主干**的前提下，"
    "**只针对改写指令所涉及的部分做最小化修改**（增删改尽量局部）。\n"
    "严格要求：\n"
    "1. 这是【改写】不是【重写】：除指令明确要求的改动外，其余代码（函数划分、"
    "变量命名、算法流程）尽量原样保留，便于用户对比 diff；\n"
    "2. 不要为了『更好』而擅自重构无关代码、替换算法或改变未被要求的视觉细节；\n"
    "3. 仍须能在 Shadertoy 编译：入口 `void mainImage(out vec4 fragColor, "
    "in vec2 fragCoord)`，仅用 iResolution/iTime/iMouse，不引外部纹理；\n"
    "4. WebGL 1.0 兼容：避免 `round()`、`roundEven()`、`trunc()` 等 ES 3.0 函数，改用 `floor(x + 0.5)`。\n"
    "5. 直接基于给定代码改，不要参考或套用其他范例。\n"
    "输出格式：先输出以 `// EXPLAIN:` 开头的**中文**多行说明块（每行以 `//` 开头），"
     "**简明说明你改了哪些部分、保留了哪些原有算法**（3~6 句，"
    "不要长篇分析原代码）；说明之后紧跟完整的改写后代码，不要 markdown 围栏。"
    "全程用简体中文。"
)


def _clip_prompt_text(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit: return text
    return text[:limit].rstrip() + "\n// ..."

def _format_reference_for_prompt(ex, idx, limit=3200):
    tags = ",".join(ex.tags_topic) if ex.tags_topic else "-"
    chunks = ", ".join(ex.matched_chunks[:5]) if ex.matched_chunks else "-"
    context = (ex.reference_context or ex.code_excerpt or "").strip()
    return (f"[reference {idx}] {ex.name or ex.shader_id} (id={ex.shader_id}, distance={ex.distance:.3f}, tags={tags})\nmatched_chunks: {chunks}\n下面是从该参考中抽取的算法摘要、命中函数块和必要上下文。只学习实现思路与局部技巧，不要整段照抄。\n{_clip_prompt_text(context, limit)}")


class DraftCodeAction(Action[DraftCodeIn, DraftCodeOut]):
    """调 LLM 生成 GLSL 草稿。

    依赖：llm_fn (Callable)。无则返回 stub 代码（用于 dry-run 测试）。
    """
    name = "draft_code"
    input_schema = DraftCodeIn
    output_schema = DraftCodeOut

    def _build_messages(self, inp: DraftCodeIn) -> list[dict[str, str]]:
        is_fix = bool(inp.prev_code and inp.prev_errors)
        spec = inp.spec
        spec_lines = [
            f"- description: {spec.description}",
            f"- effect_type: {spec.effect_type or '(unset)'}",
            f"- palette: {spec.palette or '(unset)'}",
            f"- dynamic: {spec.dynamic}",
            f"- complexity: {spec.complexity}",
        ]
        if spec.constraints:
            spec_lines.append(f"- constraints: {spec.constraints}")
        if spec.reference_report is not None and not spec.rewrite_mode:
            # 仅非改写任务才注入分析报告；改写模式靠原始代码本身，避免提示词膨胀
            ref = spec.reference_report
            spec_lines.append(
                "- 参考已分析的 shader：\n"
                f"  · 算法摘要：{ref.algorithm_summary[:200]}\n"
                f"  · 技术标签：{ref.techniques}\n"
                f"  · 关键变量：{list(ref.key_variables.keys())[:6]}"
            )

        if is_fix:
            # 修正轮：把上一轮代码与错误高亮放在最前，spec 退居参考位
            user = (
                "需要修复的上一轮代码：\n"
                f"```glsl\n{inp.prev_code[:2500]}\n```\n\n"
                f"编译/校验错误：\n{inp.prev_errors[:800]}\n\n"
                "原始需求 spec（供参考，不要改变算法主干）：\n"
                + "\n".join(spec_lines)
                + "\n\n请输出修复后的完整代码。"
            )
            sys_prompt = _SYSTEM_PROMPT_FIX
        elif spec.rewrite_mode and spec.base_code.strip():
            # 改写模式：只给原始代码 + 改写指令，不附带分析报告（提示词更短更快）
            user = (
                "原始代码（在此基础上做**最小化**修改，保留无关部分）：\n"
                f"```glsl\n{spec.base_code[:6000]}\n```\n\n"
                f"改写指令：{spec.description}\n\n"
                + (f"调色板倾向：{spec.palette}\n" if spec.palette else "")
                + (f"额外约束：{spec.constraints}\n" if spec.constraints else "")
                + "\n请输出改写后的完整代码（仅在必要处改动）。"
            )
            sys_prompt = _SYSTEM_PROMPT_REWRITE
        else:
            # 首轮：spec + examples 主导
            ex_text = ""
            if inp.examples:
                ex_lines = []
                budget = 3600; used = 0
                for i, ex in enumerate(inp.examples):
                    per_ref = 1600 if i == 1 else 1000
                    block = _format_reference_for_prompt(ex, i, limit=per_ref)
                    if used + len(block) > budget:
                        block = _clip_prompt_text(block, max(900, budget - used))
                    if block.strip():
                        ex_lines.append(block); used += len(block)
                    if used >= budget: break
                ex_text = "\n\n参考样本（只参考算法结构、函数写法和视觉风格，不要照抄）：\n" + "\n\n".join(ex_lines)
            user = (
                "需求 spec：\n" + "\n".join(spec_lines) + ex_text +
                "\n\n请输出 shader 代码（务必满足硬性约束）。"
            )
            sys_prompt = _SYSTEM_PROMPT_GENERATE

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]

    def _run(self, inp: DraftCodeIn) -> DraftCodeOut:
        llm_fn: Callable | None = self.dep("llm_fn")
        if llm_fn is None:
            import sys as _sys
            if "pytest" in _sys.modules:
                return self._stub_draft(inp)
            raise RuntimeError(
                "Generator 的 LLM 函数（code_fn）未就绪。"
                "请检查 DEEPSEEK_API_KEY 是否配置，或等待 LLM 初始化完成后重试。"
            )
        messages = self._build_messages(inp)
        text = llm_fn(messages)
        return self._parse_llm_output(text)

    @staticmethod
    def _parse_llm_output(text: str) -> DraftCodeOut:
        s = (text or "").strip()
        # 去掉可能出现的 markdown 围栏
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        explanation = ""
        # 解析以 `// EXPLAIN:` 开头的多行说明块：从 EXPLAIN 行起，
        # 连续的以 `//` 开头的行都算说明，遇到第一行非注释（代码）即停止。
        lines = s.splitlines()
        m0 = re.match(r"\s*//\s*EXPLAIN:\s*(.*)", lines[0]) if lines else None
        if m0:
            expl_lines = [m0.group(1).strip()]
            consumed = 1
            for ln in lines[1:]:
                mm = re.match(r"\s*//\s?(.*)", ln)
                if mm is not None:
                    expl_lines.append(mm.group(1).rstrip())
                    consumed += 1
                else:
                    break
            explanation = "\n".join(x for x in expl_lines if x).strip()
            s = "\n".join(lines[consumed:])
        return DraftCodeOut(code=s.strip(), explanation=explanation)

    @staticmethod
    def _stub_draft(inp: DraftCodeIn) -> DraftCodeOut:
        """无 LLM 时的占位输出：根据 spec.effect_type 给出一个最小合规 shader。

        仅用于无 LLM 时的 dry-run 回退路径。
        """
        et = inp.spec.effect_type or "2d-pattern"
        templates = {
            "raymarching": (
                "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
                "  vec2 uv=(fragCoord-0.5*iResolution.xy)/iResolution.y;\n"
                "  vec3 ro=vec3(0,0,3),rd=normalize(vec3(uv,-1.5));\n"
                "  float t=0.0; for(int i=0;i<48;i++){ vec3 p=ro+rd*t;\n"
                "    float d=length(p)-1.0; if(d<.001)break; t+=d;\n"
                "    if(t>20.) {t=-1.; break;} }\n"
                "  vec3 col= t>0.? vec3(.4,.7,.9)*0.8 : vec3(0);\n"
                "  fragColor=vec4(col,1);\n}\n"
            ),
            "noise": (
                "float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}\n"
                "void mainImage(out vec4 fragColor,in vec2 fragCoord){\n"
                "  vec2 uv=fragCoord/iResolution.xy*8.0+iTime;\n"
                "  vec2 i=floor(uv),f=fract(uv); f=f*f*(3.-2.*f);\n"
                "  float n=mix(mix(h(i),h(i+vec2(1,0)),f.x),"
                "mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);\n"
                "  fragColor=vec4(vec3(n),1);\n}\n"
            ),
        }
        code = templates.get(et, (
            "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
            "  vec2 uv=fragCoord/iResolution.xy;\n"
            "  fragColor=vec4(uv,0.5+0.5*sin(iTime),1.0);\n}\n"
        ))
        return DraftCodeOut(
            code=code,
            explanation=f"(stub) {et} 占位实现，未调用 LLM。",
        )


# =====================================================================
# 4. ValidateCodeAction
# =====================================================================

class ValidateCodeIn(BaseModel):
    code: str


class ValidateCodeOut(BaseModel):
    result: CompileResult


# --- GLSL 内置函数白名单（用于检测拼写错误）---
# 不求完整，只列高频；未在表中不报错（避免误杀自定义函数）
_GLSL_BUILTIN_FUNCS: set[str] = {
    # 数学
    "abs", "sign", "floor", "ceil", "round", "trunc", "fract", "mod",
    "min", "max", "clamp", "mix", "step", "smoothstep",
    "sqrt", "inversesqrt", "pow", "exp", "log", "exp2", "log2",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    # 向量
    "length", "distance", "dot", "cross", "normalize", "reflect", "refract",
    "faceforward",
    # 矩阵 / 类型
    "mat2", "mat3", "mat4", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    # 纹理（虽然禁用，但解析时要识别）
    "texture", "textureLod", "texture2D",
    # 杂项
    "fwidth", "dFdx", "dFdy",
    # 几何 / 投影
    "transpose", "determinant", "inverse",
    "any", "all", "not", "lessThan", "greaterThan", "equal",
    # 复合
    "outerProduct",
}

# 容易拼错的常见词
_GLSL_TYPO_HINTS: dict[str, str] = {
    "lenght": "length",
    "lentgh": "length",
    "lerp": "mix",                # GLSL 用 mix，不是 lerp
    "saturate": "clamp(x, 0.0, 1.0)",  # HLSL 函数
    "Normalize": "normalize",
    "Length": "length",
    "frac": "fract",              # HLSL 用 frac，GLSL 用 fract
}

# 必须以浮点写的常量上下文：vec*(int, ...) / float = int / mix(a, b, 0)
# 完整精确检测代价太高；这里只识别明显的 "float x = 1;" 模式


class ValidateCodeAction(Action[ValidateCodeIn, ValidateCodeOut]):
    """静态规则校验 + 可选的真实 GLSL 编译。

    规则校验（mainImage 存在 / 大括号配平 / 禁用外部 sampler）
    增加 GLSL 静态可疑模式（拼写、HLSL 误用、注释包裹的关键字）
    通过 __init__(compiler=...) 注入 moderngl 编译器，做真实 compile。

    错误格式：每行一条，便于 LLM 修正轮按行读取。
    """
    name = "validate_code"
    input_schema = ValidateCodeIn
    output_schema = ValidateCodeOut

    def _run(self, inp: ValidateCodeIn) -> ValidateCodeOut:
        code = inp.code or ""

        errors: list[str] = []
        warnings: list[str] = []

        # === 规则：硬性入口 / 配对 / 外部 sampler ===
        if "mainImage" not in code:
            errors.append("missing mainImage entry function")
        # 括号配平（先剥注释，避免注释里的括号被算入）
        stripped = _strip_comments(code)
        if stripped.count("{") != stripped.count("}"):
            errors.append(
                f"unbalanced braces: {{ count={stripped.count('{')} "
                f"vs }} count={stripped.count('}')}"
            )
        if stripped.count("(") != stripped.count(")"):
            errors.append(
                f"unbalanced parens: ( count={stripped.count('(')} "
                f"vs ) count={stripped.count(')')}"
            )
        if re.search(r"\bsampler2D\b|\bsamplerCube\b|\biChannel[0-9]\b", stripped):
            errors.append("references external sampler/iChannelN (not allowed)")

        # === HLSL 误用 ===
        for bad, fix in _GLSL_TYPO_HINTS.items():
            # 以词边界匹配避免误杀（"frac" 不能匹配 "fract"）
            if re.search(rf"\b{re.escape(bad)}\b", stripped):
                errors.append(f"unknown identifier `{bad}` — in GLSL use `{fix}`")

        # === mainImage 签名校验 ===
        sig = re.search(
            r"\bvoid\s+mainImage\s*\(\s*out\s+vec4\s+\w+\s*,\s*in\s+vec2\s+\w+\s*\)",
            stripped,
        )
        if "mainImage" in stripped and not sig:
            errors.append(
                "mainImage signature mismatch — must be "
                "`void mainImage(out vec4 fragColor, in vec2 fragCoord)`"
            )

        # === fragColor 必须被赋值 ===
        # 简单启发：找 mainImage body 内是否有 fragColor= 之类
        if sig:
            body_start = sig.end()
            body = stripped[body_start:body_start + 4000]
            if not re.search(r"\b\w+\s*=", body):
                warnings.append("mainImage body has no assignment statement")
            elif not re.search(r"\bfrag\w*\s*=", body) and not re.search(r"out\s+vec4\s+(\w+)", sig.group()):
                warnings.append("fragColor seems not assigned")

        # === 可选：注入真实编译器 ===
        compiler = self.dep("compiler")
        if compiler is not None and not errors:
            try:
                cr: CompileResult = compiler.compile(code)
                # 把静态 warnings 合并进去
                if warnings:
                    cr = CompileResult(
                        ok=cr.ok,
                        errors=cr.errors,
                        warnings=(cr.warnings + "\n" if cr.warnings else "") + "\n".join(warnings),
                    )
                return ValidateCodeOut(result=cr)
            except Exception as e:
                errors.append(f"compiler exception: {e}")

        cr = CompileResult(
            ok=(len(errors) == 0),
            errors="\n".join(errors),
            warnings="\n".join(warnings),
        )
        return ValidateCodeOut(result=cr)


def _strip_comments(code: str) -> str:
    """去掉 // 行注释与 /* */ 块注释，方便结构性检查。"""
    # 块注释
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # 行注释
    code = re.sub(r"//[^\n]*", "", code)
    return code


# =====================================================================
# 5. SelfCritiqueAction (新增，多模态自评占位)
# =====================================================================

class SelfCritiqueIn(BaseModel):
    code: str
    spec: GenerationSpec
    # 接渲染器后，会把截图（base64 PNG 字符串）传入
    rendered_image_b64: str = ""
    # 即使没有多模态/截图，也可以让 LLM 基于编译结果做文本自评
    compile_ok: bool = True
    compile_errors: str = ""


class SelfCritiqueOut(BaseModel):
    score: float = 0.0          # 0~1，越高越符合
    rationale: str = ""         # 评语
    suggested_diff: str = ""    # 可选：建议修改


class SelfCritiqueAction(Action[SelfCritiqueIn, SelfCritiqueOut]):
    """自评。支持三档，自动按可用资源降级：

    依赖：
      - critique_fn (Callable): (code, spec_text, image_b64) -> str(JSON)
        多模态/文本兼用；其文本回退版即便 image_b64 为空也能基于代码+spec 评分。
      - text_critique_fn (Callable): (code, spec_text, compile_info) -> str(JSON)
        纯文本自评：分析代码与 spec 的吻合度，并在编译失败时分析编译错误。

    档位：
      1. 有截图 + critique_fn      → 多模态自评（最强）
      2. 无截图但有 text_critique_fn → 文本 LLM 自评（可分析编译错误）★本次新增
      3. 都没有                     → 关键词启发式弱自评（最弱，离线兜底）
    """
    name = "self_critique"
    input_schema = SelfCritiqueIn
    output_schema = SelfCritiqueOut
    critical = False  # 自评失败不阻断主流程

    def _run(self, inp: SelfCritiqueIn) -> SelfCritiqueOut:
        critique_fn: Callable | None = self.dep("critique_fn")
        text_fn: Callable | None = self.dep("text_critique_fn")
        spec_text = (
            f"description={inp.spec.description}; "
            f"effect_type={inp.spec.effect_type}; "
            f"palette={inp.spec.palette}; "
            f"dynamic={inp.spec.dynamic}"
        )

        # 档位 1：多模态（有截图）
        if critique_fn is not None and inp.rendered_image_b64:
            raw = critique_fn(inp.code, spec_text, inp.rendered_image_b64)
            return self._parse_json_critique(raw)

        # 档位 2：纯文本 LLM 自评（无截图也能分析编译错误）
        if text_fn is not None:
            compile_info = (
                "编译通过" if inp.compile_ok
                else f"编译失败，错误如下：\n{(inp.compile_errors or '')[:1500]}"
            )
            try:
                raw = text_fn(inp.code, spec_text, compile_info)
                out = self._parse_json_critique(raw)
                # 编译失败时给 score 设上限，避免"代码很贴合 spec 但根本编不过"
                if not inp.compile_ok and out.score > 0.4:
                    out.score = 0.4
                return out
            except Exception as e:
                # LLM 文本自评失败，退到启发式
                return self._text_only_critique(inp, note=f"text critique failed: {e}")

        # 档位 3：关键词启发式兜底
        return self._text_only_critique(inp)

    @staticmethod
    def _parse_json_critique(raw: str) -> SelfCritiqueOut:
        try:
            import json
            obj = json.loads((raw or "").strip())
            return SelfCritiqueOut(
                score=float(obj.get("score", 0.0)),
                rationale=str(obj.get("rationale", "")),
                suggested_diff=str(obj.get("suggested_diff", "")),
            )
        except Exception:
            return SelfCritiqueOut(
                score=0.0,
                rationale=f"(critique parse failed) raw={(raw or '')[:120]!r}",
            )

    @staticmethod
    def _text_only_critique(inp: SelfCritiqueIn, note: str = "") -> SelfCritiqueOut:
        """无渲染器、无 LLM 时的弱自评：检查代码是否提及 spec 中要点 + 编译状态。"""
        code_l = (inp.code or "").lower()
        hits = 0; total = 0
        bits: list[str] = []
        # 编译状态先纳入评分
        total += 1
        if inp.compile_ok:
            hits += 1; bits.append("✓ 编译通过")
        else:
            head = (inp.compile_errors or "").strip().splitlines()
            head_txt = head[-1][:80] if head else "未知错误"
            bits.append(f"✗ 编译失败：{head_txt}")
        if inp.spec.dynamic:
            total += 1
            if "itime" in code_l:
                hits += 1; bits.append("✓ 包含时间动画")
            else:
                bits.append("✗ spec 要求 dynamic 但代码未引用 iTime")
        if inp.spec.effect_type:
            total += 1
            keywords = {
                "raymarching": ["raymarch", "ro", "rd", "march"],
                "sdf": ["sd", "distance"],
                "noise": ["noise", "hash", "fract(sin"],
                "fractal": ["mandelbrot", "julia", "iter"],
            }.get(inp.spec.effect_type, [])
            if any(k in code_l for k in keywords):
                hits += 1; bits.append(f"✓ 包含 {inp.spec.effect_type} 风格特征")
            else:
                bits.append(f"~ 未明确看出 {inp.spec.effect_type} 特征")
        score = (hits / total) if total > 0 else 0.5
        rationale = "（文本层自评，未启用渲染）" + "; ".join(bits)
        if note:
            rationale += f" [{note}]"
        return SelfCritiqueOut(score=score, rationale=rationale)
