"""共享数据契约。

设计原则：
- 任何跨 Role 的产物都用这里的 pydantic 模型表示，不传裸 dict 或 markdown；
- Analyzer 的 AnalysisReport 直接可被 Generator 的"先分析后改写"流程消费；
- 字段命名保守，留好 reserved 字段，避免扩展时频繁改协议。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


# =====================================================================
# Message：Role 之间最通用的消息载体
# =====================================================================

MessageRole = Literal["user", "system", "analyzer", "generator", "tool"]


class Message(BaseModel):
    """通用消息。

    role:
      - user / system    : 用户与系统提示
      - analyzer / generator : 来自具体 Agent 的产物
      - tool             : 工具调用结果
    """
    msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: MessageRole
    content: str = ""
    # 结构化载荷：若是 AnalysisReport / GenerationSpec / GeneratedShader 等
    # 会在 payload 中放入它们的 model_dump()；接收方按 payload_type 解析
    payload_type: str = ""        # "AnalysisReport" / "GenerationSpec" / "GeneratedShader" / ""
    payload: dict[str, Any] = Field(default_factory=dict)
    # 追踪
    parent_id: str = ""           # 上游消息 id（用于建因果链）
    created_at: float = Field(default_factory=time.time)
    meta: dict[str, Any] = Field(default_factory=dict)

    def short(self, limit: int = 80) -> str:
        c = (self.content or "").replace("\n", " ")
        if len(c) > limit:
            c = c[:limit] + "…"
        return f"[{self.role}] {c}"


# =====================================================================
# 检索结果（来自 ShaderVectorStore）
# =====================================================================

class SimilarShader(BaseModel):
    """检索到的相似 shader 的轻量引用。"""
    shader_id: str
    name: str = ""
    distance: float = 0.0
    tags_topic: list[str] = Field(default_factory=list)
    code_excerpt: str = ""           # 向后兼容旧 UI/旧向量库
    algorithm_summary: str = ""       # 算法摘要，给 Generator 用
    matched_chunks: list[str] = Field(default_factory=list)   # 命中的子块标题列表
    reference_context: str = ""       # 结构化参考上下文（Generator prompt 用）


# =====================================================================
# Analyzer 的产物
# =====================================================================

class AnalysisReport(BaseModel):
    """Analyzer 对一段 shader 代码的结构化分析报告。

    这是 Analyzer 的"最终态"产物，必须满足：
    1. 人类可读：to_markdown() 直接给人看；
    2. 机器可消费：Generator 在"先分析后改写"任务里直接吃这个对象。

    字段语义：
    - source_code     : 被分析的原始 GLSL
    - algorithm_summary: 一段自然语言概述（中文）
    - key_variables   : { 变量名: "用途/语义" } —— Generator 改写时复用
    - techniques      : 技术标签，从受控词表里选（与 corpus.tagger.TOPIC_VOCAB 一致）
    - visual_effect   : 推断的视觉效果描述（中文）
    - similar_shaders : 检索到的相似样本引用
    - section_walkthrough: 按代码段的逐段讲解（key=section标题, value=解释）
    """
    source_code: str
    algorithm_summary: str = ""
    key_variables: dict[str, str] = Field(default_factory=dict)
    techniques: list[str] = Field(default_factory=list)
    visual_effect: str = ""
    similar_shaders: list[SimilarShader] = Field(default_factory=list)
    section_walkthrough: dict[str, str] = Field(default_factory=dict)
    # 追踪
    created_at: float = Field(default_factory=time.time)
    model_used: str = ""  # 调用了哪个 LLM 模型（填充）

    PAYLOAD_TYPE: ClassVar[str] = "AnalysisReport"

    def to_message(self, parent_id: str = "") -> Message:
        return Message(
            role="analyzer",
            content=self.algorithm_summary or "Analysis completed.",
            payload_type=self.PAYLOAD_TYPE,
            payload=self.model_dump(),
            parent_id=parent_id,
        )

    def to_markdown(self) -> str:
        """人类阅读视图（完整 8 段式）。"""
        lines: list[str] = []
        lines.append("# Shader Analysis Report\n")

        # 1. Overview
        meta_bits: list[str] = []
        if self.techniques:
            meta_bits.append("**技术标签**: " + ", ".join(self.techniques))
        if self.model_used:
            meta_bits.append(f"**model**: `{self.model_used}`")
        if meta_bits:
            lines.append(" · ".join(meta_bits) + "\n")

        # 2. Visual Effect
        if self.visual_effect:
            lines.append("## 视觉效果\n")
            lines.append(self.visual_effect + "\n")

        # 3. Algorithm Summary
        if self.algorithm_summary:
            lines.append("## 算法摘要\n")
            lines.append(self.algorithm_summary + "\n")

        # 4. Walkthrough（先提出 "对照参考样本" 作为单独章节）
        comparison = ""
        walkthrough = dict(self.section_walkthrough or {})
        if "对照参考样本" in walkthrough:
            comparison = walkthrough.pop("对照参考样本")
        if walkthrough:
            lines.append("## 分段讲解\n")
            for title, expl in walkthrough.items():
                lines.append(f"### `{title}`\n\n{expl}\n")

        # 5. Key Variables
        if self.key_variables:
            lines.append("## 关键变量\n")
            for k, v in self.key_variables.items():
                lines.append(f"- `{k}` — {v}")
            lines.append("")

        # 6. Similar Shaders
        if self.similar_shaders:
            lines.append("## 相似样本\n")
            for s in self.similar_shaders:
                lines.append(
                    f"- **{s.name}** (`{s.shader_id}`, distance={s.distance:.3f}, "
                    f"tags={','.join(s.tags_topic)})"
                )
            lines.append("")

        # 7. Comparison（如有）
        if comparison:
            lines.append("## 对照参考样本\n")
            lines.append(comparison + "\n")

        # 8. Source
        if self.source_code:
            lines.append("## 源码\n")
            lines.append("```glsl")
            lines.append(self.source_code.strip())
            lines.append("```\n")

        return "\n".join(lines)


# =====================================================================
# Generator 的输入
# =====================================================================

class GenerationSpec(BaseModel):
    """Generator 的结构化输入。

    可由两种途径产生：
    1. 用户自然语言 → SpecParseAction 解析 → GenerationSpec；
    2. AnalysisReport → derived from analysis（"按这个算法重写""换种主题"等组合任务）。

    字段尽量正交，便于 prompt 拼接：
    - description      : 自由文本目标（中文 OK）
    - effect_type      : 受控词表，与 TOPIC_VOCAB 对齐；首版选 0 或 1 个
    - palette          : 主色调描述（"冷色 / 暖色 / 单色 / 霓虹蓝紫" 等）
    - dynamic          : 是否需要随时间变化
    - complexity       : "minimal" / "simple" / "moderate" / "complex"
    - constraints      : 用户提的硬约束（"不要用纹理""单 pass""<150 行" 等）
    - reference_report : 可选；用于"基于这份分析改写"任务，承载 Analyzer 产物
    - base_code        : 可选；改写模式下的原始代码。非空即表示"在此代码基础上做
                         最小化修改"，而不是从零生成。
    - rewrite_mode     : 是否为改写模式（基于 base_code 做最小改动）
    """
    description: str
    effect_type: str = ""             # raymarching / 2d-pattern / fractal / ...
    palette: str = ""                  # "neon blue", "warm sunset", "monochrome"
    dynamic: bool = True
    complexity: Literal["minimal", "simple", "moderate", "complex"] = "simple"
    constraints: list[str] = Field(default_factory=list)
    reference_report: AnalysisReport | None = None
    base_code: str = ""
    rewrite_mode: bool = False

    PAYLOAD_TYPE: ClassVar[str] = "GenerationSpec"

    def to_message(self, parent_id: str = "") -> Message:
        return Message(
            role="user",
            content=self.description,
            payload_type=self.PAYLOAD_TYPE,
            payload=self.model_dump(),
            parent_id=parent_id,
        )


# =====================================================================
# Generator 的产物
# =====================================================================

class CompileResult(BaseModel):
    """GLSL 编译验证结果。"""
    ok: bool = False
    errors: str = ""  # 编译器原文错误
    warnings: str = ""


class GeneratedShader(BaseModel):
    """Generator 最终产物。

    生命周期里可能经过多轮 generate → compile → fix 循环，
    本字段只保存"最终接受"的那一版；过程产物在 Memory 中。
    """
    code: str
    explanation: str = ""               # 给用户看的简短说明
    spec: GenerationSpec | None = None  # 派生自哪份 spec
    compile_result: CompileResult = Field(default_factory=CompileResult)
    iterations: int = 0                 # 修正循环跑了几轮
    references_used: list[SimilarShader] = Field(default_factory=list)
    # 自评结果（启用时填充，未启用时 score=0.0）
    self_critique_score: float = 0.0
    self_critique_rationale: str = ""
    # 追踪
    created_at: float = Field(default_factory=time.time)
    model_used: str = ""

    PAYLOAD_TYPE: ClassVar[str] = "GeneratedShader"

    def to_message(self, parent_id: str = "") -> Message:
        return Message(
            role="generator",
            content=self.explanation or "Generated.",
            payload_type=self.PAYLOAD_TYPE,
            payload=self.model_dump(),
            parent_id=parent_id,
        )

    def to_markdown(self) -> str:
        parts = ["# Generated Shader\n"]
        if self.explanation:
            parts.append(self.explanation + "\n")
        parts.append("```glsl\n" + self.code.strip() + "\n```\n")
        meta_bits: list[str] = []
        if self.iterations:
            meta_bits.append(f"iterations: {self.iterations}")
        if self.compile_result is not None:
            meta_bits.append(f"compile_ok: {self.compile_result.ok}")
        if self.model_used:
            meta_bits.append(f"model: `{self.model_used}`")
        if meta_bits:
            parts.append("_" + " · ".join(meta_bits) + "_\n")
        if self.self_critique_rationale:
            parts.append("## 自评\n")
            parts.append(f"score: **{self.self_critique_score:.2f}** — "
                        f"{self.self_critique_rationale}\n")
        if self.references_used:
            parts.append("**References**: " + ", ".join(
                f"{s.name}({s.shader_id})" for s in self.references_used
            ) + "\n")
        return "\n".join(parts)
