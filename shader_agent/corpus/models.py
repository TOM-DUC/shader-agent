"""ShaderRecord 数据模型。

统一描述一条 shader 记录，贯穿采集、清洗、打标、质量验证、向量化整条流水线。
质量与来源字段用于在检索时做过滤与可追溯，使知识库只保留经过验证的高质量参考。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RenderPass(BaseModel):
    """单个 render pass（Shadertoy 一个 shader 可能有多 pass）。"""
    name: str = ""
    type: str = ""  # image / buffer / common / sound / cubemap
    code: str = ""
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)


class ShaderRecord(BaseModel):
    """一条 shader 的完整描述。"""

    # ---- 来自 Shadertoy 的原始字段 ----
    shader_id: str
    name: str
    username: str = ""
    description: str = ""
    likes: int = 0
    viewed: int = 0
    tags_raw: list[str] = Field(default_factory=list)  # 原始 tags
    passes: list[RenderPass] = Field(default_factory=list)
    source: str = "shadertoy"  # 来源：shadertoy / seed / local

    # ---- 清洗与打标产物 ----
    code_image: str = ""           # 仅 Image pass 的代码（首版主要用）
    code_common: str = ""          # 可选的 Common pass
    code_hash: str = ""            # 去重用
    tags_topic: list[str] = Field(default_factory=list)  # 标签：raymarching/sdf/...
    has_external_assets: bool = False  # texture/cubemap/keyboard/buffer 等外部输入

    # ---- 来源与许可（可追溯）----
    source_url: str = ""           # 原始链接，便于回溯与署名
    license: str = ""              # 许可证标识（空表示未知）

    # ---- 质量验证产物（决定是否进入高质量参考库）----
    compile_ok: bool = False       # 是否通过静态/真实编译
    render_ok: bool = False        # 是否成功渲染出非空帧
    quality_score: float = 0.0     # 综合质量分（0~1）

    # ---- 静态分析产物（供检索与父子分块使用）----
    algorithm_summary: str = ""    # 算法摘要（静态启发式或可选 LLM）
    key_functions: list[str] = Field(default_factory=list)  # 关键自定义函数名
    visual_features: list[str] = Field(default_factory=list)  # 视觉特征关键词
    indexed_at: str = ""           # 入库时间（ISO 字符串）

    # ---- 向量化产物（不进 vector store 元数据，仅追踪用）----
    embedding_dim: int = 0

    # ---------- 工具方法 ----------
    def compute_code_hash(self) -> str:
        """基于 image+common 代码计算 sha256，用于去重。"""
        h = hashlib.sha256()
        h.update(self.code_image.encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
        h.update(self.code_common.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def to_doc_text(self) -> str:
        """拼接成喂给嵌入模型的文档文本。

        策略：把可检索性最强的字段（name / description / tags / 算法摘要 / 部分代码）
        拼成一段 plain text。**不把全部代码灌入**，因为 shader 代码长，向量化噪声大。
        """
        parts: list[str] = []
        parts.append(f"Name: {self.name}")
        if self.username:
            parts.append(f"Author: {self.username}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.tags_topic:
            parts.append(f"Topics: {', '.join(self.tags_topic)}")
        if self.tags_raw:
            parts.append(f"Raw tags: {', '.join(self.tags_raw)}")

        # 摘录代码前若干字符（包含 mainImage 头部，是结构最强的部分）
        code_snippet = (self.code_image or "")[:1500]
        if code_snippet:
            parts.append(f"Code excerpt:\n{code_snippet}")
        return "\n\n".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        """ChromaDB metadata 不能含 list/dict，需要扁平化。"""
        return {
            "shader_id": self.shader_id,
            "name": self.name,
            "username": self.username,
            "likes": int(self.likes),
            "viewed": int(self.viewed),
            "source": self.source,
            "tags_topic": ",".join(self.tags_topic),
            "tags_raw": ",".join(self.tags_raw),
            "has_external_assets": bool(self.has_external_assets),
            "code_chars": len(self.code_image or ""),
            # 质量与来源：用于检索时过滤与排序、结果可追溯
            "source_url": self.source_url,
            "license": self.license,
            "compile_ok": bool(self.compile_ok),
            "render_ok": bool(self.render_ok),
            "quality_score": float(self.quality_score),
            "key_functions": ",".join(self.key_functions),
            "visual_features": ",".join(self.visual_features),
        }

    def mark_indexed(self) -> None:
        """记录入库时间（ISO8601，UTC）。"""
        self.indexed_at = datetime.now(timezone.utc).isoformat()

    def save_json(self, dst_dir: Path) -> Path:
        """单条记录落盘成 JSON（便于 diff / 复跑）。"""
        dst_dir.mkdir(parents=True, exist_ok=True)
        path = dst_dir / f"{self.shader_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_json(cls, path: Path) -> "ShaderRecord":
        with path.open("r", encoding="utf-8") as f:
            return cls(**json.load(f))
