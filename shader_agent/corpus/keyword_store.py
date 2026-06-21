"""BM25 关键词检索（基于 rank-bm25 的内存索引）。

为什么不用 SQLite FTS5：本项目语料库规模在数十到数百条，FTS5 需要额外维护
数据库文件与分词器，且其默认分词器对 ``sdSphere`` / ``calcNormal`` 这类 GLSL
标识符切分很差。这里改用一个面向 GLSL 的分词器（见 glsl_tokenize）配合 rank-bm25，
零额外文件、检索质量更贴合代码语义。

索引粒度与向量库一致：子块（ShaderChunk）。命中后由父文档存储回溯完整 shader。
索引以 JSON 形式持久化到 data/keyword_index.json，build 时写、运行时读。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shader_agent.config.settings import settings
from shader_agent.corpus.chunker import ShaderChunk
from shader_agent.corpus.glsl_tokenize import tokenize_glsl
from shader_agent.utils.logger import logger


class KeywordStore:
    """BM25 关键词索引。

    用法：
        store = KeywordStore()
        store.build(chunks)        # 建库
        store.save()               # 落盘
        store = KeywordStore.load()  # 运行时加载
        hits = store.search("calcNormal raymarch", top_k=20)
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path or (settings.project_root / "data" / "keyword_index.json")
        # 每个元素：{chunk_id, parent_id, kind, title, tokens}
        self._docs: list[dict[str, Any]] = []
        self._bm25 = None  # 延迟构建

    # ---------- 建库 ----------
    def build(self, chunks: list[ShaderChunk]) -> int:
        self._docs = []
        for c in chunks:
            tokens = tokenize_glsl(f"{c.title}\n{c.text}")
            if not tokens:
                continue
            self._docs.append(
                {
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "kind": c.kind,
                    "title": c.title,
                    "text": c.text,
                    "tokens": tokens,
                }
            )
        self._bm25 = None
        logger.info(f"[keyword] built BM25 index over {len(self._docs)} chunks")
        return len(self._docs)

    def _ensure_bm25(self) -> bool:
        if self._bm25 is not None:
            return True
        if not self._docs:
            return False
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("[keyword] rank-bm25 未安装，关键词检索降级为空结果")
            return False
        self._bm25 = BM25Okapi([d["tokens"] for d in self._docs])
        return True

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """返回 [{chunk_id, parent_id, kind, title, score}, ...]，按 BM25 分降序。"""
        if not self._ensure_bm25():
            return []
        q_tokens = tokenize_glsl(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        out: list[dict[str, Any]] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            d = self._docs[i]
            out.append(
                {
                    "chunk_id": d["chunk_id"],
                    "parent_id": d["parent_id"],
                    "kind": d["kind"],
                    "title": d["title"],
                    "score": float(scores[i]),
                    "document": d.get("text", ""),
                }
            )
        return out

    # ---------- 持久化 ----------
    def save(self) -> Path:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump({"docs": self._docs}, f, ensure_ascii=False)
        logger.info(f"[keyword] saved index -> {self.index_path}")
        return self.index_path

    @classmethod
    def load(cls, index_path: Path | None = None) -> "KeywordStore":
        store = cls(index_path)
        if store.index_path.exists():
            try:
                with store.index_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                store._docs = list(data.get("docs", []))
                logger.info(
                    f"[keyword] loaded index ({len(store._docs)} chunks) "
                    f"from {store.index_path}"
                )
            except Exception as e:
                logger.warning(f"[keyword] load failed: {e}")
        return store

    def count(self) -> int:
        return len(self._docs)
