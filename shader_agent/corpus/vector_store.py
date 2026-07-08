"""ChromaDB 持久化向量库封装。

设计：
- 使用 PersistentClient，数据保存到 data/vector_db/；
- 不用 ChromaDB 自带的嵌入函数，自己用 BGEEmbedder 提前算好 embedding 再 upsert，
  好处：完全可控、可替换、不被 chroma 默认 onnx 模型偷偷下载；
- 同时支持两种粒度：
  * 旧的 shader 级（一条 shader 一个向量，向后兼容）；
  * 新的子块级（父子分块，检索精度更高）。子块向量带 parent_id 元数据，
    命中后由 ParentDocumentStore 回溯完整 shader。
- 元数据（用于过滤）来自 ShaderRecord.to_metadata()；
- ChromaDB cosine 距离配置在 collection metadata 的 hnsw:space。

检索接口：
  - query_by_text(text, top_k, where=None) -> list[dict]  # shader 级
  - query_chunks(text, top_k, where=None) -> list[dict]   # 子块级
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shader_agent.config.settings import settings
from shader_agent.corpus.chunker import ShaderChunk, chunk_shader
from shader_agent.corpus.models import ShaderRecord
from shader_agent.embeddings.bge_embedder import BGEEmbedder, get_embedder
from shader_agent.utils.logger import logger


class ShaderVectorStore:
    """围绕一个 chroma collection 的薄封装。"""

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedder: BGEEmbedder | None = None,
        distance: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.vector_db_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.distance = distance or settings.vector_store.distance
        self.embedder = embedder or get_embedder()

        import chromadb  # 延迟导入
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        # get_or_create 时确保 hnsw:space 一致
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        self._chunk_coll = None  # 子块集合惰性创建
        logger.info(
            f"[vstore] persist_dir={self.persist_dir} "
            f"collection={self.collection_name} "
            f"distance={self.distance} "
            f"count={self._collection.count()}"
        )

    # ---------- 写入 ----------
    def upsert(self, records: list[ShaderRecord]) -> int:
        if not records:
            return 0
        ids = [r.shader_id for r in records]
        docs = [r.to_doc_text() for r in records]
        metas = [r.to_metadata() for r in records]
        embs: np.ndarray = self.embedder.embed(docs)
        # 同步 dim 信息回 record（仅追踪用）
        for r in records:
            r.embedding_dim = int(embs.shape[1])
        self._collection.upsert(
            ids=ids,
            embeddings=embs.tolist(),
            documents=docs,
            metadatas=metas,
        )
        logger.info(f"[vstore] upserted {len(records)} records; total={self._collection.count()}")
        return len(records)

    # ---------- 检索 ----------
    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q_emb = self.embedder.embed([text])
        res = self._collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            where=where or None,
        )
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        for i, sid in enumerate(ids):
            out.append({
                "shader_id": sid,
                "distance": float(distances[i]) if i < len(distances) else None,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "document": documents[i] if i < len(documents) else "",
            })
        return out

    # ---------- 杂项 ----------
    def count(self) -> int:
        return self._collection.count()

    # ---------- 子块级索引（父子分块） ----------
    def _chunk_collection(self):
        """惰性获取/创建子块集合（与主集合分开，互不污染 Top-K）。"""
        if getattr(self, "_chunk_coll", None) is None:
            self._chunk_coll = self._client.get_or_create_collection(
                name=f"{self.collection_name}_chunks",
                metadata={"hnsw:space": self.distance},
            )
        return self._chunk_coll

    def upsert_chunks(self, records: list[ShaderRecord], batch_size: int = 5000) -> int:
        """把每条 shader 切成父子子块并写入子块集合。返回写入子块数。

        Args:
            batch_size: ChromaDB 单次 upsert 上限（默认 5000 < 5461 限制）。
        """
        if not records:
            return 0
        coll = self._chunk_collection()
        all_chunks: list[ShaderChunk] = []
        for r in records:
            all_chunks.extend(chunk_shader(r))
        if not all_chunks:
            return 0

        total = 0
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start:start + batch_size]
            ids = [c.chunk_id for c in batch]
            docs = [c.text for c in batch]
            metas = [
                {
                    "parent_id": c.parent_id,
                    "kind": c.kind,
                    "title": c.title,
                    **{k: v for k, v in c.meta.items() if isinstance(v, (str, int, float, bool))},
                }
                for c in batch
            ]
            embs: np.ndarray = self.embedder.embed(docs)
            coll.upsert(ids=ids, embeddings=embs.tolist(), documents=docs, metadatas=metas)
            total += len(batch)
            logger.info(
                f"[vstore] batch upserted {len(batch)} chunks ({total}/{len(all_chunks)})"
            )

        logger.info(
            f"[vstore] upserted {total} chunks from {len(records)} shaders; "
            f"chunk total={coll.count()}"
        )
        return total

    def query_chunks(
        self,
        text: str,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """子块级检索。返回 [{chunk_id, parent_id, kind, title, distance, document}]。"""
        coll = self._chunk_collection()
        if coll.count() == 0:
            return []
        q_emb = self.embedder.embed([text])
        res = coll.query(query_embeddings=q_emb.tolist(), n_results=top_k, where=where or None)
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        for i, cid in enumerate(ids):
            md = metadatas[i] if i < len(metadatas) else {}
            out.append(
                {
                    "chunk_id": cid,
                    "parent_id": md.get("parent_id", ""),
                    "kind": md.get("kind", ""),
                    "title": md.get("title", ""),
                    "distance": float(distances[i]) if i < len(distances) else None,
                    "document": documents[i] if i < len(documents) else "",
                }
            )
        return out

    def chunk_count(self) -> int:
        try:
            return self._chunk_collection().count()
        except Exception:
            return 0

    def reset(self) -> None:
        """清空当前 collection（保留持久化目录）。"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        # 子块集合一并清空
        try:
            self._client.delete_collection(f"{self.collection_name}_chunks")
        except Exception:
            pass
        self._chunk_coll = None
        logger.info(f"[vstore] collection '{self.collection_name}' reset")
