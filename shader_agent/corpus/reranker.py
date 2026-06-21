"""重排器：对融合候选做精排。

两种实现，自动选择：
- 交叉编码器（BAAI/bge-reranker-v2-m3）：把 (query, doc) 成对打分，精度最高；
  需要 sentence-transformers 的 CrossEncoder 与一次模型下载，约数百 MB。
- 确定性回退：当交叉编码器不可用（未装依赖 / 无网络 / 无显存）时，用候选自身的
  融合分加上轻量启发式（标题/函数名与 query 的词重叠）做稳定排序，保证全链路可跑。

调用方只关心 ``Reranker.rerank(query, candidates)``，无需关心底层走了哪条路径。
"""
from __future__ import annotations

from typing import Any

from shader_agent.corpus.glsl_tokenize import tokenize_glsl
from shader_agent.utils.logger import logger


class Reranker:
    """对候选列表精排。candidates 中每项需含 'text' 与 'fused_score'。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._tried = False

    def _try_load(self) -> bool:
        if self._model is not None:
            return True
        if self._tried or not self.enabled:
            return False
        self._tried = True
        try:
            import os

            from sentence_transformers import CrossEncoder

            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            self._model = CrossEncoder(
                self.model_name,
                local_files_only=offline,
                cache_folder="data/models",
            )
            logger.info(f"[rerank] cross-encoder loaded: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"[rerank] cross-encoder unavailable, fallback heuristic: {e}")
            self._model = None
            return False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self._try_load():
            scored = self._rerank_cross_encoder(query, candidates)
        else:
            scored = self._rerank_heuristic(query, candidates)
        scored.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k] if top_k else scored

    def _rerank_cross_encoder(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return candidates

    def _rerank_heuristic(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """无交叉编码器时的确定性排序：融合分 + query 词重叠加权。"""
        q_tokens = set(tokenize_glsl(query))
        for c in candidates:
            base = float(c.get("fused_score", 0.0))
            doc_tokens = set(tokenize_glsl(f"{c.get('title','')} {c.get('text','')}"))
            overlap = len(q_tokens & doc_tokens) / (len(q_tokens) + 1)
            c["rerank_score"] = base + 0.3 * overlap
        return candidates
