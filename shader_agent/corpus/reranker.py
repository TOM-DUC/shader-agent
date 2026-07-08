"""重排器：对融合候选做精排。

两种实现，自动选择：
- 交叉编码器（BAAI/bge-reranker-v2-m3）：把 (query, doc) 成对打分，精度最高；
  需要 sentence-transformers 的 CrossEncoder 与一次模型下载，约数百 MB。
- 确定性回退：当交叉编码器不可用（未装依赖 / 无网络 / 无显存）时，用候选自身的
  融合分加上轻量启发式（标题/函数名与 query 的词重叠）做稳定排序，保证全链路可跑。

调用方只关心 ``Reranker.rerank(query, candidates)``，无需关心底层走了哪条路径。
"""
from __future__ import annotations

import threading
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
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _try_load(self) -> bool:
        if self._model is not None:
            return True
        if self._tried or not self.enabled:
            return False
        # 关键：整个加载过程都在锁内完成。交叉编码器权重很大（~2.2GB），
        # 若像旧实现那样"在锁外加载"，并发的预热线程与首个请求线程会各自
        # 认为自己是加载者，导致同一模型被加载两次（日志中可见同一秒内两条
        # "cross-encoder loaded"）。锁内加载使后到的线程直接复用已加载实例。
        with self._load_lock:
            if self._model is not None:
                return True
            if self._tried or not self.enabled:
                return False
            self._tried = True
            try:
                import os

                from sentence_transformers import CrossEncoder

                offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                import torch
                ce_device = "cuda" if torch.cuda.is_available() else "cpu"
                from shader_agent.config.settings import settings as _st
                _cache = str(_st.embedding.cache_dir) if _st.embedding.cache_dir else None
                self._model = CrossEncoder(
                    self.model_name,
                    device=ce_device,
                    local_files_only=offline,
                    cache_folder=_cache,
                )
                logger.info(f"[rerank] cross-encoder loaded: {self.model_name} (device={ce_device})")
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
        with self._predict_lock:
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


# 进程级单例缓存：交叉编码器权重很大（~2.2GB），复用避免重复加载
_RERANKER_INSTANCE: Reranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", enabled: bool = True) -> Reranker:
    """获取重排器单例。首次调用创建并缓存实例，之后复用。

    若单例已存在但此前以 enabled=False 创建，而本次请求 enabled=True，
    则就地把它启用（不丢弃实例、不重复加载），保证"先关后开"也能生效。
    """
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        with _RERANKER_LOCK:
            if _RERANKER_INSTANCE is None:
                _RERANKER_INSTANCE = Reranker(model_name=model_name, enabled=enabled)
    elif enabled and not _RERANKER_INSTANCE.enabled:
        with _RERANKER_LOCK:
            _RERANKER_INSTANCE.enabled = True
            _RERANKER_INSTANCE._tried = False  # 允许重新尝试加载
    return _RERANKER_INSTANCE
