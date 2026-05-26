"""本地嵌入模型封装：默认 BAAI/bge-m3。

为什么选 bge-m3：
- 多语言（中英都强），后续 user query 多半中文，shader 文本以英文为主，跨语言能力关键；
- 1024 维，单 query 检索成本可控；
- 在 HuggingFace 上权重公开，sentence-transformers 直接加载。

性能：
- 模型权重 ~2.3GB，首次下载耗时，请提前 `python -m scripts.download_embedder`；
- CPU 上每条 ~50ms，小语料库够用；如有 GPU 自动启用 cuda。

依赖：sentence-transformers + transformers + torch。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from shader_agent.config.settings import settings
from shader_agent.utils.logger import logger


def _detect_device(preference: str = "auto") -> str:
    """决定加载设备。"""
    if preference != "auto":
        return preference
    try:
        import torch  # 延迟导入
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Mac Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbedder:
    """sentence-transformers 包装。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        normalize: bool | None = None,
        batch_size: int | None = None,
    ) -> None:
        cfg = settings.embedding
        self.model_name = model_name or cfg.model_name
        self.device = _detect_device(device or cfg.device)
        self.normalize = cfg.normalize_embeddings if normalize is None else normalize
        self.batch_size = batch_size or cfg.batch_size

        if cache_dir is None and cfg.cache_dir:
            cache_dir = settings.project_root / cfg.cache_dir
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._lock = threading.Lock()

    # ---------- 懒加载 ----------
    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info(
                f"[embedder] loading model={self.model_name} device={self.device} "
                f"cache_dir={self.cache_dir}"
            )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers 未安装；请先 `pip install -r requirements.txt`"
                ) from e
            # 离线开关：HF_HUB_OFFLINE 由 shader_agent/_hf_offline.py 在最早期设置。
            # 这里据此把 local_files_only 透传给 SentenceTransformer，
            # 彻底跳过对 huggingface.co 的 HEAD 校验，避免启动卡顿。
            import os
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            logger.info(
                f"[embedder] offline={offline} "
                f"(HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')})"
            )
            st_kwargs: dict = dict(
                device=self.device,
                cache_folder=str(self.cache_dir) if self.cache_dir else None,
            )
            if offline:
                # sentence-transformers>=2.3 / transformers 均支持该参数
                st_kwargs["local_files_only"] = True
            try:
                self._model = SentenceTransformer(self.model_name, **st_kwargs)
            except TypeError:
                # 老版本不接受 local_files_only：去掉后重试
                st_kwargs.pop("local_files_only", None)
                self._model = SentenceTransformer(self.model_name, **st_kwargs)
            # 探测维度（新版方法名为 get_embedding_dimension，旧版为
            # get_sentence_embedding_dimension；优先用新版避免 FutureWarning）
            dim = 0
            for meth in ("get_embedding_dimension",
                         "get_sentence_embedding_dimension"):
                fn = getattr(self._model, meth, None)
                if callable(fn):
                    try:
                        dim = int(fn() or 0)
                        break
                    except Exception:
                        continue
            logger.info(f"[embedder] loaded; embedding_dim={dim}")

    # ---------- 主接口 ----------
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """编码一批文本，返回 (N, D) ndarray。"""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        assert self._model is not None
        arr = self._model.encode(
            texts_list,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts_list) > 16,
        )
        return arr.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """单条文本编码，返回 (D,) ndarray。"""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        assert self._model is not None
        for meth in ("get_embedding_dimension",
                     "get_sentence_embedding_dimension"):
            fn = getattr(self._model, meth, None)
            if callable(fn):
                try:
                    return int(fn() or 0)
                except Exception:
                    continue
        return 0


# 全局单例（懒加载，不在 import 时下载模型）
_embedder_singleton: BGEEmbedder | None = None


def get_embedder() -> BGEEmbedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = BGEEmbedder()
    return _embedder_singleton
