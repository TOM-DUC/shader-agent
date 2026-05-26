"""预下载 / 预热嵌入模型。

第一次构建知识库前推荐先跑这个脚本，避免 build_corpus 卡在下载上：
    python -m scripts.download_embedder

模型默认从 HuggingFace 拉取到：
  - settings.embedding.cache_dir（项目内 data/models/）
若网络环境受限，可在 .env 设置 HF_ENDPOINT=https://hf-mirror.com 切镜像。

注意：本脚本会强制开启"在线模式"（设置 SHADER_AGENT_HF_ONLINE=1），
即使本地已有缓存也允许联网校验/补全文件。日常运行 UI 时不要设这个变量，
shader_agent 会自动走离线，避免 huggingface.co 连接超时拖慢启动。
"""
from __future__ import annotations

import os
import sys

# 必须在 import shader_agent 之前设置，确保 _hf_offline 判定为"在线"
os.environ.setdefault("SHADER_AGENT_HF_ONLINE", "1")

from rich.console import Console

from shader_agent.config.settings import settings
from shader_agent.embeddings.bge_embedder import get_embedder

console = Console()


def main() -> int:
    console.rule("[bold cyan]Downloading / loading embedder")
    console.print(f"model_name  = {settings.embedding.model_name}")
    console.print(f"device      = {settings.embedding.device}")
    console.print(f"cache_dir   = {settings.embedding.cache_dir}")
    console.print(f"normalize   = {settings.embedding.normalize_embeddings}")
    console.print(f"batch_size  = {settings.embedding.batch_size}")
    console.print()

    embedder = get_embedder()
    # 一次假调用触发懒加载
    vec = embedder.embed_one("warm up: a shader that draws a red circle")
    console.print(f"[green]OK[/green] embedding_dim = {vec.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
