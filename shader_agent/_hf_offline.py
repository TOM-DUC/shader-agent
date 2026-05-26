"""HuggingFace 离线模式开关（修复启动卡顿）。

问题背景
========
即便 bge-m3 权重已经缓存在 data/models 下，sentence-transformers /
transformers / huggingface_hub 仍会在加载时向 huggingface.co 发起 HEAD 请求
（adapter_config.json / processor_config.json 等）以及后台 `auto_conversion`
线程做 safetensors 转换。在国内/无外网环境，这些请求会触发
`[WinError 10060]` 连接超时，每个文件 5 次指数退避（1+2+4+8+8≈23s），
多个文件叠加导致启动等待 1 分钟以上。

解决方案
========
在**任何** transformers / sentence-transformers 被 import 之前，设置离线环境
变量。本模块在 `shader_agent/__init__.py` 顶部最先被导入，确保时机正确。

行为
====
- 默认：若检测到本地已存在 bge-m3 快照，则自动开启离线模式（推荐）。
- 强制在线（首次下载模型时用）：设置环境变量 SHADER_AGENT_HF_ONLINE=1。
- 强制离线（无论是否检测到缓存）：设置环境变量 SHADER_AGENT_HF_OFFLINE=1。
"""
from __future__ import annotations

import os
from pathlib import Path


def _has_local_model(cache_dir: Path, model_name: str = "BAAI/bge-m3") -> bool:
    """粗略判断本地是否已有该模型的快照。

    sentence-transformers 的缓存目录里，模型名会被转写为
    `models--BAAI--bge-m3` 这样的形式（HF hub 标准布局）。
    也兼容直接放在 cache_dir 下的情况。
    """
    if not cache_dir.exists():
        return False
    hub_style = "models--" + model_name.replace("/", "--")
    candidates = [
        cache_dir / hub_style,
        cache_dir / model_name.replace("/", "_"),
        cache_dir / model_name.split("/")[-1],
    ]
    for c in candidates:
        if c.exists():
            return True
    # 兜底：cache_dir 下任意目录里出现 config.json + 权重即认为有
    for p in cache_dir.rglob("config.json"):
        sib = p.parent
        if any((sib / w).exists() for w in
               ("model.safetensors", "pytorch_model.bin", "modules.json")):
            return True
    return False


def apply() -> dict[str, str]:
    """根据环境与本地缓存决定是否开启离线模式，并写入环境变量。

    返回最终生效的关键环境变量快照，便于日志记录。
    """
    force_online = os.environ.get("SHADER_AGENT_HF_ONLINE", "") == "1"
    force_offline = os.environ.get("SHADER_AGENT_HF_OFFLINE", "") == "1"

    # 计算模型缓存目录：默认 <project_root>/data/models
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(
        os.environ.get("SHADER_AGENT_MODELS_DIR", project_root / "data" / "models")
    )

    if force_online:
        offline = False
    elif force_offline:
        offline = True
    else:
        # 默认策略：本地有缓存 → 离线；没有 → 在线（允许首次下载）
        offline = _has_local_model(cache_dir)

    # 关闭遥测与隐式 token（无论在线离线，都减少无谓网络）
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # 关闭进度条以外的多余检查
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if offline:
        # 这三个是关键：阻止所有对 huggingface.co 的 HEAD / 转换请求
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 禁用 safetensors 自动转换后台线程（auto_conversion）
        os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")

    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
        "offline_decided": "1" if offline else "0",
        "cache_dir": str(cache_dir),
    }


# 模块被 import 时立即生效
HF_STATE = apply()
