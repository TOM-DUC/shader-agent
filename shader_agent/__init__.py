"""shader_agent 包初始化。

重要：本文件第一行就导入 `_hf_offline`，在任何 transformers /
sentence-transformers 被加载之前设置 HuggingFace 离线环境变量，
彻底消除启动时对 huggingface.co 的连接超时（[WinError 10060]）。
请勿在此之前放置任何会触发 transformers 导入的语句。
"""
from __future__ import annotations

# 必须最先执行：设置 HF 离线开关
from shader_agent import _hf_offline as _hf_offline  # noqa: F401

__all__ = ["_hf_offline"]
