"""Gradio UI 的公共入口。

Gradio 属于可选依赖。导入 runners、examples 等非界面模块时，
不应该连带要求安装 gradio。
"""
from __future__ import annotations

from typing import Any

__all__ = ["build_app", "launch"]


def build_app(*args: Any, **kwargs: Any):
    """懒加载并构建 Gradio 应用。"""
    from shader_agent.ui.app import build_app as _build_app

    return _build_app(*args, **kwargs)


def launch(*args: Any, **kwargs: Any):
    """懒加载并启动 Gradio 应用。"""
    from shader_agent.ui.app import launch as _launch

    return _launch(*args, **kwargs)
