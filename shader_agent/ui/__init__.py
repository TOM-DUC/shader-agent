"""Gradio 三标签页 UI。

对外仅暴露 `launch()` 函数，由 `scripts/run_ui.py` 调用。
"""
from shader_agent.ui.app import build_app, launch  # noqa: F401
