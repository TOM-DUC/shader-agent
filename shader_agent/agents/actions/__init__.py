"""Action 子包：每个 Action 是一个最小化、可单测、有 schema 的工作单元。

Analyzer 与 Generator 的核心逻辑
里会把 _run() 里的 LLM 调用接通。
"""
from shader_agent.agents.actions.base import Action, ActionResult  # noqa: F401
