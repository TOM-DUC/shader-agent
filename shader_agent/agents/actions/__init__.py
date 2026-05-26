"""Action 子包：每个 Action 是一个最小化、可单测、有 schema 的工作单元。

阶段三只放骨架与"占位实现"，阶段四（Analyzer 真正干活）与阶段五（Generator）
里会把 _run() 里的 LLM 调用接通。
"""
from shader_agent.agents.actions.base import Action, ActionResult  # noqa: F401
