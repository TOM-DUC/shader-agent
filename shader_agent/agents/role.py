"""Role 抽象基类。

借鉴 MetaGPT 的 Role：
  - 每个 Role 有 name、system_prompt、actions 列表、私有 Memory；
  - 主入口 handle(message) → 决定走哪个 Action 序列 → 返回新 Message；
  - 通过 register_action() 在 __init__ 里组装 Action。

阶段三：Role 的 handle() 由子类决定调用顺序。
阶段六（如需）：可以加 ReAct/Plan-Act 风格的动态选择，但目前固定流水线足够。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shader_agent.agents.actions.base import Action, ActionResult
from shader_agent.agents.memory import Memory
from shader_agent.agents.schemas import Message
from shader_agent.utils.logger import logger


class Role(ABC):
    """所有 Agent 角色的基类。"""

    # 子类覆盖
    role_name: str = "Role"
    system_prompt: str = ""

    def __init__(self) -> None:
        self.memory = Memory()
        self.actions: dict[str, Action] = {}
        self._setup_actions()

    # ---------- Action 注册 ----------
    @abstractmethod
    def _setup_actions(self) -> None:
        """子类在这里实例化并注册自己的 Action。"""
        raise NotImplementedError

    def register_action(self, action: Action) -> None:
        if action.name in self.actions:
            logger.warning(f"[role] {self.role_name}: action '{action.name}' overwritten")
        self.actions[action.name] = action

    def get_action(self, name: str) -> Action:
        if name not in self.actions:
            raise KeyError(f"{self.role_name} has no action named '{name}'")
        return self.actions[name]

    def run_action(self, name: str, inp: Any) -> ActionResult:
        """执行一个已注册的 Action。"""
        return self.get_action(name).run(inp)

    # ---------- 主入口 ----------
    @abstractmethod
    def handle(self, message: Message) -> Message:
        """处理一条消息，返回响应消息。子类实现自己的流水线。"""
        raise NotImplementedError

    # ---------- 工具 ----------
    def observe(self, message: Message) -> None:
        """把外部输入吸入记忆。"""
        self.memory.add(message)

    def __repr__(self) -> str:
        return f"<{self.role_name} actions={list(self.actions)} mem={len(self.memory)}>"
