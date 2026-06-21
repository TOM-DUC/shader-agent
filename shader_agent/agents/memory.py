"""角色私有的工作记忆模块。

承担单次运行内的消息记录：一条 WorkingMemory 对应一个 Role 实例，内部存 Message
列表，提供按 role / payload_type 过滤、按时间 / id 检索、按 parent 链回溯。

这是"会话内短程记忆"，不做持久化；跨会话的情节记忆与经验记忆由 shader_agent.memory
子系统负责，两者职责分明，互不混淆。

性能：列表 O(n) 检索对单会话规模（<100 条）完全够用。
"""
from __future__ import annotations

from typing import Iterable

from shader_agent.agents.schemas import Message


class WorkingMemory:
    """单角色的会话内工作记忆。"""

    def __init__(self) -> None:
        self._items: list[Message] = []

    # ---------- 写 ----------
    def add(self, msg: Message) -> None:
        self._items.append(msg)

    def extend(self, msgs: Iterable[Message]) -> None:
        for m in msgs:
            self.add(m)

    # ---------- 读 ----------
    def all(self) -> list[Message]:
        return list(self._items)

    def latest(self, n: int = 1) -> list[Message]:
        if n <= 0:
            return []
        return self._items[-n:]

    def by_role(self, role: str) -> list[Message]:
        return [m for m in self._items if m.role == role]

    def by_payload_type(self, payload_type: str) -> list[Message]:
        return [m for m in self._items if m.payload_type == payload_type]

    def find(self, msg_id: str) -> Message | None:
        for m in self._items:
            if m.msg_id == msg_id:
                return m
        return None

    def lineage(self, msg_id: str) -> list[Message]:
        """回溯一条消息的所有祖先（按时间顺序）。"""
        chain: list[Message] = []
        cur = self.find(msg_id)
        guard = 0
        while cur is not None and guard < 1024:
            chain.append(cur)
            if not cur.parent_id:
                break
            cur = self.find(cur.parent_id)
            guard += 1
        return list(reversed(chain))

    # ---------- 维护 ----------
    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"WorkingMemory(n={len(self._items)})"


# 向后兼容别名：旧代码以 `Memory` 引用工作记忆，保持不破坏导入。
Memory = WorkingMemory
