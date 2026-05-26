"""角色私有的简易记忆模块。

设计极简：
  - 一条 Memory 对应一个 Role 实例；
  - 内部存的是 Message 列表；
  - 提供按 role / payload_type 过滤、按时间 / id 检索、按 parent 链回溯；
  - 不做持久化；阶段六以后若需要长程记忆再升级。

性能：列表 O(n) 检索对当前规模（单会话 <100 条）完全够。
"""
from __future__ import annotations

from typing import Iterable

from shader_agent.agents.schemas import Message


class Memory:
    """单角色的会话记忆。"""

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
        return f"Memory(n={len(self._items)})"
