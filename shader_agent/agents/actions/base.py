"""Action 抽象基类。

参考 MetaGPT 的 Action 思想，但裁剪掉它的 InstructContent / SerializationMixin。

每个 Action 都是：
  - 一个明确的 (input → output) 转换；
  - 输入输出都有 pydantic schema（在子类中声明）；
  - 可单独被单测；
  - 可在 Role 中按顺序调用，也可被组合任务复用。

子类应实现 _run()，基类提供 run() 做异常包装、日志、重试。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from shader_agent.observability import trace_span
from shader_agent.utils.logger import logger


def _brief(obj: Any, limit: int = 400) -> Any:
    """把 Action 输入压成可读且不过长的 span 输入摘要。"""
    try:
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
        elif isinstance(obj, dict):
            data = dict(obj)
        else:
            return str(obj)[:limit]
        out: dict[str, Any] = {}
        for k, v in data.items():
            s = v if isinstance(v, (int, float, bool)) else str(v)
            if isinstance(s, str) and len(s) > limit:
                s = s[:limit] + "…"
            out[k] = s
        return out
    except Exception:
        return str(obj)[:limit]


# 输入 / 输出 类型变量
TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


class ActionResult(BaseModel, Generic[TOut]):
    """Action 的统一返回壳。

    成功时 ok=True, data=输出; 失败时 ok=False, error=报错。
    """
    ok: bool
    action: str
    data: Any = None              # 实际类型由子类指定，Any 是为了 pydantic 不强校验
    error: str = ""
    elapsed_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class Action(ABC, Generic[TIn, TOut]):
    """Action 基类。

    用法：
        class FooAction(Action[FooIn, FooOut]):
            name = "foo"
            input_schema = FooIn
            output_schema = FooOut

            def _run(self, inp: FooIn) -> FooOut:
                return FooOut(...)
    """

    name: str = "Action"
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    # 是否在 Role 流水线中算"必须成功"。False 时失败被吞，返回 None data。
    critical: bool = True

    def __init__(self, **kwargs: Any) -> None:
        # 子类可在 __init__ 里塞依赖（LLM client、vector store 等）
        self._deps: dict[str, Any] = dict(kwargs)

    # ---------- 子类实现 ----------
    @abstractmethod
    def _run(self, inp: TIn) -> TOut:
        """子类核心逻辑。失败请直接 raise。"""
        raise NotImplementedError

    # ---------- 公共入口 ----------
    def run(self, inp: TIn) -> ActionResult[TOut]:
        t0 = time.perf_counter()
        # 每个 Action 作为一个 span；未启用 Langfuse 时是 no-op，无额外开销
        with trace_span(f"action.{self.name}", input=_brief(inp)) as span:
            try:
                if self.input_schema is not None and not isinstance(inp, self.input_schema):
                    # 允许接收 dict，自动包装
                    if isinstance(inp, dict):
                        inp = self.input_schema(**inp)  # type: ignore[arg-type]
                    else:
                        raise TypeError(
                            f"{self.name}: expected {self.input_schema.__name__}, "
                            f"got {type(inp).__name__}"
                        )
                out = self._run(inp)
                elapsed = (time.perf_counter() - t0) * 1000.0
                logger.info(f"[action] {self.name} ok in {elapsed:.1f}ms")
                span.update(
                    output=_brief(out),
                    metadata={"ok": True, "elapsed_ms": round(elapsed, 1)},
                )
                return ActionResult(
                    ok=True,
                    action=self.name,
                    data=out,
                    elapsed_ms=elapsed,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000.0
                logger.exception(f"[action] {self.name} failed: {e}")
                span.update(
                    metadata={"ok": False, "elapsed_ms": round(elapsed, 1),
                              "error": f"{type(e).__name__}: {e}"},
                )
                if self.critical:
                    # critical 失败也返回 result（不抛出），由 Role 决定是否中断
                    return ActionResult(
                        ok=False,
                        action=self.name,
                        error=f"{type(e).__name__}: {e}",
                        elapsed_ms=elapsed,
                    )
                return ActionResult(
                    ok=False,
                    action=self.name,
                    error=str(e),
                    elapsed_ms=elapsed,
                )

    # ---------- 辅助 ----------
    def dep(self, key: str, default: Any = None) -> Any:
        """从 __init__ 注入的依赖里取一个。"""
        return self._deps.get(key, default)
