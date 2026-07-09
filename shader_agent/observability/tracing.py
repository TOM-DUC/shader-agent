"""追踪辅助层：对 Langfuse v3（OpenTelemetry 架构）的薄封装。

只暴露业务需要的最小面，且**在 Langfuse 未启用时全部降级为 no-op**：

  - trace_span(name, ...)          : 以上下文管理器开一个 span/generation，
                                     自动按 OTel 上下文嵌套（最外层即 trace 根）。
  - update_current_trace(**kw)     : 给当前 trace 挂 session_id / user_id / tags / metadata。
  - score_current_trace(name,value): 给当前 trace 打分（deepeval 分数回流用）。
  - get_current_trace_id()         : 取当前 trace id（跨系统关联用）。
  - bind_current_context(fn)       : 捕获当前 OTel 上下文，返回在该上下文中运行的
                                     包装函数——用于 ThreadPoolExecutor 并行子任务
                                     的 span 正确挂到父 trace 下。
  - flush()                        : 透传到客户端 flush。

设计原则：任何一步失败都不得影响主业务链路，全部 try/except 兜底。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from shader_agent.observability.langfuse_client import (
    flush as _client_flush,
    get_langfuse,
    is_enabled,
)


# ============================================================
# Span 代理：无论底层是否可用，都给调用方一个统一的 .update() 接口
# ============================================================

class _SpanProxy:
    """包一层，屏蔽 langfuse span 与 no-op 的差异。"""

    __slots__ = ("_span",)

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def update(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update(**kwargs)
        except Exception:
            pass

    def update_trace(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            # v3 span 支持 update_trace；不支持时退回全局 update_current_trace
            fn = getattr(self._span, "update_trace", None)
            if fn is not None:
                fn(**kwargs)
            else:
                update_current_trace(**kwargs)
        except Exception:
            pass

    @property
    def raw(self) -> Any:
        return self._span


@contextmanager
def trace_span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: Optional[dict] = None,
) -> Iterator[_SpanProxy]:
    """开一个 span（或 generation）。未启用 Langfuse 时是纯 no-op。

    用法：
        with trace_span("action.draft_code", input={"spec": ...}) as span:
            result = do_work()
            span.update(output=result, metadata={"ok": True})
    """
    if not is_enabled():
        yield _SpanProxy(None)
        return

    client = get_langfuse()
    if client is None:
        yield _SpanProxy(None)
        return

    kwargs: dict = {"as_type": as_type, "name": name}
    if input is not None:
        kwargs["input"] = input
    if metadata:
        kwargs["metadata"] = metadata

    try:
        cm = client.start_as_current_observation(**kwargs)
    except TypeError:
        # 兼容签名差异：去掉 input/metadata 再试
        try:
            cm = client.start_as_current_observation(as_type=as_type, name=name)
        except Exception:
            yield _SpanProxy(None)
            return
    except Exception:
        yield _SpanProxy(None)
        return

    # 注意：这里**不能**用 try/except 包住 `yield`。
    # 生成器在 yield 处收到业务异常时，若被捕获后再 yield 一次，Python 会抛
    # RuntimeError("generator didn't stop after throw()") 并掩盖原始异常。
    # 业务异常必须原样向上传播；langfuse 的上下文管理器自身会记录该异常并结束 span。
    with cm as span:
        yield _SpanProxy(span)


def update_current_trace(**kwargs: Any) -> None:
    """给当前 trace 设置 session_id / user_id / tags / metadata / name 等。"""
    if not is_enabled():
        return
    client = get_langfuse()
    try:
        client.update_current_trace(**kwargs)
    except Exception:
        pass


def score_current_trace(name: str, value: float, comment: str = "") -> None:
    """给当前 trace 打一个数值分（deepeval 指标回流的主要入口）。"""
    if not is_enabled():
        return
    client = get_langfuse()
    # 不同小版本方法名略有差异，逐个尝试
    for meth in ("score_current_trace", "score_current_span"):
        fn = getattr(client, meth, None)
        if fn is None:
            continue
        try:
            if comment:
                fn(name=name, value=value, comment=comment)
            else:
                fn(name=name, value=value)
            return
        except Exception:
            continue
    # 最后兜底：create_score + 当前 trace id
    try:
        tid = get_current_trace_id()
        if tid is not None:
            client.create_score(trace_id=tid, name=name, value=value,
                                comment=comment or None)
    except Exception:
        pass


def score_trace_by_id(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """按 trace_id 给已结束的 trace 打分（离线批量评估回流用）。"""
    if not is_enabled() or not trace_id:
        return
    client = get_langfuse()
    try:
        client.create_score(trace_id=trace_id, name=name, value=value,
                            comment=comment or None)
    except Exception:
        pass


def get_current_trace_id() -> Optional[str]:
    if not is_enabled():
        return None
    client = get_langfuse()
    for meth in ("get_current_trace_id",):
        fn = getattr(client, meth, None)
        if fn is not None:
            try:
                return fn()
            except Exception:
                return None
    return None


def flush() -> None:
    _client_flush()


# ============================================================
# 线程上下文传播（用于 ThreadPoolExecutor 并行的四段式分析）
# ============================================================

def bind_current_context(fn: Callable) -> Callable:
    """捕获"当前"OTel 上下文，返回一个在该上下文中执行 fn 的包装函数。

    Analyzer 用 ThreadPoolExecutor 并行跑 walkthrough/summary 等子任务。OTel 的
    上下文是 contextvar，不会自动跨线程传播；不处理的话，子任务里的 span/generation
    会各自开成孤立 trace。这里在**提交前**捕获上下文，在**worker 线程内**attach，
    使并行子 span 正确挂到父 trace 下。

    Langfuse/opentelemetry 未安装时原样返回 fn。
    """
    if not is_enabled():
        return fn
    try:
        from opentelemetry import context as otel_context
    except Exception:
        return fn

    ctx = otel_context.get_current()

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        token = otel_context.attach(ctx)
        try:
            return fn(*args, **kwargs)
        finally:
            try:
                otel_context.detach(token)
            except Exception:
                pass

    return _wrapper
