"""可观测性层的离线单测。

核心断言：**在 langfuse 未安装 / 未配置时，所有接口必须是无害的 no-op**，
业务代码可以无条件调用它们。这是本次接入的第一原则。
"""
from __future__ import annotations

import pytest


def test_import_never_raises():
    """即使没装 langfuse，import 也必须成功。"""
    import shader_agent.observability as obs
    assert hasattr(obs, "trace_span")
    assert hasattr(obs, "score_current_trace")
    assert hasattr(obs, "is_enabled")


def test_is_enabled_returns_bool():
    from shader_agent.observability import is_enabled
    assert isinstance(is_enabled(), bool)


def test_trace_span_is_context_manager_and_noop_safe():
    """未启用时 trace_span 仍应能进出，且 span.update() 不抛错。"""
    from shader_agent.observability import trace_span

    with trace_span("test.span", input={"a": 1}) as span:
        span.update(output={"b": 2}, metadata={"ok": True})
        span.update_trace(name="x")
    # 走到这里即通过


def test_trace_span_does_not_swallow_business_exception():
    """span 不能吞掉业务异常——否则错误会被静默。"""
    from shader_agent.observability import trace_span

    with pytest.raises(ValueError):
        with trace_span("test.span"):
            raise ValueError("boom")


def test_score_and_update_are_noop_safe():
    from shader_agent.observability import (
        get_current_trace_id,
        score_current_trace,
        score_trace_by_id,
        update_current_trace,
    )

    update_current_trace(name="t", tags=["a"], metadata={"k": "v"})
    score_current_trace("m", 0.5, comment="c")
    score_trace_by_id("", "m", 0.5)
    tid = get_current_trace_id()
    assert tid is None or isinstance(tid, str)


def test_bind_current_context_returns_callable():
    """未启用时应原样返回函数；启用时返回等价包装。"""
    from shader_agent.observability import bind_current_context

    def f(x, y=2):
        return x + y

    g = bind_current_context(f)
    assert callable(g)
    assert g(1) == 3
    assert g(1, y=5) == 6


def test_get_openai_class_returns_pair():
    """OpenAI 类选择器：无论装没装 langfuse，都要返回 (class, bool)。"""
    pytest.importorskip("openai")
    from shader_agent.observability import get_openai_class

    cls, is_lf = get_openai_class()
    assert cls is not None
    assert isinstance(is_lf, bool)


def test_flush_is_safe():
    from shader_agent.observability import flush
    flush()  # 不应抛错


def test_trace_span_propagates_exception_when_enabled(monkeypatch):
    """回归测试：启用路径下，业务异常必须原样传播。

    早期实现用 try/except 包住了 `yield`，导致业务异常被捕获后二次 yield，
    Python 抛出 RuntimeError("generator didn't stop after throw()") 并**掩盖**
    了原始异常。这里用假客户端锁死该行为。
    """
    from contextlib import contextmanager

    import shader_agent.observability.tracing as tracing

    events: list[str] = []

    class _FakeSpan:
        def update(self, **kw):
            events.append("update")

    class _FakeClient:
        @contextmanager
        def start_as_current_observation(self, **kw):
            events.append("start")
            try:
                yield _FakeSpan()
            finally:
                events.append("end")

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "get_langfuse", lambda: _FakeClient())

    with pytest.raises(ValueError, match="boom"):
        with tracing.trace_span("x"):
            raise ValueError("boom")

    # span 必须被正常关闭，而不是泄漏
    assert events == ["start", "end"]


def test_trace_span_nests_when_enabled(monkeypatch):
    """启用路径下，子 span 应在父 span 之内开合。"""
    from contextlib import contextmanager

    import shader_agent.observability.tracing as tracing

    events: list[tuple[str, str]] = []

    class _FakeSpan:
        def __init__(self, name):
            self.name = name

        def update(self, **kw):
            events.append(("update", self.name))

    class _FakeClient:
        @contextmanager
        def start_as_current_observation(self, name="", **kw):
            events.append(("start", name))
            try:
                yield _FakeSpan(name)
            finally:
                events.append(("end", name))

    monkeypatch.setattr(tracing, "is_enabled", lambda: True)
    monkeypatch.setattr(tracing, "get_langfuse", lambda: _FakeClient())

    with tracing.trace_span("root") as r:
        with tracing.trace_span("child") as c:
            c.update(output=1)
        r.update(output=2)

    assert events == [
        ("start", "root"), ("start", "child"), ("update", "child"),
        ("end", "child"), ("update", "root"), ("end", "root"),
    ]


def test_action_run_still_works_with_tracing():
    """Action 被 span 包裹后，行为契约不能变。"""
    from pydantic import BaseModel

    from shader_agent.agents.actions.base import Action

    class _In(BaseModel):
        x: int

    class _Out(BaseModel):
        y: int

    class _OkAction(Action):
        name = "test_ok"
        input_schema = _In
        output_schema = _Out

        def _run(self, inp):
            return _Out(y=inp.x * 2)

    class _FailAction(Action):
        name = "test_fail"
        input_schema = _In
        output_schema = _Out

        def _run(self, inp):
            raise RuntimeError("expected")

    r = _OkAction().run(_In(x=3))
    assert r.ok is True
    assert r.data.y == 6
    assert r.elapsed_ms >= 0

    r2 = _FailAction().run(_In(x=1))
    assert r2.ok is False
    assert "expected" in r2.error


def test_action_accepts_dict_input_under_tracing():
    from pydantic import BaseModel

    from shader_agent.agents.actions.base import Action

    class _In(BaseModel):
        x: int

    class _Out(BaseModel):
        y: int

    class _A(Action):
        name = "t"
        input_schema = _In
        output_schema = _Out

        def _run(self, inp):
            return _Out(y=inp.x)

    r = _A().run({"x": 7})
    assert r.ok and r.data.y == 7


def test_settings_have_observability_and_evaluation_sections():
    from shader_agent.config.settings import settings

    assert settings.observability.enabled in ("auto", "on", "off")
    assert isinstance(settings.observability.service_name, str)
    assert 0.0 <= settings.evaluation.threshold_generation <= 1.0
    assert 0.0 <= settings.evaluation.threshold_retrieval <= 1.0
