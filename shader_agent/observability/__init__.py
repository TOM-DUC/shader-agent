"""可观测性子系统：Langfuse tracing 的统一入口。

对外只暴露"业务需要"的最小面。所有接口在 Langfuse 未安装或未配置时都会
优雅降级为 no-op，因此可以在任何业务模块无条件 import 使用。

典型用法：
    from shader_agent.observability import trace_span, score_current_trace

    with trace_span("task.generate", input={"prompt": text}) as span:
        result = do_work()
        span.update(output=result)
"""
from shader_agent.observability.langfuse_client import (
    flush,
    get_langfuse,
    get_openai_class,
    is_enabled,
    langfuse_openai_active,
)
from shader_agent.observability.tracing import (
    bind_current_context,
    get_current_trace_id,
    score_current_trace,
    score_trace_by_id,
    trace_span,
    update_current_trace,
)

__all__ = [
    "flush",
    "get_langfuse",
    "get_openai_class",
    "is_enabled",
    "langfuse_openai_active",
    "bind_current_context",
    "get_current_trace_id",
    "score_current_trace",
    "score_trace_by_id",
    "trace_span",
    "update_current_trace",
]
