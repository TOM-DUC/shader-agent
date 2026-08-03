"""业务门面层：UI / HTTP API / 自动化测试共用的唯一业务入口。"""
from shader_agent.service.assembly import (  # noqa: F401
    AssemblyOptions,
    clear_assembly_cache,
    get_assembly,
)
from shader_agent.service.errors import (  # noqa: F401
    ErrorCode,
    ServiceError,
    classify_retrieval_error,
    classify_upstream_error,
    http_status_of,
)
from shader_agent.service.shader_service import (  # noqa: F401
    ShaderService,
    get_service,
    reset_service,
)

__all__ = [
    "AssemblyOptions",
    "get_assembly",
    "clear_assembly_cache",
    "ErrorCode",
    "ServiceError",
    "classify_retrieval_error",
    "classify_upstream_error",
    "http_status_of",
    "ShaderService",
    "get_service",
    "reset_service",
]
