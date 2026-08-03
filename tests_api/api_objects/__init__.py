"""API Object 层：把 HTTP 细节收敛在这里，用例只面对业务语义。"""
from tests_api.api_objects.base_api import ApiResult, BaseAPI  # noqa: F401
from tests_api.api_objects.shader_api import (  # noqa: F401
    FaultAPI,
    RetrievalAPI,
    ShaderAPI,
    SystemAPI,
)

__all__ = ["ApiResult", "BaseAPI", "ShaderAPI", "RetrievalAPI", "SystemAPI", "FaultAPI"]
