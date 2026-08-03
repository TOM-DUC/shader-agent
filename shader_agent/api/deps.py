"""接口层依赖注入。

所有路由拿 Service 都必须走 `Depends(get_shader_service)`——这是整套接口自动化
测试的支点：测试可以用 `app.dependency_overrides[get_shader_service] = ...`
把整个业务门面换成注入了故障的实例，而无需改一行业务代码，也无需真的把上游
服务弄挂。
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from fastapi import Depends, Header, Request

from shader_agent.service.assembly import AssemblyOptions
from shader_agent.service.errors import ErrorCode, ServiceError
from shader_agent.service.shader_service import ShaderService, get_service


def get_shader_service() -> ShaderService:
    """默认门面。测试通过 dependency_overrides 替换。"""
    return get_service()


def build_service(profile: str = "", **kwargs) -> ShaderService:
    """按 profile 现场构造一个门面（供测试与脚本使用）。"""
    return ShaderService(AssemblyOptions(profile=profile, **kwargs))


def get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", "")
    if not rid:
        rid = uuid.uuid4().hex[:16]
        request.state.request_id = rid
    return rid


def elapsed_ms(request: Request) -> float:
    t0 = getattr(request.state, "t0", None)
    if t0 is None:
        return 0.0
    return round((time.perf_counter() - t0) * 1000.0, 2)


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """可选鉴权：设置了 SHADER_AGENT_API_KEY 才启用。

    默认关闭，方便本地起服务；CI 里会专门起一个开启鉴权的实例跑鉴权用例。
    """
    expected = os.environ.get("SHADER_AGENT_API_KEY", "")
    if not expected:
        return
    if x_api_key != expected:
        raise ServiceError(ErrorCode.UNAUTHORIZED, "缺少或错误的 X-API-Key")


ServiceDep = Depends(get_shader_service)
AuthDep = Depends(require_api_key)
