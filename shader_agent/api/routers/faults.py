"""故障注入路由（**仅在 test profile 下注册**）。

为什么要做成 HTTP 接口而不是只留 pytest 内的 `with fault(...)`：
接口自动化测试有两种运行形态——进程内（ASGI 直连，CI 默认）和跨进程（打已经
部署好的实例，预发环境用）。后者没法直接改被测进程的内存，只能通过这个受控
后门下发故障配置。`create_app()` 在非 test profile 下根本不会挂载它，生产不存在
这个路径。
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from shader_agent.api.errors import envelope
from shader_agent.api.schemas import ApiResponse, FaultRequest
from shader_agent.service.errors import ErrorCode, ServiceError
from shader_agent.testing import faults

router = APIRouter(prefix="/api/v1/_test", tags=["_test"])

_ENUMS = {
    "llm_mode": faults.LLM_MODES,
    "compiler_mode": faults.COMPILER_MODES,
    "renderer_mode": faults.RENDERER_MODES,
    "retriever_mode": faults.RETRIEVER_MODES,
}


@router.get("/faults", response_model=ApiResponse, summary="查看当前故障配置")
def get_faults(request: Request) -> dict:
    return envelope(request, faults.current().to_dict())


@router.post("/faults", response_model=ApiResponse, summary="下发故障配置")
def set_faults(req: FaultRequest, request: Request) -> dict:
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    for field, allowed in _ENUMS.items():
        val = payload.get(field)
        if val is not None and val not in allowed:
            raise ServiceError(
                ErrorCode.INVALID_PARAM,
                f"{field} 只能取 {list(allowed)}，收到 {val!r}")
    cfg = faults.set_faults(**payload)
    return envelope(request, cfg.to_dict())


@router.delete("/faults", response_model=ApiResponse, summary="复位故障配置")
def reset_faults(request: Request) -> dict:
    return envelope(request, faults.reset().to_dict())
