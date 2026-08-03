"""存活 / 就绪 / 元信息。

`/healthz` 只回答"进程还在吗"，不碰任何依赖，用于容器存活探针；
`/readyz` 会逐个依赖体检并回 `ok | degraded | down`，用于流量准入与冒烟测试的
第一道门——测试套件启动时先打它，依赖没起来就直接失败，而不是让后面几十条
用例一起红成一片、还得逐个看日志才知道是环境问题。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from shader_agent.api.errors import envelope
from shader_agent.api.deps import get_shader_service
from shader_agent.api.schemas import ApiResponse
from shader_agent.config.settings import settings
from shader_agent.service.shader_service import ShaderService

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


@router.get("/healthz", response_model=ApiResponse, summary="存活探针")
def healthz(request: Request,
            service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.health())


@router.get("/readyz", response_model=ApiResponse, summary="就绪探针（逐依赖体检）")
def readyz(request: Request,
           service: ShaderService = Depends(get_shader_service)) -> dict:
    data = service.readiness()
    message = "ok" if data["status"] == "ok" else data["status"]
    return envelope(request, data, message=message)


@router.get("/api/v1/meta", response_model=ApiResponse, summary="服务元信息")
def meta(request: Request,
         service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, {
        "api_version": API_VERSION,
        "service": settings.observability.service_name,
        "environment": settings.observability.environment,
        "profile": service.options.resolved_profile(),
        "chat_model": settings.llm.chat_model,
        "coder_model": settings.llm.coder_model,
    })
