"""检索接口：把混合检索单独暴露出来。

单独开这个口子是为了让"检索质量"可以被独立度量：分析/生成链路里检索只是一个
中间环节，效果好坏会被大模型的输出掩盖；直接打检索接口，就能对召回顺序、
融合分构成、标签命中做确定性断言，也便于后续做检索侧的回归基线。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from shader_agent.api.deps import get_shader_service, require_api_key
from shader_agent.api.errors import envelope
from shader_agent.api.schemas import ApiResponse, SearchRequest
from shader_agent.service.shader_service import ShaderService

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["retrieval"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/search", response_model=ApiResponse, summary="混合检索相似 shader")
def search(req: SearchRequest, request: Request,
           service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.retrieve(
        req.query, top_k=req.top_k, tags=req.tags))
