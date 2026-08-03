"""Shader 能力接口：校验 / 编译 / 渲染 / 分析 / 生成 / 改写。

路由函数一律用同步 `def`：底层是阻塞的 LLM 调用与 GL 渲染，交给 FastAPI 的
线程池比放进事件循环更安全，也让压测时的并发行为可预期。

每个能力都是**独立可调用**的，这一点对测试很关键——
生成链路慢（数秒到数十秒），但 `/validate` 与 `/compile` 是毫秒级的，
于是绝大多数规则类断言可以在快接口上跑，慢接口只留少量端到端用例。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from shader_agent.api.deps import get_shader_service, require_api_key
from shader_agent.api.errors import envelope
from shader_agent.api.schemas import (
    AnalyzeRequest,
    ApiResponse,
    CompileRequest,
    GenerateRequest,
    RemixRequest,
    RenderRequest,
    ValidateRequest,
)
from shader_agent.service.shader_service import ShaderService

router = APIRouter(
    prefix="/api/v1/shader",
    tags=["shader"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/validate", response_model=ApiResponse, summary="GLSL 静态规则校验")
def validate(req: ValidateRequest, request: Request,
             service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.validate(
        req.code, require_dynamic=req.require_dynamic))


@router.post("/compile", response_model=ApiResponse, summary="GLSL 编译验证")
def compile_shader(req: CompileRequest, request: Request,
                   service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.compile(req.code))


@router.post("/render", response_model=ApiResponse, summary="离屏渲染单帧（base64 PNG）")
def render(req: RenderRequest, request: Request,
           service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.render(
        req.code, width=req.width, height=req.height, time_s=req.time))


@router.post("/analyze", response_model=ApiResponse, summary="分析 shader 并检索参考样本")
def analyze(req: AnalyzeRequest, request: Request,
            service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.analyze(
        req.code, top_k=req.top_k, with_render=req.with_render,
        with_reference_code=req.with_reference_code))


@router.post("/generate", response_model=ApiResponse, summary="按中文需求生成 shader")
def generate(req: GenerateRequest, request: Request,
             service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.generate(
        req.description, palette=req.palette, complexity=req.complexity,
        dynamic=req.dynamic, effect_type=req.effect_type,
        constraints=req.constraints, max_fix_loops=req.max_fix_loops,
        top_k=req.top_k, enable_self_critique=req.enable_self_critique,
        with_render=req.with_render))


@router.post("/remix", response_model=ApiResponse, summary="基于原代码局部改写")
def remix(req: RemixRequest, request: Request,
          service: ShaderService = Depends(get_shader_service)) -> dict:
    return envelope(request, service.remix(
        req.code, req.instruction, analyze_first=req.analyze_first,
        max_fix_loops=req.max_fix_loops, with_render=req.with_render))
