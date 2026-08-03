"""FastAPI 应用装配入口。

启动：
    uvicorn shader_agent.api.main:app --host 0.0.0.0 --port 8000
测试内联启动（不占端口，CI 默认走这条）：
    from shader_agent.api.main import create_app
    app = create_app(profile="test")
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shader_agent.api.deps import get_shader_service
from shader_agent.api.errors import register_exception_handlers
from shader_agent.api.middleware import register_middleware
from shader_agent.api.routers import health, retrieval, shader
from shader_agent.service.assembly import AssemblyOptions
from shader_agent.service.shader_service import ShaderService
from shader_agent.utils.logger import logger

DESCRIPTION = """\
Shader Agent 的 HTTP 能力层。

* 每个响应都是统一信封 `{code, message, request_id, elapsed_ms, data}`；
* `code=0` 才是成功，HTTP 状态码只做粗分类；
* 响应头 `X-Request-Id` 可用于串联服务端日志与 Langfuse trace。
"""


def create_app(profile: str = "", **assembly_kwargs) -> FastAPI:
    """构造应用。

    profile 留空时读环境变量 SHADER_AGENT_PROFILE（默认 auto）。
    test profile 会额外挂载故障注入路由。
    """
    opts = AssemblyOptions(profile=profile, **assembly_kwargs)
    resolved = opts.resolved_profile()

    app = FastAPI(
        title="Shader Agent API",
        version="1.0.0",
        description=DESCRIPTION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    allow = os.environ.get("SHADER_AGENT_CORS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[x.strip() for x in allow if x.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_middleware(app)
    register_exception_handlers(app)

    # 该实例固定使用一份装配好的门面；测试可用 dependency_overrides 整体替换
    service = ShaderService(opts)
    app.dependency_overrides[get_shader_service] = lambda: service
    app.state.profile = resolved
    app.state.service = service

    app.include_router(health.router)
    app.include_router(shader.router)
    app.include_router(retrieval.router)

    if resolved == "test":
        from shader_agent.api.routers import faults as faults_router
        app.include_router(faults_router.router)
        logger.warning("[api] test profile：已挂载 /api/v1/_test/faults 故障注入路由")

    logger.info(f"[api] app created (profile={resolved})")
    return app


app = create_app()
