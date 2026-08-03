"""请求级中间件：请求 ID、耗时、访问日志。

请求 ID 的作用不止于排障：接口测试失败时，报告里附上 `X-Request-Id`，就能直接
在服务端日志（以及 Langfuse trace）里定位到同一次调用，把"接口报错了"变成
"这一步的哪个环节报错了"。
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request

from shader_agent.utils.logger import logger

REQUEST_ID_HEADER = "X-Request-Id"
ELAPSED_HEADER = "X-Elapsed-Ms"


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _trace_request(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        request.state.t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            cost = (time.perf_counter() - request.state.t0) * 1000.0
            logger.exception(
                f"[api] {request.method} {request.url.path} rid={rid} "
                f"cost={cost:.1f}ms raised"
            )
            raise
        cost = (time.perf_counter() - request.state.t0) * 1000.0
        response.headers[REQUEST_ID_HEADER] = rid
        response.headers[ELAPSED_HEADER] = f"{cost:.2f}"
        logger.info(
            f"[api] {request.method} {request.url.path} "
            f"status={response.status_code} rid={rid} cost={cost:.1f}ms"
        )
        return response
