"""异常 → 统一响应信封 的映射。

一个原则：**接口不向调用方泄漏栈**。所有异常在这里收口成
`{code, message, request_id, elapsed_ms, data}`，栈只进日志。
这样测试永远面对同一种响应结构，负向用例才能写得和正向一样稳。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from shader_agent.api.deps import elapsed_ms, get_request_id
from shader_agent.service.errors import ErrorCode, ServiceError, http_status_of
from shader_agent.utils.logger import logger


def envelope(request: Request, data: Any = None, *,
             code: int = 0, message: str = "ok") -> dict[str, Any]:
    return {
        "code": int(code),
        "message": message,
        "request_id": get_request_id(request),
        "elapsed_ms": elapsed_ms(request),
        "data": data,
    }


def json_error(request: Request, code: ErrorCode | int, message: str,
               *, detail: Any = None, http_status: int | None = None) -> JSONResponse:
    body = envelope(request, detail, code=int(code), message=message)
    return JSONResponse(
        status_code=http_status or http_status_of(int(code)),
        content=body,
        headers={"X-Request-Id": body["request_id"]},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        logger.info(f"[api] ServiceError {int(exc.code)} {exc.message}")
        return json_error(request, exc.code, exc.message, detail=exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # 把 pydantic 的错误压平成 {字段路径: 原因}，便于用例精确断言到字段
        fields = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
            fields.append({"field": loc or "body",
                           "type": err.get("type", ""),
                           "msg": err.get("msg", "")})
        return json_error(
            request, ErrorCode.INVALID_PARAM,
            f"参数校验失败：{fields[0]['field']} {fields[0]['msg']}" if fields else "参数校验失败",
            detail={"errors": fields}, http_status=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_PARAM
        if exc.status_code >= 500:
            code = ErrorCode.INTERNAL
        return json_error(request, code, str(exc.detail),
                          http_status=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("[api] 未捕获异常")
        return json_error(
            request, ErrorCode.INTERNAL,
            f"internal error: {type(exc).__name__}",
            http_status=500,
        )
