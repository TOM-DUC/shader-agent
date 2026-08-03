"""各业务模块的 API Object。

方法签名按"业务语义"设计（`generate(description, palette=...)`），而不是照抄
请求体字段，这样用例读起来接近自然语言；同时保留 `payload=` 逃生口，
用于构造缺字段、多字段、类型错误这类异常入参。
"""
from __future__ import annotations

from typing import Any, Optional

from tests_api.api_objects.base_api import ApiResult, BaseAPI


class SystemAPI(BaseAPI):
    """健康检查与元信息。"""

    def healthz(self) -> ApiResult:
        return self.get("/healthz")

    def readyz(self) -> ApiResult:
        return self.get("/readyz")

    def meta(self) -> ApiResult:
        return self.get("/api/v1/meta")

    def openapi(self) -> ApiResult:
        return self.get("/openapi.json")


class ShaderAPI(BaseAPI):
    """Shader 能力：校验 / 编译 / 渲染 / 分析 / 生成 / 改写。"""

    VALIDATE = "/api/v1/shader/validate"
    COMPILE = "/api/v1/shader/compile"
    RENDER = "/api/v1/shader/render"
    ANALYZE = "/api/v1/shader/analyze"
    GENERATE = "/api/v1/shader/generate"
    REMIX = "/api/v1/shader/remix"

    def validate(self, code: str = "", *, require_dynamic: Optional[bool] = None,
                 payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none(
            {"code": code, "require_dynamic": require_dynamic})
        return self.post(self.VALIDATE, body)

    def compile(self, code: str = "", *, payload: Any = None) -> ApiResult:
        return self.post(self.COMPILE,
                         payload if payload is not None else {"code": code})

    def render(self, code: str = "", *, width: Optional[int] = None,
               height: Optional[int] = None, time: Optional[float] = None,
               payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none(
            {"code": code, "width": width, "height": height, "time": time})
        return self.post(self.RENDER, body)

    def analyze(self, code: str = "", *, top_k: Optional[int] = None,
                with_render: Optional[bool] = None,
                with_reference_code: Optional[bool] = None,
                payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none({
            "code": code, "top_k": top_k, "with_render": with_render,
            "with_reference_code": with_reference_code})
        return self.post(self.ANALYZE, body)

    def generate(self, description: str = "", *, palette: Optional[str] = None,
                 complexity: Optional[str] = None, dynamic: Optional[bool] = None,
                 effect_type: Optional[str] = None,
                 constraints: Optional[list] = None,
                 max_fix_loops: Optional[int] = None, top_k: Optional[int] = None,
                 enable_self_critique: Optional[bool] = None,
                 with_render: Optional[bool] = None,
                 payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none({
            "description": description, "palette": palette,
            "complexity": complexity, "dynamic": dynamic,
            "effect_type": effect_type, "constraints": constraints,
            "max_fix_loops": max_fix_loops, "top_k": top_k,
            "enable_self_critique": enable_self_critique,
            "with_render": with_render})
        return self.post(self.GENERATE, body)

    def remix(self, code: str = "", instruction: str = "", *,
              analyze_first: Optional[bool] = None,
              max_fix_loops: Optional[int] = None,
              with_render: Optional[bool] = None,
              payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none({
            "code": code, "instruction": instruction,
            "analyze_first": analyze_first, "max_fix_loops": max_fix_loops,
            "with_render": with_render})
        return self.post(self.REMIX, body)


class RetrievalAPI(BaseAPI):
    SEARCH = "/api/v1/retrieval/search"

    def search(self, query: str = "", *, top_k: Optional[int] = None,
               tags: Optional[list] = None, payload: Any = None) -> ApiResult:
        body = payload if payload is not None else _drop_none(
            {"query": query, "top_k": top_k, "tags": tags})
        return self.post(self.SEARCH, body)


class FaultAPI(BaseAPI):
    """故障注入（仅 test profile 可用）。"""

    PATH = "/api/v1/_test/faults"

    def set(self, **kwargs: Any) -> ApiResult:
        return self.post(self.PATH, _drop_none(kwargs))

    def current(self) -> ApiResult:
        return self.get(self.PATH)

    def reset(self) -> ApiResult:
        return self.delete(self.PATH)


def _drop_none(d: dict) -> dict:
    """只发送显式指定的字段，让"用默认值"和"传 null"成为两种可区分的用例。"""
    return {k: v for k, v in d.items() if v is not None}
