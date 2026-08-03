"""API Object 基类。

设计要点：
1. **用例里不出现 URL、不出现 requests/httpx**。接口一旦改路径或改信封，只动
   api_objects 一层，几十条用例不用碰。
2. 每次调用都返回结构化的 `ApiResult`，把"HTTP 状态码"和"业务错误码"分开暴露，
   避免用例里到处 `resp.json()["code"]` 这种脆弱写法。
3. 同一套 API Object 既能打进程内 ASGI（CI 默认，快且不占端口），也能打已部署
   实例（`--base-url`），用例代码完全不变。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tests_api.utils.report import attach_text


@dataclass
class ApiResult:
    """一次接口调用的结果。"""
    status_code: int
    code: int
    message: str
    data: Any
    request_id: str
    elapsed_ms: float                 # 客户端观测耗时
    server_elapsed_ms: float = 0.0    # 服务端自报耗时
    raw: dict[str, Any] = field(default_factory=dict)
    url: str = ""
    payload: Any = None

    @property
    def ok(self) -> bool:
        """业务成功：HTTP 2xx 且 code == 0。"""
        return 200 <= self.status_code < 300 and self.code == 0

    def __str__(self) -> str:  # 断言失败时 pytest 会打出来，便于定位
        return (f"<{self.url} http={self.status_code} code={self.code} "
                f"msg={self.message!r} rid={self.request_id} "
                f"cost={self.elapsed_ms:.0f}ms>")


class BaseAPI:
    """所有 API Object 的父类。"""

    def __init__(self, client, api_key: str = "") -> None:
        self._client = client
        self._api_key = api_key

    # ---------- 底层 ----------
    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        if extra:
            h.update(extra)
        return h

    def request(self, method: str, path: str, *, json: Any = None,
                headers: Optional[dict] = None) -> ApiResult:
        t0 = time.perf_counter()
        resp = self._client.request(method, path, json=json,
                                    headers=self._headers(headers))
        cost = (time.perf_counter() - t0) * 1000.0
        try:
            body = resp.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {"code": -1, "message": "响应不是 JSON 对象", "data": body}

        result = ApiResult(
            status_code=resp.status_code,
            code=int(body.get("code", -1)),
            message=str(body.get("message", "")),
            data=body.get("data"),
            request_id=str(body.get("request_id", "")
                           or resp.headers.get("X-Request-Id", "")),
            elapsed_ms=cost,
            server_elapsed_ms=float(body.get("elapsed_ms", 0.0) or 0.0),
            raw=body,
            url=f"{method.upper()} {path}",
            payload=json,
        )
        # 失败时把请求/响应带进报告，省掉"复现一遍才知道传了什么"的环节
        if not result.ok:
            attach_text(f"{result.url} 请求体", _pretty(json))
            attach_text(f"{result.url} 响应体", _pretty(body))
        return result

    def post(self, path: str, payload: Any = None, **kw) -> ApiResult:
        return self.request("POST", path, json=payload, **kw)

    def get(self, path: str, **kw) -> ApiResult:
        return self.request("GET", path, **kw)

    def delete(self, path: str, **kw) -> ApiResult:
        return self.request("DELETE", path, **kw)


def _pretty(obj: Any) -> str:
    import json as _json
    try:
        text = _json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = str(obj)
    # base64 图片会把报告撑爆，截断
    return text if len(text) <= 4000 else text[:4000] + "\n...(truncated)"
