"""健康检查与协议层通用约定。

这组用例是整个套件的"地基"：如果信封结构、request_id、错误映射不稳定，
上面几十条业务用例的断言就都建在流沙上。
"""
from __future__ import annotations

import pytest

from tests_api.utils.assertions import assert_envelope, assert_ok, assert_schema

pytestmark = [pytest.mark.api]


@pytest.mark.smoke
def test_healthz_returns_ok(system_api):
    data = assert_ok(system_api.healthz(), max_ms=2000)
    assert data["status"] == "ok"


@pytest.mark.smoke
def test_readyz_reports_every_component(system_api):
    """就绪探针必须逐依赖上报，而不是笼统给一个 ok。"""
    data = assert_ok(system_api.readyz())
    assert data["status"] in ("ok", "degraded")
    for name in ("llm", "retrieval", "render", "profile"):
        assert name in data["components"], f"缺少组件 {name} 的状态"
        assert data["components"][name]["status"] in ("ok", "degraded", "down")


def test_meta_exposes_versions(system_api):
    data = assert_ok(system_api.meta())
    assert data["api_version"]
    assert data["profile"] in ("test", "auto", "real")


@pytest.mark.contract
def test_every_response_uses_the_same_envelope(system_api, shader_api, valid_shader):
    """同一套信封覆盖成功、业务失败、参数失败三种路径。"""
    for result in (
        system_api.healthz(),
        shader_api.validate(valid_shader),
        shader_api.validate(payload={}),          # 参数校验失败
    ):
        assert_envelope(result)
        assert_schema(result.raw, "envelope.json")


def test_request_id_is_echoed_back(shader_api, valid_shader):
    """客户端传入的 X-Request-Id 应被透传，便于把测试报告与服务端日志对齐。"""
    rid = "qa-fixed-request-id"
    result = shader_api.request(
        "POST", shader_api.VALIDATE,
        json={"code": valid_shader}, headers={"X-Request-Id": rid})
    assert result.request_id == rid, f"request_id 未透传：{result}"


def test_request_id_is_unique_per_call(shader_api, valid_shader):
    ids = {shader_api.validate(valid_shader).request_id for _ in range(3)}
    assert len(ids) == 3, "未传入 request_id 时服务端应自行生成且不重复"


def test_unknown_route_returns_envelope_not_html(shader_api):
    """404 也必须是信封结构——否则客户端要写两套解析逻辑。"""
    result = shader_api.get("/api/v1/shader/not-exists")
    assert result.status_code == 404
    assert result.code == 40401
    assert_envelope(result)


def test_openapi_document_is_available(system_api):
    result = system_api.openapi()
    assert result.status_code == 200
    doc = result.raw
    assert "paths" in doc
    for path in ("/api/v1/shader/generate", "/api/v1/shader/render",
                 "/api/v1/retrieval/search"):
        assert path in doc["paths"], f"OpenAPI 缺少 {path}"
