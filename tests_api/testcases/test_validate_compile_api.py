"""静态规则校验与编译校验。

这两个接口是整套质量体系里"最便宜的一层"：不调大模型、不占 GPU，毫秒级返回。
因此绝大多数规则类断言都压在这里跑，慢链路只保留少量端到端验证。
"""
from __future__ import annotations

import pytest

from tests_api.utils.assertions import assert_error, assert_expect, assert_ok
from tests_api.utils.yaml_loader import parametrize, resolve_payload

pytestmark = [pytest.mark.api]


# ============================================================
# /validate —— 数据驱动
# ============================================================

@parametrize("validate_cases.yaml")
def test_validate_cases(shader_api, shaders, case):
    payload = resolve_payload(case, shaders)
    assert_expect(shader_api.validate(payload=payload), case["expect"])


def test_validate_reports_rule_id_and_level(shader_api, unsupported_shader):
    """规则命中必须给出可定位的 rule_id 与 level，而不是一句"代码有问题"。"""
    data = assert_ok(shader_api.validate(unsupported_shader))
    assert data["passed"] is False
    ids = {v["rule_id"] for v in data["violations"]}
    assert "GLSL020" in ids, f"未命中多通道规则：{data['violations']}"
    for v in data["violations"]:
        assert v["level"] in ("error", "warn")
        assert v["message"]


def test_validate_is_fast(shader_api, valid_shader):
    """静态校验不该触碰任何外部依赖，超过 1s 说明实现里混进了慢调用。"""
    assert_ok(shader_api.validate(valid_shader), max_ms=1000)


# ============================================================
# /compile
# ============================================================

@pytest.mark.smoke
def test_compile_valid_shader(shader_api, valid_shader):
    data = assert_ok(shader_api.compile(valid_shader))
    assert data["ok"] is True
    assert data["errors"] == ""
    assert data["backend"]


def test_compile_broken_shader_is_business_failure_not_http_error(
        shader_api, broken_shader):
    """编不过是**业务结果**，不是接口错误：HTTP 200 + code 0 + ok=false。

    这条用例守的是一条设计约定——只有调用方用错接口或依赖挂了才返回非 0
    错误码。若哪天有人把编译失败改成 500，这里会立刻红。
    """
    data = assert_ok(shader_api.compile(broken_shader))
    assert data["ok"] is False
    assert data["errors"], "编译失败必须带回编译器原文，否则修复循环无从下手"


def test_compile_error_message_carries_user_line_number(shader_api, broken_shader):
    """错误信息应把包装层行号翻译回用户代码行号，否则定位不到。"""
    data = assert_ok(shader_api.compile(broken_shader))
    assert "user:" in data["errors"] or "error" in data["errors"].lower()


def test_compile_empty_code_rejected(shader_api):
    assert_error(shader_api.compile(""), 40001, http=422)
