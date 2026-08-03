"""多层断言工具。

分层的理由：接口返回 200 只能证明"服务没崩"，离"功能是对的"还差得远。
本项目把断言分成五层，从便宜到昂贵依次是：

    L1 协议层  assert_ok / assert_error        —— 信封结构、错误码、请求 ID
    L2 契约层  assert_schema                   —— JSON Schema，字段名/类型/必填
    L3 规则层  glsl_checker                    —— Shader 代码是否合规
    L4 编译层  compile 接口 / compile_ok       —— GLSL 能不能真的编过
    L5 图像层  image_checker                   —— 渲染出来的画面对不对

一条用例通常只做到它该做的那一层：`/validate` 的用例停在 L3，
端到端生成用例才一路做到 L5。
"""
from __future__ import annotations

from typing import Any, Optional

import pytest

from tests_api.api_objects.base_api import ApiResult
from tests_api.utils.yaml_loader import DATA_DIR

_UNSET = object()


# ---------------- L1：协议层 ----------------

def assert_ok(result: ApiResult, *, max_ms: Optional[float] = None) -> Any:
    """断言业务成功，返回 data 便于链式使用。"""
    assert result.status_code == 200, f"期望 HTTP 200，实际 {result}"
    assert result.code == 0, f"期望 code=0，实际 {result}"
    assert result.request_id, f"响应缺少 request_id：{result}"
    assert result.data is not None, f"成功响应 data 不应为 null：{result}"
    if max_ms is not None:
        assert result.elapsed_ms <= max_ms, (
            f"耗时 {result.elapsed_ms:.0f}ms 超过阈值 {max_ms}ms：{result}")
    return result.data


def assert_error(result: ApiResult, code: int,
                 *, http: Optional[int] = None,
                 message_contains: str = "") -> ApiResult:
    """断言失败，且失败的**原因**符合预期（而不是"随便报个错就算过"）。"""
    assert result.code == code, f"期望 code={code}，实际 {result}"
    if http is not None:
        assert result.status_code == http, f"期望 HTTP {http}，实际 {result}"
    else:
        assert result.status_code >= 400, f"失败响应应为 4xx/5xx：{result}"
    if message_contains:
        assert message_contains in result.message, (
            f"message 未包含 {message_contains!r}：{result}")
    return result


def assert_envelope(result: ApiResult) -> None:
    """信封结构本身的断言：任何接口、任何结果都必须满足。"""
    for key in ("code", "message", "request_id", "elapsed_ms", "data"):
        assert key in result.raw, f"响应缺少信封字段 `{key}`：{result.raw}"
    assert isinstance(result.raw["code"], int)
    assert isinstance(result.raw["message"], str)
    assert isinstance(result.raw["elapsed_ms"], (int, float))
    assert result.raw["elapsed_ms"] >= 0


# ---------------- L2：契约层 ----------------

def assert_schema(payload: Any, schema_name: str) -> None:
    """用 JSON Schema 校验响应体结构。

    比逐字段 assert 更能挡住"字段被悄悄改名/类型从 int 变 str"这类破坏性变更，
    也是前后端契约的可执行版本。
    """
    try:
        import json

        import jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("未安装 jsonschema，跳过契约校验")
        return
    path = DATA_DIR / "schemas" / schema_name
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as e:  # pragma: no cover - 失败路径
        pytest.fail(f"Schema[{schema_name}] 校验失败：{e.message}\n"
                    f"路径：{'.'.join(str(x) for x in e.absolute_path)}")


# ---------------- YAML 驱动的通用期望 ----------------

def get_path(obj: Any, path: str, default: Any = _UNSET) -> Any:
    """按 `a.b.0.c` 取值，支持 dict 与 list 混合。"""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                if default is _UNSET:
                    raise AssertionError(f"路径 {path} 在列表中不存在")
                return default
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        if default is _UNSET:
            raise AssertionError(f"路径 {path} 不存在，实际对象键={_keys(cur)}")
        return default
    return cur


def _keys(obj: Any) -> Any:
    return list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__


def assert_expect(result: ApiResult, expect: dict[str, Any]) -> None:
    """执行 YAML 用例里的 `expect` 段。"""
    assert_envelope(result)
    if "http" in expect:
        assert result.status_code == expect["http"], (
            f"期望 HTTP {expect['http']}，实际 {result}")
    if "code" in expect:
        assert result.code == expect["code"], (
            f"期望 code={expect['code']}，实际 {result}")
    if "message_contains" in expect:
        assert expect["message_contains"] in result.message, (
            f"message 未包含 {expect['message_contains']!r}：{result}")
    for path, want in (expect.get("data") or {}).items():
        got = get_path(result.data, path)
        assert got == want, f"data.{path} 期望 {want!r}，实际 {got!r}（{result}）"
    for path, want in (expect.get("contains") or {}).items():
        got = get_path(result.data, path)
        assert isinstance(got, (str, list)), f"data.{path} 不是可包含类型：{type(got)}"
        assert want in got, f"data.{path} 未包含 {want!r}（{result}）"
    for path, want in (expect.get("not_contains") or {}).items():
        got = get_path(result.data, path)
        assert want not in got, f"data.{path} 不应包含 {want!r}（{result}）"
    for path, bound in (expect.get("min") or {}).items():
        got = float(get_path(result.data, path))
        assert got >= bound, f"data.{path}={got} 应 >= {bound}（{result}）"
    for path, bound in (expect.get("max") or {}).items():
        got = float(get_path(result.data, path))
        assert got <= bound, f"data.{path}={got} 应 <= {bound}（{result}）"
    if "max_ms" in expect:
        assert result.elapsed_ms <= expect["max_ms"], (
            f"耗时 {result.elapsed_ms:.0f}ms 超过 {expect['max_ms']}ms（{result}）")
    if "schema" in expect:
        assert_schema(result.raw, expect["schema"])
