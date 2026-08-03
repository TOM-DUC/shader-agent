"""YAML 用例数据加载。

数据与代码分离的实际收益：新增一条边界用例只是加 6 行 YAML，不需要改 Python，
非框架作者也能补用例；同时 `case_id` 会直接变成 pytest 的用例名，报告里一眼
看出失败的是哪条业务场景，而不是 `test_generate[params3]`。

YAML 结构：
    cases:
      - id: generate_normal_blue      # 必填，用作 pytest id
        desc: 蓝色调 + 动态
        marks: [smoke]                # 可选，转成 pytest.mark
        payload: {...}                # 请求体
        expect:                       # 期望
          http: 200
          code: 0
          data: {compile_ok: true}    # 逐字段相等断言（支持点号路径）
          contains: {code: "mainImage"}   # 子串包含
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_yaml(name: str) -> dict[str, Any]:
    path = DATA_DIR / name if not name.endswith(".yaml") else DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cases(name: str, group: str = "cases") -> list[dict[str, Any]]:
    data = load_yaml(name)
    cases = data.get(group) or []
    for i, c in enumerate(cases):
        if "id" not in c:
            raise ValueError(f"{name}[{group}][{i}] 缺少 id 字段")
    return cases


def parametrize(name: str, group: str = "cases", argname: str = "case"):
    """把 YAML 用例转成 `@pytest.mark.parametrize`，并带上 marks 与 id。"""
    cases = load_cases(name, group)
    params = []
    for c in cases:
        marks = [getattr(pytest.mark, m) for m in (c.get("marks") or [])]
        params.append(pytest.param(c, marks=marks, id=c["id"]))
    return pytest.mark.parametrize(argname, params)


def load_shaders() -> dict[str, str]:
    """公共 GLSL 素材（正常 / 编不过 / 不支持特性 / 超大 …）。"""
    return {k: v.strip() for k, v in (load_yaml("shaders.yaml") or {}).items()}


def resolve_payload(case: dict[str, Any], shaders: dict[str, str]) -> dict[str, Any]:
    """把用例里的 `payload_ref: {code: valid_plasma}` 展开成真实代码。

    这样 YAML 里既不用重复粘贴几十行 GLSL，也能一眼看出这条用例用的是哪份素材。
    另外支持 `repeat: {field: n}` 生成超长入参，用于边界值用例。
    """
    payload = dict(case.get("payload") or {})
    for field, ref in (case.get("payload_ref") or {}).items():
        if ref not in shaders:
            raise KeyError(f"用例 {case.get('id')} 引用了不存在的素材 {ref!r}")
        payload[field] = shaders[ref]
    for field, n in (case.get("repeat") or {}).items():
        payload[field] = payload.get(field, "x") * int(n)
    return payload
