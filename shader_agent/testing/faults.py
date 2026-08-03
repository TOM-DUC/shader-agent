"""故障注入开关。

只在 `profile=test` 的装配下生效：Stub LLM / Mock 编译器 / Fake 检索器在每次被
调用前都会读一次这里的配置，从而在**不改业务代码**的前提下模拟外部依赖异常。

两种驱动方式：
1. 进程内（pytest 首选）：`with fault(llm_mode="timeout"): ...`
2. 跨进程（对已部署实例做接口测试）：调用 `POST /api/v1/_test/faults`
   —— 该路由仅在 test profile 下注册，生产装配不会暴露。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterator

# ---- 可选值（同时用于接口参数校验与测试用例 YAML 的取值域）----
LLM_MODES = (
    "ok",             # 正常
    "timeout",        # 抛超时
    "rate_limit",     # 抛 429
    "auth_error",     # 鉴权失败
    "malformed_json", # 返回非法 JSON（考验解析容错）
    "empty",          # 返回空串
    "invalid_code",   # 返回编译不过的 GLSL（考验修复循环）
    "unsupported",    # 返回带 iChannel0 的代码（考验规则校验）
    "slow",           # 正常但慢（考验超时与耗时统计）
)
COMPILER_MODES = ("ok", "always_fail", "fail_first_n")
RENDERER_MODES = ("ok", "unavailable", "blank", "slow")
RETRIEVER_MODES = ("ok", "empty", "error", "slow")


@dataclass
class FaultConfig:
    """一次测试期望的依赖行为。默认全 ok，即完全正常的链路。"""

    llm_mode: str = "ok"
    #: 前 N 次调用按 llm_mode 失败，之后恢复正常 —— 用于验证重试/自愈
    llm_fail_times: int = 0
    llm_latency_ms: int = 0

    compiler_mode: str = "ok"
    #: 0 表示一直失败（配合 always_fail）；>0 表示只失败前 N 次（配合 fail_first_n）
    compiler_fail_times: int = 0

    renderer_mode: str = "ok"
    renderer_latency_ms: int = 0

    retriever_mode: str = "ok"
    retriever_latency_ms: int = 0

    #: 调用计数（只读，供测试断言"重试了几次"）
    counters: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str) -> int:
        n = self.counters.get(key, 0) + 1
        self.counters[key] = n
        return n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_LOCK = threading.RLock()
_CURRENT = FaultConfig()


def current() -> FaultConfig:
    return _CURRENT


def set_faults(**kwargs: Any) -> FaultConfig:
    """更新故障配置；未指定的字段保持不变。返回更新后的配置。"""
    global _CURRENT
    valid = {f for f in FaultConfig.__dataclass_fields__ if f != "counters"}
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(f"unknown fault fields: {sorted(unknown)}")
    with _LOCK:
        _CURRENT = replace(_CURRENT, **kwargs)
        _CURRENT.counters = {}
    return _CURRENT


def reset() -> FaultConfig:
    """恢复全默认。每个用例结束后必须调用，避免故障配置串味。"""
    global _CURRENT
    with _LOCK:
        _CURRENT = FaultConfig()
    return _CURRENT


@contextmanager
def fault(**kwargs: Any) -> Iterator[FaultConfig]:
    """`with fault(llm_mode="timeout"): ...`，退出时自动复位。"""
    prev = current()
    try:
        yield set_faults(**kwargs)
    finally:
        global _CURRENT
        with _LOCK:
            _CURRENT = prev


def should_fail(kind: str, mode_field: str, times_field: str) -> bool:
    """判断本次调用是否应当失败。

    `*_fail_times <= 0` 表示"一直失败"；否则只有前 N 次失败，用来验证重试后成功。
    """
    cfg = current()
    mode = getattr(cfg, mode_field)
    if mode == "ok":
        return False
    n = cfg.bump(kind)
    times = getattr(cfg, times_field, 0)
    if times <= 0:
        return True
    return n <= times
