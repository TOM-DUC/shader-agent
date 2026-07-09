"""Langfuse 客户端装配（懒加载 + 优雅降级）。

设计目标（与全项目"可控、可解释、可降级"一致）：
  1. **零侵入**：未安装 langfuse 或未配置密钥时，全部接口降级为 no-op，
     现有链路行为完全不变。
  2. **单例**：进程内只初始化一次 Langfuse 客户端。
  3. **OpenAI drop-in**：对外暴露 `get_openai_class()`，若 langfuse 可用则返回
     `langfuse.openai.OpenAI`（自动把每次 chat.completions 记为 generation，
     携带 token/延迟/成本），否则返回原生 `openai.OpenAI`。

启用条件（三态，由 config.observability.enabled 控制）：
  - "on"   : 只要 langfuse 已安装就启用（缺密钥时 SDK 自身不外发，仅本地 no-op）。
  - "off"  : 强制关闭。
  - "auto" : 检测到 LANGFUSE_PUBLIC_KEY 环境变量即启用（默认）。

敏感字段（LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST）放 .env，
不进版本库；非敏感开关放 config.yaml 的 observability 段。
"""
from __future__ import annotations

import atexit
import os
from typing import Any, Optional, Tuple

# 说明：这里刻意不在模块顶层 import shader_agent.utils.logger，
# 因为 logger 依赖 settings，而 settings 会 import 本模块的兄弟模块，
# 避免潜在的循环导入。改为函数内延迟取 logger。


def _get_logger():
    try:
        from shader_agent.utils.logger import logger
        return logger
    except Exception:  # pragma: no cover - 极端早期导入兜底
        import logging
        return logging.getLogger("shader_agent.observability")


# ---------------- 启用判定 ----------------

def _resolve_enabled() -> bool:
    """综合 config 开关 + 环境变量，判断是否启用 Langfuse。"""
    mode = "auto"
    try:
        from shader_agent.config.settings import settings
        mode = (getattr(settings.observability, "enabled", "auto") or "auto").lower()
    except Exception:
        mode = os.environ.get("SHADER_AGENT_OBSERVABILITY", "auto").lower()

    if mode in ("off", "false", "0", "no"):
        return False
    if mode in ("on", "true", "1", "yes"):
        return _langfuse_installed()
    # auto：装了 langfuse 且提供了 public key 才算启用
    return _langfuse_installed() and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def _langfuse_installed() -> bool:
    try:
        import langfuse  # noqa: F401
        return True
    except Exception:
        return False


# ---------------- 单例状态 ----------------

_CLIENT: Any = None
_INITIALIZED: bool = False
_ENABLED: bool = False


def _init_once() -> None:
    global _CLIENT, _INITIALIZED, _ENABLED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    _ENABLED = _resolve_enabled()
    if not _ENABLED:
        return
    logger = _get_logger()
    try:
        from langfuse import Langfuse, get_client

        # 环境变量优先；若 config 里给了 host/environment 也补进去
        kwargs: dict = {}
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        if host:
            kwargs["host"] = host
        try:
            from shader_agent.config.settings import settings
            env = getattr(settings.observability, "environment", "") or ""
            if env:
                # v3 支持通过 environment 参数或 LANGFUSE_TRACING_ENVIRONMENT
                os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", env)
        except Exception:
            pass

        # 显式构造一次以固化配置，再取全局单例
        try:
            Langfuse(**kwargs)
        except TypeError:
            # 老/新版本参数差异时退化为无参构造（读环境变量）
            Langfuse()
        _CLIENT = get_client()
        logger.info("[observability] Langfuse 已启用（tracing on）")
        atexit.register(_flush_atexit)
    except Exception as e:  # 任意异常都不致命
        _ENABLED = False
        _CLIENT = None
        logger.warning(f"[observability] Langfuse 初始化失败，降级为 no-op：{e}")


def _flush_atexit() -> None:
    try:
        if _CLIENT is not None:
            _CLIENT.flush()
    except Exception:
        pass


# ---------------- 对外 API ----------------

def is_enabled() -> bool:
    """Langfuse 是否已启用（安装 + 配置 + 初始化成功）。"""
    _init_once()
    return _ENABLED and _CLIENT is not None


def get_langfuse() -> Optional[Any]:
    """返回 Langfuse 客户端单例；未启用时返回 None。"""
    _init_once()
    return _CLIENT


def flush() -> None:
    """强制上报缓冲区。短生命周期脚本（verify_*/run_eval）结束前应调用。"""
    if is_enabled():
        try:
            _CLIENT.flush()
        except Exception:
            pass


# ---------------- OpenAI drop-in 选择 ----------------

_OPENAI_CLASS: Any = None
_OPENAI_IS_LANGFUSE: bool = False
_OPENAI_RESOLVED: bool = False


def get_openai_class() -> Tuple[Any, bool]:
    """返回 (OpenAI 类, 是否为 langfuse 包装版)。

    - langfuse 已安装：返回 `langfuse.openai.OpenAI`，自动记录每次 LLM 调用为
      generation（含 model / token / 延迟 / 成本）。即便未配置密钥也可安全使用
      （SDK 不外发，仅无害地接受 name/metadata 等额外 kwargs）。
    - 否则：返回原生 `openai.OpenAI`。

    之所以"只要装了 langfuse 就用包装版"，是为了让 deepseek_client 能无条件地
    透传 name/metadata 这类 langfuse 专用 kwargs；原生 OpenAI 不接受它们。
    """
    global _OPENAI_CLASS, _OPENAI_IS_LANGFUSE, _OPENAI_RESOLVED
    if _OPENAI_RESOLVED:
        return _OPENAI_CLASS, _OPENAI_IS_LANGFUSE
    _OPENAI_RESOLVED = True

    # observability 明确 off 时，直接用原生 OpenAI，避免任何 langfuse 副作用
    force_off = False
    try:
        from shader_agent.config.settings import settings
        force_off = (getattr(settings.observability, "enabled", "auto") or "").lower() == "off"
    except Exception:
        force_off = os.environ.get("SHADER_AGENT_OBSERVABILITY", "").lower() == "off"

    if not force_off:
        try:
            from langfuse.openai import OpenAI as _LFOpenAI
            _OPENAI_CLASS = _LFOpenAI
            _OPENAI_IS_LANGFUSE = True
            return _OPENAI_CLASS, _OPENAI_IS_LANGFUSE
        except Exception:
            pass

    try:
        from openai import OpenAI as _PlainOpenAI
        _OPENAI_CLASS = _PlainOpenAI
    except Exception:
        # openai 是项目硬依赖；这里不抛异常，交由调用方（DeepSeekClient）给出清晰报错。
        # 这样 langfuse_openai_active() 这种热路径函数才能保证永不抛出。
        _OPENAI_CLASS = None
    _OPENAI_IS_LANGFUSE = False
    return _OPENAI_CLASS, _OPENAI_IS_LANGFUSE


def langfuse_openai_active() -> bool:
    """当前 OpenAI 客户端是否为 langfuse 包装版（决定能否透传 name/metadata）。

    该函数位于每次 LLM 调用的热路径上，必须保证永不抛异常。
    """
    try:
        _, is_lf = get_openai_class()
        return bool(is_lf)
    except Exception:
        return False
