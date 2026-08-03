"""配置体检：`python -m shader_agent.config.doctor` / `make doctor`。

存在的理由：本项目的依赖是分层可选的（LLM / 真 GL / 向量库都可以缺席，
各自降级到不同路径），于是"我这台机器现在到底跑在哪条路径上"是一个每天都要
回答的问题。没有这个命令时，回答方式是起服务、看一屏日志、再猜。

刻意保持的三个性质：
  · 不发起任何网络请求，不加载模型权重——秒级返回，可以随手跑；
  · 凭据一律脱敏，输出可以直接贴进 issue 或 CI 日志；
  · 退出码有意义：0 = 当前 profile 可用，1 = 当前 profile 起不来。
    于是它可以直接当 CI 的前置门禁，而不只是给人看。
"""
from __future__ import annotations

import os
import sys

from shader_agent.config.settings import (
    DOTENV_INJECTED,
    LOAD_DOTENV,
    ConfigError,
    MissingCredentialsError,
    reload_settings,
)

OK = "  ok "
WARN = " warn"
FAIL = " fail"


def _line(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name:<22} {detail}")


def _check_optional_import(module: str, purpose: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        _line(WARN, module, f"未安装（{purpose}）")
        return False
    _line(OK, module, purpose)
    return True


def main() -> int:
    print("=" * 68)
    print("Shader Agent 配置体检")
    print("=" * 68)

    # ---------- 配置本身 ----------
    try:
        settings = reload_settings()
    except ConfigError as e:
        _line(FAIL, "config.yaml", str(e).splitlines()[0])
        print("\n配置文件本身有问题，先修它。")
        return 1

    profile = os.environ.get("SHADER_AGENT_PROFILE", "auto").strip().lower() or "auto"
    _line(OK, "profile", profile)
    _line(OK, "project_root", str(settings.project_root))
    env_file = settings.project_root / ".env"
    # 只报"文件在不在"是不够的：SHADER_AGENT_LOAD_DOTENV=0 时文件在也不生效，
    # 而"我明明配了 .env，为什么说没 key"正是这个命令要一次性回答掉的问题。
    if not LOAD_DOTENV:
        _line(WARN, ".env", "已被 SHADER_AGENT_LOAD_DOTENV=0 跳过（凭据只认环境变量）")
    elif env_file.exists():
        _line(OK, ".env", f"{env_file}（{'已加载' if DOTENV_INJECTED else '存在但未注入新变量'}）")
    else:
        _line(WARN, ".env", "不存在（凭据只能从环境变量来）")

    # ---------- 凭据 ----------
    print("-" * 68)
    for name, masked in settings.credential_status().items():
        _line(OK if masked != "<unset>" else WARN, name, masked)

    # ---------- 可选依赖 ----------
    print("-" * 68)
    _check_optional_import("fastapi", "HTTP 接口层")
    _check_optional_import("pytest", "自动化测试")
    _check_optional_import("numpy", "test profile 的确定性渲染")
    _check_optional_import("PIL", "图像层校验")
    _check_optional_import("dotenv", "从 .env 读凭据（缺失时仍可用环境变量）")
    _check_optional_import("moderngl", "真实 GL 渲染（缺失自动降级）")

    # ---------- 当前 profile 能否起来 ----------
    print("-" * 68)
    if profile == "test":
        _line(OK, "LLM", "stub（确定性桩，无需 key）")
    elif settings.has_llm_credentials:
        _line(OK, "LLM", f"已配置，chat_model={settings.llm.chat_model}")
    elif profile == "real":
        try:
            settings.require_llm_credentials("real profile")
        except MissingCredentialsError as e:
            _line(FAIL, "LLM", "real profile 缺 DEEPSEEK_API_KEY")
            print("\n" + str(e))
            return 1
    else:
        _line(WARN, "LLM", "未配置 key，auto profile 将降级为无 LLM 路径")

    print("-" * 68)
    print("结论：当前 profile 可以启动。")
    print("  · 起服务      make api        （test profile，无需 key/GPU）")
    print("  · 跑配置守护  make config")
    print("  · 跑冒烟      make smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
