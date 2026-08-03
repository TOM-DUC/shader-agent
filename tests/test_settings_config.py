"""配置层守护用例。

这些用例守的不是某个功能，而是几条**容易被下一次改动悄悄破坏的约定**。
每一条都对应一个真实踩过的坑，注释里写清"坏掉之后会怎样"，避免后来者
觉得断言多余而顺手删掉。

跑法：
    pytest tests/test_settings_config.py -v
    pytest -m config
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from shader_agent.config.settings import (
    ConfigError,
    LoggingConfig,
    EmbeddingConfig,
    LLMConfig,
    MissingCredentialsError,
    Settings,
)

pytestmark = pytest.mark.config

# 所有敏感字段都必须是可选的
SECRET_FIELDS = [
    "deepseek_api_key",
    "shadertoy_api_key",
    "langfuse_public_key",
    "langfuse_secret_key",
]

# 子进程用例里需要剔除的环境变量：凭据本身 + 会改变装配路径的开关
_ENV_KEYS_TO_CLEAR = {
    "DEEPSEEK_API_KEY", "SHADERTOY_API_KEY",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "SHADER_AGENT_DEEPSEEK_API_KEY",
}


# ============================================================
# 1) 凭据缺失不得成为导入期错误
# ============================================================

@pytest.mark.parametrize("field_name", SECRET_FIELDS)
def test_secret_fields_are_optional(field_name: str) -> None:
    """回归用例：`deepseek_api_key` 曾经是 `Field(...)` 必填。

    坏掉之后的表现：没有 key 的环境（CI、新同事的机器、test profile）连
    `import shader_agent.config.settings` 都过不去，业务里所有判空降级分支
    永远执行不到——症状是"一堆用例集体 ImportError"，根因却在配置层。

    这里断言的是模型元信息而不是"能否 import"，因为开发机上通常配了 .env，
    "能 import"这个断言在有 key 的机器上恒真，起不到守护作用。
    """
    assert Settings.model_fields[field_name].is_required() is False, (
        f"{field_name} 不允许是必填字段"
    )


def test_module_imports_in_a_clean_environment() -> None:
    """真·冒烟：另起一个不含任何凭据的子进程，完整走一遍 import 流程。

    上面那条断言的是模型元信息，这条断言的是"整个模块真的能在无凭据环境下加载"
    ——包括 .env 缺失、python-dotenv 缺失、config.yaml 缺失这些分支。

    环境用 os.environ 的副本再剔除凭据（而不是从零构造），是为了保住 PATH、
    SYSTEMROOT、TMP 这些平台相关变量，否则用例在 Windows / conda 上会以
    "环境残缺"的形式假红。
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS_TO_CLEAR}
    env["PYTHONPATH"] = str(repo_root)
    # 开发机上 .env 存在真实 key：load_dotenv 默认会在 import 时把它注入
    # os.environ，导致这里"无凭据"的模拟失效。显式关闭 .env 加载。
    env["SHADER_AGENT_LOAD_DOTENV"] = "0"

    code = textwrap.dedent(
        """
        from shader_agent.config.settings import Settings
        s = Settings(_env_file=None)
        assert s.has_llm_credentials is False, "环境未清干净，用例失去意义"
        s.require_llm_credentials  # 属性存在即可，不调用
        print("CLEAN_IMPORT_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, f"无凭据环境下导入失败：\n{proc.stderr}"
    assert "CLEAN_IMPORT_OK" in proc.stdout


# ============================================================
# 2) 凭据判断只有一个口径
# ============================================================

def test_missing_key_reports_false_and_requires_raise() -> None:
    s = Settings(_env_file=None, deepseek_api_key="")
    assert s.has_llm_credentials is False
    with pytest.raises(MissingCredentialsError) as ei:
        s.require_llm_credentials()
    # 报错必须给出可执行的下一步，而不是只说"缺 key"
    msg = str(ei.value)
    assert "DEEPSEEK_API_KEY" in msg
    assert "SHADER_AGENT_PROFILE=test" in msg


def test_present_key_reports_true_and_requires_pass() -> None:
    s = Settings(_env_file=None, deepseek_api_key="sk-real-key")
    assert s.has_llm_credentials is True
    s.require_llm_credentials()  # 不应抛异常


@pytest.mark.parametrize(
    "raw",
    ['  sk-abc123  ', '"sk-abc123"', "'sk-abc123'", ' "sk-abc123" '],
)
def test_secret_is_stripped_and_unquoted(raw: str) -> None:
    """`.env` 里带引号或空格是最常见的低级坑。

    坏掉之后的表现：`if settings.deepseek_api_key:` 判定"已配置"，
    真实请求却 401——排查成本远高于问题本身的技术含量。
    """
    assert Settings(_env_file=None, deepseek_api_key=raw).deepseek_api_key == "sk-abc123"


@pytest.mark.parametrize("blank", ["", "   ", '""', "' '"])
def test_blank_like_values_count_as_missing(blank: str) -> None:
    assert Settings(_env_file=None, deepseek_api_key=blank).has_llm_credentials is False


def test_credential_status_never_leaks_the_full_secret() -> None:
    """脱敏输出会进日志、/readyz 响应和 CI 产物，必须确保不含明文。"""
    secret = "sk-0123456789abcdef"
    status = Settings(_env_file=None, deepseek_api_key=secret).credential_status()
    assert secret not in str(status)
    assert status["deepseek_api_key"].startswith("sk-0")
    assert status["shadertoy_api_key"] == "<unset>"


# ============================================================
# 3) 子配置不得读环境变量
# ============================================================

@pytest.mark.parametrize(
    "model_cls, env_name, field_name, default",
    [
        (LoggingConfig, "LEVEL", "level", "INFO"),
        (EmbeddingConfig, "DEVICE", "device", "auto"),
        (LLMConfig, "TEMPERATURE", "temperature", 0.3),
        (EmbeddingConfig, "BATCH_SIZE", "batch_size", 8),
    ],
)
def test_subconfig_ignores_generic_env_vars(
    monkeypatch, model_cls, env_name, field_name, default
) -> None:
    """子配置曾经继承 BaseSettings，于是会去读 `LEVEL` / `DEVICE` /
    `TEMPERATURE` / `BATCH_SIZE` 这些极常见的通用环境变量。

    坏掉之后的表现：CI runner 或 conda 环境里随便一个同名变量就能静默覆盖
    config.yaml，且没有任何日志——线上线下行为不一致，且完全无从定位。
    """
    monkeypatch.setenv(env_name, "999" if isinstance(default, (int, float)) else "cuda")
    assert getattr(model_cls(), field_name) == default


def test_subconfig_ignores_unknown_yaml_keys() -> None:
    """config.yaml 新增字段时，旧版本代码不应直接崩掉。"""
    assert LLMConfig(**{"chat_model": "x", "some_future_key": 1}).chat_model == "x"


# ============================================================
# 4) 配置写错要给出可读报错
# ============================================================

def test_broken_yaml_raises_config_error(tmp_path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("llm:\n  chat_model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        Settings.load(config_path=bad)
    assert "config.yaml" in str(ei.value)


def test_bad_section_value_reports_which_section(tmp_path) -> None:
    """报错必须点名是哪一段。

    直接抛 pydantic 原始异常时只能看到类名（RetrievalConfig），
    在一份两百行的 config.yaml 里还得反查段名。
    """
    bad = tmp_path / "config.yaml"
    bad.write_text("retrieval:\n  recall_k: 这不是数字\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        Settings.load(config_path=bad)
    assert "retrieval" in str(ei.value)


def test_section_must_be_a_mapping(tmp_path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("paths:\n  - data\n  - logs\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        Settings.load(config_path=bad)
    assert "paths" in str(ei.value)


def test_missing_config_yaml_falls_back_to_defaults(tmp_path) -> None:
    """clone 下来不做任何配置也应该能跑 test profile 的测试。"""
    s = Settings.load(config_path=tmp_path / "not-exists.yaml")
    assert s.llm.chat_model == LLMConfig().chat_model
    assert s.retrieval.recall_k == 20
