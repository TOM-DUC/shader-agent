"""统一配置入口。

业务代码一律通过 `from shader_agent.config.settings import settings`
拿到配置，不要直接读 os.environ 或 yaml。

设计约定（配套用例在 `tests/test_settings_config.py`）：

1. **凭据缺失不是导入期错误。**
   所有 API Key 字段一律可选（默认空串），`import settings` 在任何环境下都不抛异常。
   缺 key 的后果由**调用方**决定：`auto` profile 降级走 stub，`real` profile 显式
   调 `require_llm_credentials()` 快速失败。原先 `deepseek_api_key` 写成 `Field(...)`
   必填，导致没有 key 时连 import 都过不去——业务侧那些判空降级分支根本执行不到，
   CI、单元测试、test profile 全部起不来。

2. **凭据是否可用只有一个判断口径**：`settings.has_llm_credentials`。
   不要在业务里各写各的 `os.environ.get("DEEPSEEK_API_KEY") or settings.xxx`，
   否则 " " 这种带空格的脏值在两处判断结果不一致，表现为"检测到 key 但调用 401"。

3. **只有 Settings 读环境变量，子配置不读。**
   子配置（LLMConfig / LoggingConfig / ...）改为纯 `BaseModel`。它们原先继承
   `BaseSettings`，意味着 `LoggingConfig.level` 会去读环境变量 `LEVEL`、
   `EmbeddingConfig.device` 会去读 `DEVICE`、`LLMConfig.temperature` 会去读
   `TEMPERATURE`——这些都是极常见的通用变量名，CI 或 conda 环境里随便一个就能
   静默覆盖掉 config.yaml，且没有任何日志。

4. **非敏感字段统一加 `SHADER_AGENT_` 前缀**，避免占用 `LLM`、`PATHS` 这类裸名字；
   敏感字段用显式 alias 保持 `DEEPSEEK_API_KEY` 等业界惯例名不变（向后兼容）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件位于 shader_agent/config/settings.py，向上三级
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 显式加载 .env（pydantic-settings 也会读 env_file，这里只是让"手写脚本直接
# os.environ 取值"的场景也能拿到）。python-dotenv 是可选依赖：没装不影响启动，
# 因为 env_file 那条路径已经覆盖了主要用法。
#
# 环境开关 SHADER_AGENT_LOAD_DOTENV=0 可跳过 .env 加载——供"无凭据环境"
# 模拟（CI / 配置层守护用例）使用。开发机上 .env 存在真实 key，不设开关时
# import 会把 key 注入 os.environ，导致 `Settings(_env_file=None)` 仍能看到
# 凭据，"clean environment" 用例永远无法复现。
#
# 注意：读 .env 有**两条**路径——这里的 load_dotenv（注入 os.environ）与
# pydantic-settings 的 env_file。开关必须同时管住两条，否则语义只兑现一半：
# `SHADER_AGENT_LOAD_DOTENV=0 make doctor` 仍会从 env_file 读到真 key，
# 而调用方以为自己已经关掉了 .env。见 Settings.load()。
LOAD_DOTENV: bool = os.environ.get("SHADER_AGENT_LOAD_DOTENV", "1") != "0"

#: .env 是否真的被 load_dotenv 注入过（供 doctor / 诊断输出解释现象）
DOTENV_INJECTED: bool = False

if LOAD_DOTENV:
    try:  # pragma: no cover - 仅取决于环境是否装了 python-dotenv
        from dotenv import load_dotenv

        DOTENV_INJECTED = bool(load_dotenv(PROJECT_ROOT / ".env"))
    except ImportError:  # pragma: no cover
        pass


def _env_file_arg() -> str | None:
    """`Settings.load()` 传给 pydantic-settings 的 `_env_file`。

    单独抽成函数有两个作用：一是让"开关同时管住两条读 .env 的路径"这件事
    有一个可直接断言的落点；二是 `LOAD_DOTENV` 在这里是**运行时**读模块全局，
    测试可以 monkeypatch 它而不必重新 import 整个模块。
    """
    return str(PROJECT_ROOT / ".env") if LOAD_DOTENV else None


class ConfigError(RuntimeError):
    """配置本身有问题（yaml 语法错、字段类型错）。

    单独立一个类型，是为了把"配置写错了"与"运行期业务异常"区分开：
    前者应当在启动时就以可读信息终止，而不是被上层的 except Exception 吞掉后
    以某个莫名其妙的 AttributeError 形式出现在半小时后的请求里。
    """


class MissingCredentialsError(RuntimeError):
    """需要真实凭据的路径上没有凭据。

    与 ConfigError 区分：配置文件本身没错，只是这台机器没配 key。
    `auto` profile 会捕获它并降级；`real` profile 让它冒泡，启动即失败。
    """


# ============================================================
# 子配置：纯数据模型，只从 config.yaml 来，不读环境变量
# ============================================================

class _Section(BaseModel):
    """所有 config.yaml 子段的基类。

    extra="ignore"：config.yaml 里多出的键不报错，方便新增配置时新旧代码共存。
    """

    model_config = ConfigDict(extra="ignore")


class LLMConfig(_Section):
    """从 config.yaml 的 llm 段读取，非敏感。"""
    chat_model: str = "deepseek-v4-pro"
    coder_model: str = "deepseek-v4-flash"
    reasoner_model: str = "deepseek-v4-pro"
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.95
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: int = 2


class OrchestrationConfig(_Section):
    framework: Literal["metagpt", "custom"] = "custom"


class LoggingConfig(_Section):
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"


class PathsConfig(_Section):
    data_dir: str = "data"
    shadertoy_corpus_dir: str = "data/shadertoy_corpus"
    vector_db_dir: str = "data/vector_db"
    models_dir: str = "data/models"


class CorpusConfig(_Section):
    """Shadertoy 语料库构建配置。"""
    shadertoy_api_base: str = "https://www.shadertoy.com/api/v1"
    max_shaders: int = 300
    min_likes: int = 30
    search_queries: list[str] = Field(default_factory=list)
    per_query_limit: int = 60
    request_interval_seconds: float = 0.4
    enable_llm_tagging: bool = False
    min_code_chars: int = 200
    max_code_chars: int = 12000


class EmbeddingConfig(_Section):
    """嵌入模型配置。"""
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 8
    cache_dir: str | None = "data/models"


class VectorStoreConfig(_Section):
    """向量库配置。"""
    collection_name: str = "shadertoy_shaders"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class RetrievalConfig(_Section):
    """混合检索配置：召回规模、融合权重、阈值与重排开关。"""
    recall_k: int = 20            # 向量与关键词各自的召回上限
    w_vector: float = 0.50        # 向量相关度权重
    w_bm25: float = 0.25          # 关键词（BM25）权重
    w_tag: float = 0.15           # 标签匹配度权重
    w_quality: float = 0.10       # 质量分权重
    min_score: float = 0.15       # 融合分阈值，低于此值不返回参考
    use_rerank: bool = True       # 是否启用交叉编码器重排
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


class ObservabilityConfig(_Section):
    """可观测性（Langfuse）配置。非敏感开关放这里；密钥放 .env。

    enabled 三态：
      - "auto" : 检测到 LANGFUSE_PUBLIC_KEY 即启用（默认）。
      - "on"   : 只要装了 langfuse 就启用（缺密钥时 SDK 本地 no-op，不外发）。
      - "off"  : 强制关闭，走原生 OpenAI，无任何 langfuse 副作用。
    """
    enabled: Literal["auto", "on", "off"] = "auto"
    service_name: str = "shader-agent"      # 服务名（写入 trace metadata）
    environment: str = "dev"                # 环境标（dev/staging/prod）
    trace_llm_io: bool = True               # 是否记录 LLM 输入输出正文
    tags: list[str] = Field(default_factory=list)


class EvaluationConfig(_Section):
    """离线评估（DeepEval）配置。

    judge_model 为空时用 llm.chat_model 作为 LLM-as-a-judge 的评审模型。
    评估复用项目现有的 DeepSeek 客户端，无需额外的评审 API key。
    """
    judge_model: str = ""                   # 评审模型；空=用 llm.chat_model
    judge_temperature: float = 0.0
    push_scores_to_langfuse: bool = True    # 评估分数是否回流到 Langfuse
    threshold_generation: float = 0.6       # 生成质量通过门槛
    threshold_retrieval: float = 0.5        # 检索相关性通过门槛
    threshold_analysis: float = 0.6         # 分析忠实度通过门槛


# config.yaml 的段名 -> (Settings 字段名, 模型类)
_YAML_SECTIONS: dict[str, tuple[str, type[_Section]]] = {
    "llm": ("llm", LLMConfig),
    "orchestration": ("orchestration", OrchestrationConfig),
    "logging": ("logging_cfg", LoggingConfig),
    "paths": ("paths", PathsConfig),
    "corpus": ("corpus", CorpusConfig),
    "embedding": ("embedding", EmbeddingConfig),
    "vector_store": ("vector_store", VectorStoreConfig),
    "retrieval": ("retrieval", RetrievalConfig),
    "observability": ("observability", ObservabilityConfig),
    "evaluation": ("evaluation", EvaluationConfig),
}


def _mask(secret: str, keep: int = 4) -> str:
    """脱敏展示：用于日志、/readyz、诊断信息。

    诊断信息经常要回答"到底读到 key 没有""是不是读成了另一个环境的 key"，
    直接打全文会让密钥漏进日志文件和 CI 产物里。
    """
    if not secret:
        return "<unset>"
    if len(secret) <= keep:
        return "*" * len(secret)
    return f"{secret[:keep]}{'*' * 6}(len={len(secret)})"


class Settings(BaseSettings):
    """全局配置对象。

    敏感字段（API key）来自 .env；其余来自 config.yaml。
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="SHADER_AGENT_",   # 非敏感字段统一前缀，不占用裸名字
        case_sensitive=False,
        populate_by_name=True,        # 允许 Settings(deepseek_api_key=...) 直接构造，测试友好
        extra="ignore",
    )

    # ---------- 敏感字段（来自 .env，保留业界惯例的裸名字）----------
    # 全部可选：缺失只影响对应能力，不影响 import。
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "deepseek_api_key"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "deepseek_base_url"),
    )
    shadertoy_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("SHADERTOY_API_KEY", "shadertoy_api_key"),
    )
    langfuse_public_key: str = Field(
        default="",
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
    )
    langfuse_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "langfuse_secret_key"),
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "langfuse_host"),
    )

    # ---------- 嵌套配置（来自 config.yaml）----------
    llm: LLMConfig = Field(default_factory=LLMConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    logging_cfg: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # 项目根
    project_root: Path = PROJECT_ROOT

    # ---------- 清洗 ----------
    @field_validator(
        "deepseek_api_key", "deepseek_base_url", "shadertoy_api_key",
        "langfuse_public_key", "langfuse_secret_key", "langfuse_host",
        mode="before",
    )
    @classmethod
    def _clean_secret(cls, v: Any) -> Any:
        """去掉 .env 里粘贴时常见的首尾空白与包裹引号。

        `DEEPSEEK_API_KEY="sk-xxx"` 和 `DEEPSEEK_API_KEY= sk-xxx ` 这两种写法在
        很多 shell / dotenv 实现下会把引号或空格一并带进来，结果是
        `if settings.deepseek_api_key:` 判定为"已配置"，实际请求 401。
        这类问题排查成本远高于它的技术含量，在入口处一次清干净。
        """
        if not isinstance(v, str):
            return v
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1].strip()
        return v

    # ---------- 凭据契约：全项目唯一判断口径 ----------
    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def has_shadertoy_credentials(self) -> bool:
        return bool(self.shadertoy_api_key)

    @property
    def has_langfuse_credentials(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def require_llm_credentials(self, purpose: str = "调用大模型") -> None:
        """需要真实大模型的路径显式调用，缺凭据时以可读信息快速失败。"""
        if not self.has_llm_credentials:
            raise MissingCredentialsError(
                f"{purpose}需要 DEEPSEEK_API_KEY，但未配置。\n"
                f"  · 生产/联调：在 {self.project_root / '.env'} 中设置 DEEPSEEK_API_KEY=sk-...\n"
                f"  · 本地开发：SHADER_AGENT_PROFILE=auto（缺 key 自动降级为无 LLM 路径）\n"
                f"  · 跑测试  ：SHADER_AGENT_PROFILE=test（使用确定性桩，无需 key、无需 GPU）"
            )

    def credential_status(self) -> dict[str, str]:
        """脱敏后的凭据概览，供 /readyz、启动日志与 `make doctor` 使用。"""
        return {
            "deepseek_api_key": _mask(self.deepseek_api_key),
            "shadertoy_api_key": _mask(self.shadertoy_api_key),
            "langfuse_public_key": _mask(self.langfuse_public_key),
            "langfuse_secret_key": _mask(self.langfuse_secret_key),
        }

    # ---------- 派生路径属性 ----------
    @property
    def corpus_raw_dir(self) -> Path:
        return self.project_root / self.paths.shadertoy_corpus_dir / "raw"

    @property
    def corpus_clean_dir(self) -> Path:
        return self.project_root / self.paths.shadertoy_corpus_dir / "clean"

    @property
    def vector_db_path(self) -> Path:
        return self.project_root / self.paths.vector_db_dir

    @property
    def models_path(self) -> Path:
        return self.project_root / self.paths.models_dir

    # ---------- 装载 ----------
    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        """合并 .env + config.yaml 生成实例。

        config.yaml 不存在时全部走默认值——这是刻意的：clone 下来不做任何配置
        也应该能 `import settings`、能跑 test profile 的测试。

        `_env_file` 显式跟随 `SHADER_AGENT_LOAD_DOTENV`：开关为 "0" 时连
        pydantic-settings 的 env_file 一并关掉，凭据就只可能来自真实环境变量。
        """
        yaml_path = Path(config_path) if config_path else (PROJECT_ROOT / "config.yaml")
        yaml_data: dict = {}

        if yaml_path.exists():
            try:
                with yaml_path.open("r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ConfigError(f"config.yaml 解析失败（{yaml_path}）：{e}") from e
            if not isinstance(yaml_data, dict):
                raise ConfigError(
                    f"config.yaml 顶层应为映射（{yaml_path}），实际是 {type(yaml_data).__name__}"
                )

        sections: dict[str, Any] = {}
        for yaml_key, (field_name, model_cls) in _YAML_SECTIONS.items():
            raw = yaml_data.get(yaml_key) or {}
            if not isinstance(raw, dict):
                raise ConfigError(
                    f"config.yaml 的 `{yaml_key}` 段应为映射，实际是 {type(raw).__name__}"
                )
            try:
                sections[field_name] = model_cls(**raw)
            except ValidationError as e:
                # 直接抛 pydantic 原始异常时，报错里只有类名（如 RetrievalConfig），
                # 看不出对应 yaml 里哪一段，补上段名可以省掉一次全文搜索。
                raise ConfigError(f"config.yaml 的 `{yaml_key}` 段配置无效：\n{e}") from e

        return cls(**sections, _env_file=_env_file_arg())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


def reload_settings() -> Settings:
    """清缓存并重新装载，返回新实例。

    仅供测试与 `make doctor` 使用。注意：已经 `from ... import settings` 的模块
    仍持有旧对象，所以测试里请用本函数的返回值断言，不要断言模块级的 `settings`。
    """
    get_settings.cache_clear()
    return get_settings()


# 模块级别单例，供业务直接 import
settings: Settings = get_settings()
