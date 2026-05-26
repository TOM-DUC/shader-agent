"""统一配置入口。

业务代码一律通过 `from shader_agent.config.settings import settings`
拿到配置，不要直接读 os.environ 或 yaml。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录：本文件位于 shader_agent/config/settings.py，向上三级
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 显式加载 .env（pydantic-settings 也会自动读，但显式更稳）
load_dotenv(PROJECT_ROOT / ".env")


class LLMConfig(BaseSettings):
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


class OrchestrationConfig(BaseSettings):
    framework: Literal["metagpt", "custom"] = "custom"


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"


class PathsConfig(BaseSettings):
    data_dir: str = "data"
    shadertoy_corpus_dir: str = "data/shadertoy_corpus"
    vector_db_dir: str = "data/vector_db"
    models_dir: str = "data/models"


class CorpusConfig(BaseSettings):
    """Shadertoy 语料库构建配置（阶段二）。"""
    shadertoy_api_base: str = "https://www.shadertoy.com/api/v1"
    max_shaders: int = 300
    min_likes: int = 30
    search_queries: list[str] = []
    per_query_limit: int = 60
    request_interval_seconds: float = 0.4
    enable_llm_tagging: bool = False
    min_code_chars: int = 200
    max_code_chars: int = 12000


class EmbeddingConfig(BaseSettings):
    """嵌入模型配置（阶段二）。"""
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    normalize_embeddings: bool = True
    batch_size: int = 8
    cache_dir: str | None = "data/models"


class VectorStoreConfig(BaseSettings):
    """向量库配置（阶段二）。"""
    collection_name: str = "shadertoy_shaders"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class Settings(BaseSettings):
    """全局配置对象。

    敏感字段（API key）来自 .env；
    其余来自 config.yaml。
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 敏感字段（来自 .env）
    deepseek_api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_BASE_URL",
    )
    # 阶段二新增；允许为空字符串（无 key 时走 seed 流程）
    shadertoy_api_key: str = Field(default="", alias="SHADERTOY_API_KEY")

    # 嵌套配置（来自 config.yaml）
    llm: LLMConfig = LLMConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    logging_cfg: LoggingConfig = LoggingConfig()
    paths: PathsConfig = PathsConfig()
    corpus: CorpusConfig = CorpusConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()

    # 项目根
    project_root: Path = PROJECT_ROOT

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

    @classmethod
    def load(cls) -> "Settings":
        """合并 .env + config.yaml 生成实例。"""
        yaml_path = PROJECT_ROOT / "config.yaml"
        yaml_data: dict = {}
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        # 取出 yaml 中的嵌套段，构造对应子配置
        llm = LLMConfig(**(yaml_data.get("llm") or {}))
        orch = OrchestrationConfig(**(yaml_data.get("orchestration") or {}))
        log_cfg = LoggingConfig(**(yaml_data.get("logging") or {}))
        paths = PathsConfig(**(yaml_data.get("paths") or {}))
        corpus = CorpusConfig(**(yaml_data.get("corpus") or {}))
        embedding = EmbeddingConfig(**(yaml_data.get("embedding") or {}))
        vstore = VectorStoreConfig(**(yaml_data.get("vector_store") or {}))

        return cls(
            llm=llm,
            orchestration=orch,
            logging_cfg=log_cfg,
            paths=paths,
            corpus=corpus,
            embedding=embedding,
            vector_store=vstore,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


# 模块级别单例，供业务直接 import
settings: Settings = get_settings()
