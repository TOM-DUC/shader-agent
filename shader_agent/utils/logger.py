"""统一日志：基于 loguru，结合 settings.logging_cfg 配置。

import 方式：
    from shader_agent.utils.logger import logger
    logger.info("hello")
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from shader_agent.config.settings import settings

_logger.remove()  # 清掉默认 handler

# 控制台
_logger.add(
    sys.stderr,
    level=settings.logging_cfg.level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    enqueue=True,
)

# 文件
if settings.logging_cfg.log_to_file:
    log_dir = settings.project_root / settings.logging_cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    _logger.add(
        log_dir / "shader_agent_{time:YYYY-MM-DD}.log",
        level=settings.logging_cfg.level,
        rotation="20 MB",
        retention="14 days",
        encoding="utf-8",
        enqueue=True,
    )

logger = _logger
