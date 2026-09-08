"""项目统一日志配置。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from base import config as cfg
from medical.privacy import redact_sensitive_text

_HANDLER_MARK = "_med_agent_managed_handler"


class RedactingFormatter(logging.Formatter):
    """在最终日志文本中遮盖常见直接标识符，包括异常堆栈。"""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record))


def _level() -> int:
    level = getattr(logging, cfg.LOG_LEVEL, None)
    return level if isinstance(level, int) else logging.INFO


def _formatter() -> RedactingFormatter:
    return RedactingFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging() -> Path | None:
    """幂等初始化控制台与轮转文件日志，返回实际日志路径。"""
    root_logger = logging.getLogger()
    level = _level()
    root_logger.setLevel(level)

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(_formatter())
        setattr(console, _HANDLER_MARK, "console")
        root_logger.addHandler(console)

    if not cfg.ENABLE_FILE_LOG:
        return None

    log_path = Path(cfg.ensure_in_root(cfg.LOG_PATH))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = next(
        (
            handler
            for handler in root_logger.handlers
            if getattr(handler, _HANDLER_MARK, None) == "file"
            and Path(getattr(handler, "baseFilename", "")).resolve() == log_path
        ),
        None,
    )
    if existing is None:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=cfg.LOG_MAX_BYTES,
            backupCount=cfg.LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_formatter())
        setattr(file_handler, _HANDLER_MARK, "file")
        root_logger.addHandler(file_handler)

    # 降低常见依赖库的噪声，项目自身仍按 LOG_LEVEL 输出。
    for logger_name in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    return log_path
