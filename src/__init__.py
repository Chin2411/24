from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.common.paths import LOG_PATH


def setup_logging() -> None:
    """Configure root logger for the application."""
    LOG_PATH.parent.mkdir(exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
