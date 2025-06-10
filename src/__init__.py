from __future__ import annotations

import logging
import sys

from src.common.paths import LOG_PATH
from config import LOG_LEVEL


def setup_logging() -> None:
    """Configure root logger for the application."""
    LOG_PATH.parent.mkdir(exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.flush = file_handler.stream.flush  # type: ignore[attr-defined]

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(LOG_LEVEL)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
