"""Initialization for src package."""

import logging
import logging.config

try:  # pragma: no cover - no need to test
    from config import LOGGING
    logging.config.dictConfig(LOGGING)
except Exception:  # pragma: no cover - if config missing
    logging.basicConfig(level=logging.INFO)
