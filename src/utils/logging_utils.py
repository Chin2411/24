import logging


def logger_flush() -> None:
    """Flush handlers of the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        try:
            if hasattr(handler, "flush"):
                handler.flush()
        except Exception:
            pass
