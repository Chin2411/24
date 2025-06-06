import logging


def logger_flush() -> None:
    """Flush and close handlers of the root logger if needed."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        try:
            if hasattr(handler, "flush"):
                handler.flush()
            if hasattr(handler, "close"):
                handler.close()
        except Exception:
            pass
