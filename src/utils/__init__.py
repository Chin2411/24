import logging
from typing import Iterable, Any, List


def fix_row(row: Iterable[Any] | None, length: int, default: Any = "") -> List[Any]:
    """Return list of exactly ``length`` items from ``row``.

    Missing items are replaced with ``default`` and extra items are discarded.
    Any unexpected errors are logged and ``default`` values are returned for the
    entire row.
    """
    try:
        cells = list(row) if row else []
        if len(cells) < length:
            cells.extend([default] * (length - len(cells)))
        elif len(cells) > length:
            cells = cells[:length]
        return [default if c is None else c for c in cells]
    except Exception as exc:  # pragma: no cover - unexpected errors
        logging.getLogger(__name__).error("fix_row error: %s", exc)
        return [default for _ in range(length)]


def unpack3(row: Iterable[Any] | None, default: Any = "") -> tuple[Any, Any, Any]:
    """Return exactly three values from ``row``.

    Missing items are replaced with ``default`` and extra items are discarded.
    Any unexpected errors are logged and ``default`` values are returned.
    """
    fixed = fix_row(row, 3, default)
    return tuple(fixed[:3])  # type: ignore[misc]
