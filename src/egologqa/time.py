from __future__ import annotations

from typing import Any, Tuple


def extract_stamp_ns(message: Any, fallback_ns: int) -> tuple[int, str, bool]:
    """Return (timestamp_ns, source, used_fallback)."""
    sec = _nested_attr(message, "header.stamp.sec")
    nsec = _nested_attr(message, "header.stamp.nanosec")
    header_ns = None
    if sec is not None and nsec is not None:
        try:
            header_ns = int(sec) * 1_000_000_000 + int(nsec)
        except (TypeError, ValueError):
            header_ns = None
    if is_valid_timestamp_ns(header_ns):
        return int(header_ns), "header", False
    if is_valid_timestamp_ns(fallback_ns):
        return int(fallback_ns), "log_time", True
    return 0, "invalid", True


def is_valid_timestamp_ns(value: Any) -> bool:
    if value is None:
        return False
    try:
        intval = int(value)
    except (TypeError, ValueError):
        return False
    return intval > 0


def _nested_attr(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur
