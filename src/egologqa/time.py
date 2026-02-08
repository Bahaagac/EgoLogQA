from __future__ import annotations

from typing import Any


def extract_stamp_ns(message: Any, fallback_ns: int) -> tuple[int, str, bool]:
    """Return (timestamp_ns, source, used_fallback)."""
    header_ns = extract_header_stamp_ns(message)
    if is_valid_timestamp_ns(header_ns):
        return int(header_ns), "header", False
    if is_valid_timestamp_ns(fallback_ns):
        return int(fallback_ns), "log_time", True
    return 0, "invalid", True


def extract_header_stamp_ns(message: Any) -> int:
    sec = _nested_attr(message, "header.stamp.sec")
    nsec = _nested_attr(message, "header.stamp.nanosec")
    if sec is None or nsec is None:
        return 0
    try:
        header_ns = int(sec) * 1_000_000_000 + int(nsec)
    except (TypeError, ValueError):
        return 0
    return int(header_ns) if header_ns > 0 else 0


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
