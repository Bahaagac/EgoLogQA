from __future__ import annotations

from typing import Any

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_depth_message(msg: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        import cv2
    except Exception:
        return None, "DEPTH_PNG_IMDECODE_FAIL"
    payload = getattr(msg, "data", None)
    if payload is None:
        return None, "DEPTH_PNG_IMDECODE_FAIL"
    idx = payload.find(PNG_SIGNATURE)
    if idx < 0:
        return None, "DEPTH_PNG_SIGNATURE_NOT_FOUND"
    png_blob = payload[idx:]
    buf = np.frombuffer(png_blob, dtype=np.uint8)
    depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None, "DEPTH_PNG_IMDECODE_FAIL"
    if depth.dtype != np.uint16:
        return None, "DEPTH_UNEXPECTED_DTYPE"
    if depth.ndim != 2:
        return None, "DEPTH_UNEXPECTED_SHAPE"
    return depth, None
