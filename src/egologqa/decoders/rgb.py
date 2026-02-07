from __future__ import annotations

from typing import Any

import numpy as np


def decode_rgb_message(msg: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        import cv2
    except Exception:
        return None, "RGB_DECODE_FAIL"
    payload = getattr(msg, "data", None)
    if payload is None:
        return None, "RGB_DECODE_FAIL"
    buf = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "RGB_DECODE_FAIL"
    return frame, None
