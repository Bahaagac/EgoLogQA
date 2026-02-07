from __future__ import annotations

import json

from egologqa.report import sanitize_json_value


def test_report_sanitize_round_and_null() -> None:
    payload = {
        "a": 1.234567,
        "b": float("nan"),
        "c": float("inf"),
        "d": {"x": -1.23456},
    }
    out = sanitize_json_value(payload)
    assert out["a"] == 1.2346
    assert out["b"] is None
    assert out["c"] is None
    assert out["d"]["x"] == -1.2346
    json.dumps(out)
