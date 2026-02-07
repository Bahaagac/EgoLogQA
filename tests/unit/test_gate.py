from __future__ import annotations

from egologqa.gate import evaluate_gate
from egologqa.models import QAConfig


def test_gate_reason_order_is_fixed() -> None:
    cfg = QAConfig()
    metrics = {
        "sync_p95_ms": 40.0,
        "drop_ratio": 0.2,
        "imu_combined_missing_ratio": 0.5,
        "blur_fail_ratio": 0.8,
        "exposure_bad_ratio": 0.9,
        "depth_invalid_mean": 0.9,
    }
    streams = {
        "rgb_timestamps_present": True,
        "depth_timestamps_present": False,
        "decode_status": {"rgb_pixels": "unsupported", "depth_pixels": "unsupported"},
    }
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=streams,
        duration_s=40.0,
        segments=[],
        errors=[],
    )
    assert out["gate"] == "FAIL"
    assert out["fail_reasons"] == [
        "FAIL_SYNC_P95_GT_FAIL",
        "FAIL_DROP_RATIO_GT_FAIL",
        "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH",
    ]
    assert out["warn_reasons"][0] == "WARN_DEPTH_TIMESTAMP_MISSING"


def test_warn_floor_from_error_code() -> None:
    cfg = QAConfig()
    metrics = {"sync_p95_ms": 1.0, "drop_ratio": 0.0}
    streams = {
        "rgb_timestamps_present": True,
        "depth_timestamps_present": True,
        "decode_status": {"rgb_pixels": "supported", "depth_pixels": "supported"},
    }
    errors = [
        {
            "severity": "WARN",
            "code": "TIMESTAMP_OUT_OF_ORDER_HIGH",
            "message": "x",
            "context": {},
        }
    ]
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=streams,
        duration_s=10.0,
        segments=[{"start_ns": 0, "end_ns": 1_000_000_000, "duration_s": 1.0}],
        errors=errors,
    )
    assert out["gate"] == "WARN"
