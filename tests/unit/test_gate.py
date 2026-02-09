from __future__ import annotations

from egologqa.gate import evaluate_gate
from egologqa.models import QAConfig


def test_gate_reason_order_is_fixed() -> None:
    cfg = QAConfig()
    cfg.decode.warn_on_depth_pixel_decode_failure = True
    metrics = {
        "sync_sample_count": 64,
        "sync_p95_ms": 40.0,
        "sync_jitter_p95_ms": 8.0,
        "sync_drift_ms_per_min": 12.0,
        "drop_ratio": 0.2,
        "imu_combined_missing_ratio": 0.5,
        "blur_fail_ratio": 0.8,
        "exposure_bad_ratio": 0.9,
        "depth_fail_ratio": 0.9,
        "depth_invalid_mean": 0.9,
        "depth_decode_success_count": 64,
        "depth_valid_frame_count": 64,
    }
    streams = {
        "rgb_timestamps_present": True,
        "depth_topic_present": True,
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
        clean_segments=[],
    )
    assert out["gate"] == "FAIL"
    assert out["fail_reasons"] == [
        "FAIL_SYNC_P95_GT_FAIL",
        "FAIL_DROP_RATIO_GT_FAIL",
        "FAIL_DEPTH_FAIL_RATIO_GT_FAIL",
        "FAIL_DEPTH_INVALID_MEAN_GT_FAIL",
        "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH",
    ]
    assert out["warn_reasons"] == [
        "WARN_DEPTH_TIMESTAMP_MISSING",
        "WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED",
        "WARN_RGB_PIXEL_DECODE_UNSUPPORTED",
        "WARN_SYNC_P95_GT_WARN",
        "WARN_SYNC_JITTER_P95_GT_WARN",
        "WARN_SYNC_DRIFT_ABS_GT_WARN",
        "WARN_DROP_RATIO_GT_WARN",
        "WARN_IMU_MISSING_RATIO_GT_WARN",
        "WARN_BLUR_FAIL_RATIO_GT_WARN",
        "WARN_EXPOSURE_BAD_RATIO_GT_WARN",
        "WARN_DEPTH_INVALID_MEAN_GT_WARN",
    ]


def test_warn_floor_from_error_code() -> None:
    cfg = QAConfig()
    metrics = {"sync_sample_count": 0, "sync_p95_ms": 1.0, "drop_ratio": 0.0}
    streams = {
        "rgb_timestamps_present": True,
        "depth_topic_present": True,
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
        clean_segments=[{"start_ns": 0, "end_ns": 1_000_000_000, "duration_s": 1.0}],
    )
    assert out["gate"] == "WARN"
