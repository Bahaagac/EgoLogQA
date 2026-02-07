from __future__ import annotations

from egologqa.gate import evaluate_gate
from egologqa.models import QAConfig


def _base_streams() -> dict[str, object]:
    return {
        "rgb_timestamps_present": True,
        "depth_timestamps_present": True,
        "decode_status": {"rgb_pixels": "supported", "depth_pixels": "supported"},
    }


def _base_metrics() -> dict[str, float | None]:
    return {
        "sync_p95_ms": 1.0,
        "drop_ratio": 0.0,
        "imu_combined_missing_ratio": 0.0,
        "depth_invalid_mean": 0.0,
    }


def test_blur_warn_not_triggered_when_ratio_is_null() -> None:
    cfg = QAConfig()
    metrics = _base_metrics()
    metrics["blur_fail_ratio"] = None
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_base_streams(),
        duration_s=10.0,
        segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
        errors=[],
    )
    assert "WARN_BLUR_FAIL_RATIO_GT_WARN" not in out["warn_reasons"]


def test_blur_warn_triggered_when_ratio_exceeds_threshold() -> None:
    cfg = QAConfig()
    metrics = _base_metrics()
    metrics["blur_fail_ratio"] = cfg.thresholds.blur_fail_warn_ratio + 0.01
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_base_streams(),
        duration_s=10.0,
        segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
        errors=[],
    )
    assert "WARN_BLUR_FAIL_RATIO_GT_WARN" in out["warn_reasons"]
