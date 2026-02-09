from __future__ import annotations

from egologqa.gate import evaluate_gate
from egologqa.models import QAConfig


def _streams() -> dict[str, object]:
    return {
        "rgb_timestamps_present": True,
        "depth_topic_present": True,
        "depth_timestamps_present": True,
        "decode_status": {"rgb_pixels": "supported", "depth_pixels": "supported"},
    }


def _base_metrics() -> dict[str, float]:
    return {
        "sync_sample_count": 40,
        "sync_p95_ms": 5.0,
        "sync_signed_p50_ms": 0.0,
        "sync_signed_std_ms": 0.1,
        "sync_jitter_p95_ms": 0.1,
        "sync_drift_ms_per_min": 0.1,
        "drop_ratio": 0.0,
        "imu_combined_missing_ratio": 0.0,
        "blur_fail_ratio": 0.0,
        "exposure_bad_ratio": 0.0,
        "depth_decode_success_count": 40,
        "depth_valid_frame_count": 40,
        "depth_fail_ratio": 0.0,
        "depth_invalid_mean": 0.0,
    }


def test_gate_precedence_cases() -> None:
    cfg = QAConfig()

    metrics = _base_metrics()
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_streams(),
        duration_s=10.0,
        segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
        errors=[],
        clean_segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
    )
    assert out["gate"] == "PASS"
    assert out["recommended_action"] == "USE_FULL_SEQUENCE"

    metrics = _base_metrics()
    metrics.update(
        {
            "sync_p95_ms": 20.0,
            "sync_signed_p50_ms": 20.0,
            "sync_signed_std_ms": 1.0,
            "sync_jitter_p95_ms": 1.0,
            "sync_drift_ms_per_min": 1.0,
        }
    )
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_streams(),
        duration_s=10.0,
        segments=[],
        errors=[],
        clean_segments=[],
    )
    assert out["gate"] == "WARN"
    assert out["recommended_action"] == "FIX_TIME_ALIGNMENT"

    metrics = _base_metrics()
    metrics.update(
        {
            "sync_p95_ms": 20.0,
            "sync_signed_p50_ms": 20.0,
            "sync_signed_std_ms": 1.0,
            "sync_jitter_p95_ms": 7.0,
            "sync_drift_ms_per_min": 1.0,
        }
    )
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_streams(),
        duration_s=10.0,
        segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
        errors=[],
        clean_segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
    )
    assert out["gate"] == "WARN"
    assert out["recommended_action"] == "USE_SEGMENTS_ONLY"

    metrics = _base_metrics()
    metrics.update(
        {
            "sync_p95_ms": 40.0,
            "sync_signed_p50_ms": 20.0,
            "sync_signed_std_ms": 1.0,
            "sync_jitter_p95_ms": 1.0,
            "sync_drift_ms_per_min": 1.0,
            "depth_fail_ratio": cfg.thresholds.depth_fail_ratio_fail,
        }
    )
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_streams(),
        duration_s=40.0,
        segments=[],
        errors=[],
        clean_segments=[],
        clean_segments_nosync=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
    )
    assert out["gate"] == "FAIL"
    assert out["recommended_action"] == "RECAPTURE_OR_SKIP"

    metrics = _base_metrics()
    metrics.update({"sync_sample_count": 0, "sync_p95_ms": 5.0})
    out = evaluate_gate(
        config=cfg,
        metrics=metrics,
        streams=_streams(),
        duration_s=40.0,
        segments=[{"start_ns": 0, "end_ns": 6_000_000_000, "duration_s": 6.0}],
        errors=[],
        clean_segments=[],
    )
    assert out["gate"] == "FAIL"
    assert out["fail_reasons"] == ["FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH"]
    assert out["recommended_action"] == "USE_SEGMENTS_ONLY"
