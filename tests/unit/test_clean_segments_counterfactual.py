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


def _stable_sync_metrics(sync_p95_ms: float) -> dict[str, float]:
    return {
        "sync_sample_count": 40,
        "sync_p95_ms": sync_p95_ms,
        "sync_signed_p50_ms": 20.0,
        "sync_signed_std_ms": 1.0,
        "sync_jitter_p95_ms": 1.0,
        "sync_drift_ms_per_min": 1.0,
        "drop_ratio": 0.0,
        "depth_decode_success_count": 40,
        "depth_valid_frame_count": 40,
        "depth_fail_ratio": 0.0,
        "depth_invalid_mean": 0.0,
    }


def test_no_clean_downgrades_when_nosync_counterfactual_has_segments() -> None:
    cfg = QAConfig()
    out = evaluate_gate(
        config=cfg,
        metrics=_stable_sync_metrics(sync_p95_ms=20.0),
        streams=_streams(),
        duration_s=40.0,
        segments=[],
        errors=[],
        clean_segments=[],
        clean_segments_nosync=[{"start_ns": 0, "end_ns": 9_000_000_000, "duration_s": 9.0}],
    )
    assert out["gate"] == "WARN"
    assert out["fail_reasons"] == []
    assert "WARN_SYNC_P95_GT_WARN" in out["warn_reasons"]
    assert out["recommended_action"] == "FIX_TIME_ALIGNMENT"


def test_no_clean_not_downgraded_when_counterfactual_is_also_empty() -> None:
    cfg = QAConfig()
    out = evaluate_gate(
        config=cfg,
        metrics=_stable_sync_metrics(sync_p95_ms=20.0),
        streams=_streams(),
        duration_s=40.0,
        segments=[],
        errors=[],
        clean_segments=[],
        clean_segments_nosync=[],
    )
    assert out["gate"] == "FAIL"
    assert out["fail_reasons"] == ["FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH"]


def test_sync_fail_stays_fail_but_uses_fix_time_alignment_action() -> None:
    cfg = QAConfig()
    out = evaluate_gate(
        config=cfg,
        metrics=_stable_sync_metrics(sync_p95_ms=40.0),
        streams=_streams(),
        duration_s=40.0,
        segments=[],
        errors=[],
        clean_segments=[],
        clean_segments_nosync=[{"start_ns": 0, "end_ns": 9_000_000_000, "duration_s": 9.0}],
    )
    assert out["gate"] == "FAIL"
    assert "FAIL_SYNC_P95_GT_FAIL" in out["fail_reasons"]
    assert out["recommended_action"] == "FIX_TIME_ALIGNMENT"
