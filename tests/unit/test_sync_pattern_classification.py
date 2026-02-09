from __future__ import annotations

from egologqa.gate import classify_sync_pattern, sync_offset_estimate_ms
from egologqa.models import QAConfig


def test_sync_pattern_fallthrough_order() -> None:
    cfg = QAConfig()
    assert (
        classify_sync_pattern(
            {"sync_sample_count": 5, "sync_p95_ms": 100.0},
            cfg,
        )
        == "unavailable"
    )
    assert (
        classify_sync_pattern(
            {"sync_sample_count": 30, "sync_p95_ms": 10.0},
            cfg,
        )
        == "ok"
    )
    assert (
        classify_sync_pattern(
            {
                "sync_sample_count": 30,
                "sync_p95_ms": 20.0,
                "sync_signed_p50_ms": 20.0,
                "sync_signed_std_ms": 1.0,
                "sync_jitter_p95_ms": 6.0,
                "sync_drift_ms_per_min": 1.0,
            },
            cfg,
        )
        == "unstable_timing"
    )
    assert (
        classify_sync_pattern(
            {
                "sync_sample_count": 30,
                "sync_p95_ms": 20.0,
                "sync_signed_p50_ms": 20.0,
                "sync_signed_std_ms": 1.0,
                "sync_jitter_p95_ms": 1.0,
                "sync_drift_ms_per_min": 1.0,
            },
            cfg,
        )
        == "stable_offset"
    )
    assert (
        classify_sync_pattern(
            {
                "sync_sample_count": 30,
                "sync_p95_ms": 20.0,
                "sync_signed_p50_ms": 5.0,
                "sync_signed_std_ms": 1.0,
                "sync_jitter_p95_ms": 1.0,
                "sync_drift_ms_per_min": 1.0,
            },
            cfg,
        )
        == "mixed_or_unclear"
    )


def test_sync_offset_estimate_prefers_p50_then_mean() -> None:
    assert sync_offset_estimate_ms({"sync_signed_p50_ms": 12.0, "sync_signed_mean_ms": 7.0}) == 12.0
    assert sync_offset_estimate_ms({"sync_signed_mean_ms": 7.0}) == 7.0
    assert sync_offset_estimate_ms({}) is None
