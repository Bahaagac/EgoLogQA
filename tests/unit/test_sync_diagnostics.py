from __future__ import annotations

import numpy as np
import pytest

from egologqa.metrics.time_metrics import compute_sync_diagnostics, compute_sync_metrics


def test_sync_diagnostics_with_linear_drift_and_zero_jitter() -> None:
    rgb = [0.0, 60_000.0, 120_000.0]
    depth = [10.0, 60_020.0, 120_030.0]

    out = compute_sync_diagnostics(rgb, depth)
    assert out["sync_signed_p50_ms"] == 20.0
    assert out["sync_signed_mean_ms"] == 20.0
    assert out["sync_signed_std_ms"] == np.std(np.array([10.0, 20.0, 30.0]), ddof=0)
    assert out["sync_drift_ms_per_min"] == pytest.approx(10.0)
    assert out["sync_jitter_p95_ms"] == pytest.approx(0.0)


def test_sync_diagnostics_omit_drift_and_jitter_for_small_n() -> None:
    out = compute_sync_diagnostics([0.0, 10.0], [1.0, 11.0])
    assert "sync_signed_p50_ms" in out
    assert "sync_signed_std_ms" in out
    assert "sync_drift_ms_per_min" not in out
    assert "sync_jitter_p95_ms" not in out


def test_sync_diagnostics_omit_drift_and_jitter_when_var_zero() -> None:
    out = compute_sync_diagnostics([100.0, 100.0, 100.0], [101.0, 102.0, 103.0])
    assert "sync_signed_p50_ms" in out
    assert "sync_signed_std_ms" in out
    assert "sync_drift_ms_per_min" not in out
    assert "sync_jitter_p95_ms" not in out


def test_sync_diagnostics_numerically_stable_with_large_timestamps() -> None:
    base = 1_000_000_000_000.0
    rgb = [base + i * 60_000.0 for i in range(4)]
    # d_ms = 5 + 2*x_min where x_min in [0,1,2,3]
    depth = [rgb[0] + 5.0, rgb[1] + 7.0, rgb[2] + 9.0, rgb[3] + 11.0]

    out = compute_sync_diagnostics(rgb, depth)
    assert out["sync_drift_ms_per_min"] == pytest.approx(2.0)
    assert out["sync_jitter_p95_ms"] == pytest.approx(0.0)


def test_existing_sync_metrics_contract_unchanged() -> None:
    rgb = [10.0, 20.0, 40.0]
    depth = [0.0, 18.0, 44.0]
    out = compute_sync_metrics(rgb, depth, sync_fail_ms=33.0)
    assert out["sync_p50_ms"] == 4.0
    assert out["sync_p95_ms"] == pytest.approx(7.6)
    assert out["sync_max_ms"] == 8.0
    assert out["sync_fail_ratio"] == 0.0
