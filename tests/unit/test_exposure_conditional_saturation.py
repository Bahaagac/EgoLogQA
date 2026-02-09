from __future__ import annotations

import numpy as np

from egologqa.metrics.pixel_metrics import compute_rgb_pixel_metrics
from egologqa.models import ThresholdsConfig


def _as_bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def test_low_clip_high_but_median_normal_is_not_exposure_bad() -> None:
    # 30% near-black pixels plus 70% mid-tone pixels => high low_clip, normal median.
    gray = np.full((100, 100), 80, dtype=np.uint8)
    gray[:, :30] = 0
    frame = _as_bgr(gray)
    th = ThresholdsConfig()

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    assert exposure_ok == [True]
    assert rows[0]["exposure_bad"] == 0
    assert rows[0]["low_clip"] > th.low_clip_warn
    assert rows[0]["p50"] >= th.median_dark
    assert rows[0]["reasons"] == ""
    assert metrics["exposure_bad_reason_counts"]["low_clip"] == 0


def test_dark_high_low_clip_is_exposure_bad_low_clip() -> None:
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:, 80:] = 30
    frame = _as_bgr(gray)
    th = ThresholdsConfig()

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    assert exposure_ok == [False]
    assert rows[0]["exposure_bad"] == 1
    assert "low_clip" in str(rows[0]["reasons"])
    assert metrics["exposure_bad_reason_counts"]["low_clip"] == 1


def test_bright_high_high_clip_is_exposure_bad_high_clip() -> None:
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[:, :20] = 220
    frame = _as_bgr(gray)
    th = ThresholdsConfig()

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    assert exposure_ok == [False]
    assert rows[0]["exposure_bad"] == 1
    assert "high_clip" in str(rows[0]["reasons"])
    assert metrics["exposure_bad_reason_counts"]["high_clip"] == 1


def test_flat_dark_is_exposure_bad_flat_and_dark() -> None:
    gray = np.full((100, 100), 20, dtype=np.uint8)
    frame = _as_bgr(gray)
    th = ThresholdsConfig()

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    assert exposure_ok == [False]
    assert rows[0]["exposure_bad"] == 1
    assert "flat_and_dark" in str(rows[0]["reasons"])
    assert metrics["exposure_bad_reason_counts"]["flat_and_dark"] == 1


def test_high_clip_detects_channel_clipping_even_when_luma_clip_is_low() -> None:
    # Yellow-tinted saturation can clip color channels heavily while grayscale
    # stays below the high-clip pixel threshold.
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[:, :] = np.array([0, 255, 255], dtype=np.uint8)  # BGR yellow
    th = ThresholdsConfig()

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    row = rows[0]

    assert exposure_ok == [False]
    assert row["exposure_bad"] == 1
    assert "high_clip" in str(row["reasons"])
    assert float(row["high_clip_luma"]) < th.high_clip_warn
    assert float(row["high_clip_any_channel"]) > th.high_clip_warn
    assert float(row["high_clip"]) == float(row["high_clip_any_channel"])
    assert metrics["exposure_bad_reason_counts"]["high_clip"] == 1
