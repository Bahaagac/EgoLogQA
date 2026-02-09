from __future__ import annotations

import numpy as np

from egologqa.metrics.pixel_metrics import compute_rgb_pixel_metrics
from egologqa.models import ThresholdsConfig


def _as_bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _reason_set(row: dict[str, float | int | str]) -> set[str]:
    return set(filter(None, str(row.get("reasons", "")).split(";")))


def test_low_clip_black_scene_with_highlights_is_not_exposure_bad() -> None:
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[10:50, 10:50] = 255
    frame = _as_bgr(gray)
    th = ThresholdsConfig()
    th.exposure_roi_margin_ratio = 0.0
    th.low_clip_warn = 0.20
    th.median_dark = 40.0
    th.low_clip_p95_max = 180.0
    th.high_clip_warn = 0.90

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    row = rows[0]

    assert "low_clip" in row
    assert "p50" in row
    assert "p95" in row
    assert "reasons" in row
    assert row["low_clip"] > th.low_clip_warn
    assert row["p50"] < th.median_dark
    assert row["p95"] >= th.low_clip_p95_max
    assert exposure_ok == [True]
    assert row["exposure_bad"] == 0
    assert "low_clip" not in _reason_set(row)
    assert metrics["exposure_bad_reason_counts"]["low_clip"] == 0


def test_low_clip_true_dark_underexposure_still_flags() -> None:
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[10:20, 10:20] = 30
    frame = _as_bgr(gray)
    th = ThresholdsConfig()
    th.exposure_roi_margin_ratio = 0.0
    th.low_clip_warn = 0.20
    th.median_dark = 40.0
    th.low_clip_p95_max = 180.0
    th.high_clip_warn = 0.90

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    row = rows[0]

    assert "low_clip" in row
    assert "p50" in row
    assert "p95" in row
    assert "reasons" in row
    assert row["low_clip"] > th.low_clip_warn
    assert row["p50"] < th.median_dark
    assert row["p95"] < th.low_clip_p95_max
    assert exposure_ok == [False]
    assert row["exposure_bad"] == 1
    assert "low_clip" in _reason_set(row)
    assert metrics["exposure_bad_reason_counts"]["low_clip"] == 1


def test_bright_high_high_clip_is_exposure_bad_high_clip() -> None:
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[:, :20] = 220
    frame = _as_bgr(gray)
    th = ThresholdsConfig()
    th.exposure_roi_margin_ratio = 0.0
    th.high_clip_warn = 0.20
    th.median_bright = 215.0
    th.high_clip_pixel_value = 250

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    assert exposure_ok == [False]
    assert rows[0]["exposure_bad"] == 1
    assert "high_clip" in _reason_set(rows[0])
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
    th.high_clip_warn = 0.20
    th.median_bright = 215.0
    th.high_clip_pixel_value = 250

    metrics, _, exposure_ok, rows, _ = compute_rgb_pixel_metrics([frame], th, [0], [0.0])
    row = rows[0]

    assert exposure_ok == [False]
    assert row["exposure_bad"] == 1
    assert "high_clip" in _reason_set(row)
    assert float(row["high_clip_luma"]) < th.high_clip_warn
    assert float(row["high_clip_any_channel"]) > th.high_clip_warn
    assert float(row["high_clip"]) == float(row["high_clip_any_channel"])
    assert metrics["exposure_bad_reason_counts"]["high_clip"] == 1
