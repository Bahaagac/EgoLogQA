from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _records() -> list[MessageRecord]:
    out: list[MessageRecord] = []
    base = 1_000_000_000
    for i in range(50):
        t = base + i * 100_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"x")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"y")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 1000
    return cfg


def _fake_depth_metrics(frames, _thresholds):
    return (
        {
            "depth_invalid_mean": 0.0,
            "depth_invalid_p95": 0.0,
            "depth_fail_ratio": 0.0,
        },
        [True] * len(frames),
    )


def _fake_decode_rgb(_msg):
    return np.full((10, 10, 3), 127, dtype=np.uint8), None


def _fake_decode_depth(_msg):
    return np.full((10, 10), 50, dtype=np.uint16), None


def test_exposure_debug_csv_written_when_enabled() -> None:
    cfg = _cfg()
    cfg.debug.export_exposure_csv = True
    source = InMemoryMessageSource(_records())

    def fake_rgb_metrics(frames, _thresholds, sample_indices=None, sample_times_ms=None):
        n = len(frames)
        rows = [
            {
                "sample_i": int(sample_indices[i] if sample_indices else i),
                "t_ms": float(sample_times_ms[i] if sample_times_ms else i),
                "roi_margin_ratio": 0.05,
                "low_clip": 0.0,
                "high_clip": 0.0,
                "p01": 10.0,
                "p05": 20.0,
                "p50": 120.0,
                "p95": 200.0,
                "p99": 230.0,
                "contrast": 220.0,
                "dynamic_range": 180.0,
                "exposure_bad": 0,
                "reasons": "",
            }
            for i in range(n)
        ]
        return (
            {
                "blur_median": 100.0,
                "blur_threshold": 80.0,
                "blur_fail_ratio": 0.0,
                "exposure_bad_ratio": 0.0,
                "low_clip_mean": 0.0,
                "low_clip_p95": 0.0,
                "high_clip_mean": 0.0,
                "high_clip_p95": 0.0,
                "contrast_mean": 220.0,
                "contrast_p05": 220.0,
                "dynamic_range_mean": 180.0,
                "dynamic_range_p05": 180.0,
                "p50_mean": 120.0,
                "p50_p05": 120.0,
                "p50_p95": 120.0,
                "dark_frame_ratio": 0.0,
                "low_clip_when_dark_mean": None,
                "exposure_bad_reason_counts": {
                    "low_clip": 0,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [True] * n,
            [True] * n,
            rows,
            [],
        )

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=_fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=_fake_decode_depth), patch(
        "egologqa.pipeline.compute_rgb_pixel_metrics", side_effect=fake_rgb_metrics
    ), patch("egologqa.pipeline.compute_depth_pixel_metrics", side_effect=_fake_depth_metrics):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)
        rel = result.report["metrics"]["exposure_debug_csv_path"]
        assert rel is not None
        csv_path = Path(d) / rel
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert "high_clip_luma" in reader.fieldnames
            assert "high_clip_any_channel" in reader.fieldnames


def test_exposure_debug_csv_not_written_when_disabled() -> None:
    cfg = _cfg()
    cfg.debug.export_exposure_csv = False
    source = InMemoryMessageSource(_records())

    def fake_rgb_metrics(frames, _thresholds, sample_indices=None, sample_times_ms=None):
        n = len(frames)
        return (
            {
                "blur_median": 100.0,
                "blur_threshold": 80.0,
                "blur_fail_ratio": 0.0,
                "exposure_bad_ratio": 0.0,
                "low_clip_mean": 0.0,
                "low_clip_p95": 0.0,
                "high_clip_mean": 0.0,
                "high_clip_p95": 0.0,
                "contrast_mean": 220.0,
                "contrast_p05": 220.0,
                "dynamic_range_mean": 180.0,
                "dynamic_range_p05": 180.0,
                "p50_mean": 120.0,
                "p50_p05": 120.0,
                "p50_p95": 120.0,
                "dark_frame_ratio": 0.0,
                "low_clip_when_dark_mean": None,
                "exposure_bad_reason_counts": {
                    "low_clip": 0,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [True] * n,
            [True] * n,
            [],
            [],
        )

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=_fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=_fake_decode_depth), patch(
        "egologqa.pipeline.compute_rgb_pixel_metrics", side_effect=fake_rgb_metrics
    ), patch("egologqa.pipeline.compute_depth_pixel_metrics", side_effect=_fake_depth_metrics):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)
        assert result.report["metrics"]["exposure_debug_csv_path"] is None
        assert not (Path(d) / "debug" / "exposure_samples.csv").exists()


def test_exposure_debug_unavailable_warn_when_enabled_and_no_rows() -> None:
    cfg = _cfg()
    cfg.debug.export_exposure_csv = True
    source = InMemoryMessageSource(_records())

    def fake_rgb_metrics(frames, _thresholds, sample_indices=None, sample_times_ms=None):
        n = len(frames)
        return (
            {
                "blur_median": 100.0,
                "blur_threshold": 80.0,
                "blur_fail_ratio": 0.0,
                "exposure_bad_ratio": 0.0,
                "low_clip_mean": None,
                "low_clip_p95": None,
                "high_clip_mean": None,
                "high_clip_p95": None,
                "contrast_mean": None,
                "contrast_p05": None,
                "dynamic_range_mean": None,
                "dynamic_range_p05": None,
                "p50_mean": None,
                "p50_p05": None,
                "p50_p95": None,
                "dark_frame_ratio": None,
                "low_clip_when_dark_mean": None,
                "exposure_bad_reason_counts": {
                    "low_clip": 0,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [True] * n,
            [True] * n,
            [],
            [],
        )

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=_fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=_fake_decode_depth), patch(
        "egologqa.pipeline.compute_rgb_pixel_metrics", side_effect=fake_rgb_metrics
    ), patch("egologqa.pipeline.compute_depth_pixel_metrics", side_effect=_fake_depth_metrics):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)
        assert any(
            err.get("code") == "RGB_EXPOSURE_DEBUG_UNAVAILABLE"
            for err in result.report["errors"]
        )
