from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _build_records(count: int = 400, step_ns: int = 100_000_000) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    base = 1_000_000_000
    for i in range(count):
        t = base + i * step_ns
        records.append(
            MessageRecord(
                topic="/rgb",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t, data=b"rgb"),
            )
        )
        records.append(
            MessageRecord(
                topic="/depth",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t, data=b"depth"),
            )
        )
        records.append(
            MessageRecord(
                topic="/imu",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t),
            )
        )
    return records


def _base_cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 5000
    return cfg


def test_exposure_bad_does_not_erase_integrity_segments() -> None:
    cfg = _base_cfg()
    source = InMemoryMessageSource(records=_build_records())

    def fake_decode_rgb(_msg):
        return np.full((10, 10, 3), 128, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return np.full((10, 10), 100, dtype=np.uint16), None

    def fake_rgb_metrics(frames, _thresholds, sample_indices=None, sample_times_ms=None):
        n = len(frames)
        return (
            {
                "blur_median": 100.0,
                "blur_threshold": 80.0,
                "blur_fail_ratio": 0.0,
                "exposure_bad_ratio": 1.0,
                "low_clip_mean": 0.0,
                "low_clip_p95": 0.0,
                "high_clip_mean": 0.0,
                "high_clip_p95": 0.0,
                "contrast_mean": 0.0,
                "contrast_p05": 0.0,
                "dynamic_range_mean": 0.0,
                "dynamic_range_p05": 0.0,
                "p50_mean": 120.0,
                "p50_p05": 120.0,
                "p50_p95": 120.0,
                "dark_frame_ratio": 0.0,
                "low_clip_when_dark_mean": None,
                "exposure_bad_reason_counts": {
                    "low_clip": n,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [True] * n,
            [False] * n,
            [],
            [],
        )

    def fake_depth_metrics(frames, _thresholds):
        return (
            {
                "depth_invalid_mean": 0.0,
                "depth_invalid_p95": 0.0,
                "depth_fail_ratio": 0.0,
            },
            [True] * len(frames),
        )

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=fake_decode_depth), patch(
        "egologqa.pipeline.compute_rgb_pixel_metrics", side_effect=fake_rgb_metrics
    ), patch("egologqa.pipeline.compute_depth_pixel_metrics", side_effect=fake_depth_metrics):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)

    assert len(result.report["segments"]) > 0
    assert "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH" in result.report["gate"]["fail_reasons"]
    assert result.report["gate"]["gate"] == "FAIL"
    assert result.report["gate"]["recommended_action"] == "USE_SEGMENTS_ONLY"
    assert "WARN_EXPOSURE_BAD_RATIO_GT_WARN" in result.report["gate"]["warn_reasons"]


def test_exposure_compute_failure_records_error_without_erasing_segments() -> None:
    cfg = _base_cfg()
    source = InMemoryMessageSource(records=_build_records())

    def fake_decode_rgb(_msg):
        return np.full((10, 10, 3), 128, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return np.full((10, 10), 100, dtype=np.uint16), None

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
                "contrast_mean": 20.0,
                "contrast_p05": 20.0,
                "dynamic_range_mean": 20.0,
                "dynamic_range_p05": 20.0,
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
            [{"sample_i": 1, "t_ms": 1000.0, "message": "boom"}],
        )

    def fake_depth_metrics(frames, _thresholds):
        return (
            {
                "depth_invalid_mean": 0.0,
                "depth_invalid_p95": 0.0,
                "depth_fail_ratio": 0.0,
            },
            [True] * len(frames),
        )

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=fake_decode_depth), patch(
        "egologqa.pipeline.compute_rgb_pixel_metrics", side_effect=fake_rgb_metrics
    ), patch("egologqa.pipeline.compute_depth_pixel_metrics", side_effect=fake_depth_metrics):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)

    assert len(result.report["segments"]) > 0
    assert any(
        e.get("code") == "EXPOSURE_COMPUTE_FAILED" and e.get("severity") == "ERROR"
        for e in result.report["errors"]
    )
