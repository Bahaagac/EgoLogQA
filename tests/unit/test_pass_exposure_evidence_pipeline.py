from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 1000
    cfg.debug.export_evidence_frames = False
    cfg.debug.export_evidence_on_warn = False
    cfg.debug.export_preview_frames = True
    cfg.thresholds.pass_exposure_evidence_k = 2
    return cfg


def _records(n: int = 40) -> list[MessageRecord]:
    out: list[MessageRecord] = []
    t0 = 1_000_000_000
    for i in range(n):
        t = t0 + i * 33_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def test_pass_exposure_evidence_exports_all_detected_low_clip_frames() -> None:
    cfg = _cfg()
    source = InMemoryMessageSource(_records())
    low_clip_rows = 6

    def fake_decode_rgb(_msg):
        return np.full((16, 16, 3), 127, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return np.full((16, 16), 100, dtype=np.uint16), None

    def fake_rgb_metrics(frames, _thresholds, sample_indices=None, sample_times_ms=None):
        del sample_indices
        del sample_times_ms
        n = len(frames)
        rows = [
            {
                "sample_i": i,
                "t_ms": float(i) * 1000.0,
                "reasons": "low_clip",
                "low_clip": 0.8 - (i * 0.01),
                "high_clip": 0.0,
                "dynamic_range": 25.0,
                "p50": 35.0,
            }
            for i in range(low_clip_rows)
        ]
        return (
            {
                "blur_median": 100.0,
                "blur_threshold": 80.0,
                "blur_fail_ratio": 0.0,
                "blur_p10": 95.0,
                "blur_p50": 100.0,
                "blur_p90": 110.0,
                "exposure_bad_ratio": 0.05,
                "low_clip_mean": 0.1,
                "low_clip_p95": 0.8,
                "high_clip_mean": 0.0,
                "high_clip_p95": 0.0,
                "contrast_mean": 50.0,
                "contrast_p05": 50.0,
                "dynamic_range_mean": 60.0,
                "dynamic_range_p05": 40.0,
                "p50_mean": 100.0,
                "p50_p05": 90.0,
                "p50_p95": 110.0,
                "dark_frame_ratio": 0.0,
                "low_clip_when_dark_mean": 0.2,
                "exposure_bad_reason_counts": {
                    "low_clip": low_clip_rows,
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
        metrics = result.report["metrics"]

        assert result.report["gate"]["gate"] == "PASS"
        low_clip_dir_rel = metrics.get("exposure_low_clip_frames_dir")
        assert isinstance(low_clip_dir_rel, str)
        low_clip_dir = Path(d) / low_clip_dir_rel
        exported = sorted(low_clip_dir.glob("*.jpg"))
        assert len(exported) == low_clip_rows

        preview_relpaths = metrics.get("preview_relpaths", [])
        assert isinstance(preview_relpaths, list)
        preview_positions = {
            int(Path(rel).stem.split("_sample_")[-1])
            for rel in preview_relpaths
            if "_sample_" in Path(rel).stem
        }
        assert preview_positions.isdisjoint(set(range(low_clip_rows)))
