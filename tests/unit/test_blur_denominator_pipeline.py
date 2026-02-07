from __future__ import annotations

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
    for i in range(4):
        t = base + i * 100_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
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
    cfg.sampling.max_rgb_frames = 100
    cfg.thresholds.blur_threshold_min = 80.0
    return cfg


def test_blur_ratio_uses_decoded_frames_only() -> None:
    cfg = _cfg()
    source = InMemoryMessageSource(_records())
    sharp = np.random.default_rng(123).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    blurry = np.full((64, 64, 3), 100, dtype=np.uint8)
    calls = {"n": 0}

    def fake_decode_rgb(_msg):
        calls["n"] += 1
        if calls["n"] <= 2:
            return None, "RGB_DECODE_FAIL"
        if calls["n"] == 3:
            return sharp, None
        return blurry, None

    def fake_decode_depth(_msg):
        return np.full((10, 10), 100, dtype=np.uint16), None

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=fake_decode_depth):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)

    metrics = result.report["metrics"]
    assert metrics["rgb_decode_attempt_count"] == 4
    assert metrics["rgb_decode_success_count"] == 2
    assert metrics["blur_valid_frame_count"] == 2
    assert metrics["blur_fail_ratio"] == 0.5
