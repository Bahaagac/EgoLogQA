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
    for i in range(5):
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
    return cfg


def test_depth_non_uint16_is_aggregated_warn() -> None:
    cfg = _cfg()
    source = InMemoryMessageSource(_records())

    def fake_decode_rgb(_msg):
        return np.full((10, 10, 3), 128, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return None, "DEPTH_UNEXPECTED_DTYPE"

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=fake_decode_depth):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)

    dtype_warn = [
        err for err in result.report["errors"] if err.get("code") == "DEPTH_DTYPE_NON_UINT16_SEEN"
    ]
    assert len(dtype_warn) == 1
    assert dtype_warn[0]["severity"] == "WARN"
    assert dtype_warn[0]["context"]["count"] >= 1
