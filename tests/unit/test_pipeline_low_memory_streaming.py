from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from egologqa import pipeline
from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from tests.conftest import make_message_from_ns


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 10_000
    cfg.debug.export_preview_frames = True
    cfg.debug.export_evidence_frames = False
    cfg.debug.export_evidence_on_warn = False
    return cfg


def _records(n: int) -> list[MessageRecord]:
    out: list[MessageRecord] = []
    t0 = 1_000_000_000
    for i in range(n):
        t = t0 + i * 33_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def test_large_sample_run_uses_streaming_frame_materialization(monkeypatch) -> None:
    cfg = _cfg()
    sample_count = pipeline.MAX_BATCH_FRAME_CACHE_SAMPLES + 6
    source = InMemoryMessageSource(_records(sample_count))

    def fake_decode_rgb(_msg):
        return np.full((12, 12, 3), 120, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return np.full((12, 12), 100, dtype=np.uint16), None

    predecoded_flags: list[bool] = []
    original_materialize = pipeline._materialize_rgb_frames_for_positions

    def wrapped_materialize(
        source_obj,
        rgb_topic,
        sample_pos_by_rgb_idx,
        positions,
        predecoded=None,
    ):
        predecoded_flags.append(predecoded is None)
        return original_materialize(
            source_obj,
            rgb_topic,
            sample_pos_by_rgb_idx,
            positions,
            predecoded=predecoded,
        )

    monkeypatch.setattr(pipeline, "decode_rgb_message", fake_decode_rgb)
    monkeypatch.setattr(pipeline, "decode_depth_message", fake_decode_depth)
    monkeypatch.setattr(
        pipeline,
        "_materialize_rgb_frames_for_positions",
        wrapped_materialize,
    )

    with TemporaryDirectory() as d:
        result = pipeline.analyze_file("dummy.mcap", Path(d), cfg, source=source)
        metrics = result.report["metrics"]

        assert metrics.get("rgb_decode_success_count") == sample_count
        assert metrics.get("preview_count", 0) > 0
        assert predecoded_flags and all(predecoded_flags)
