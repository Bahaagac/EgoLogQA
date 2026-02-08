from __future__ import annotations

from pathlib import Path

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import FakeMessage, FakeHeader, FakeStamp, make_message_from_ns


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = None
    cfg.topics.imu_accel_topic = None
    cfg.topics.imu_gyro_topic = None
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 32
    return cfg


def test_timebase_metrics_emitted_when_paired_samples_exist(tmp_path: Path) -> None:
    t0 = 1_000_000_000
    records = [
        MessageRecord("/rgb", t0, t0, make_message_from_ns(t0, data=b"rgb")),
        MessageRecord("/rgb", t0 + 100_000_000, t0 + 100_000_000, make_message_from_ns(t0 + 110_000_000, data=b"rgb")),
        MessageRecord("/rgb", t0 + 200_000_000, t0 + 200_000_000, FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=0, nanosec=0)), data=b"rgb")),
        MessageRecord("/rgb", t0 + 300_000_000, t0 + 300_000_000, FakeMessage(header=None, data=b"rgb")),
        MessageRecord("/rgb", 0, 0, make_message_from_ns(t0 + 400_000_000, data=b"rgb")),
    ]
    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=_cfg(),
        source=InMemoryMessageSource(records),
    )
    metrics = result.report["metrics"]

    assert metrics["rgb_timebase_diff_sample_count"] == 2
    assert metrics["rgb_timebase_diff_signed_mean_ms"] == 5.0
    assert metrics["rgb_timebase_diff_signed_p50_ms"] == 5.0
    assert metrics["rgb_timebase_diff_abs_max_ms"] == 10.0
    assert metrics["rgb_timebase_diff_abs_p95_ms"] == 9.5
    assert metrics["rgb_timebase_header_present_ratio"] == 0.6
    assert not any(err.get("code") == "TIMEBASE_DISAGREEMENT" for err in result.report["errors"])


def test_timebase_metrics_omitted_when_no_paired_samples(tmp_path: Path) -> None:
    t0 = 1_000_000_000
    records = [
        MessageRecord(
            "/rgb",
            t0,
            t0,
            FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=0, nanosec=0)), data=b"rgb"),
        ),
        MessageRecord(
            "/rgb",
            t0 + 100_000_000,
            t0 + 100_000_000,
            FakeMessage(header=None, data=b"rgb"),
        ),
    ]
    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=_cfg(),
        source=InMemoryMessageSource(records),
    )
    metrics = result.report["metrics"]

    assert "rgb_timebase_diff_sample_count" not in metrics
    assert "rgb_timebase_header_present_ratio" not in metrics
