from __future__ import annotations

from pathlib import Path

import pytest

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
    cfg.sampling.max_rgb_frames = 100
    return cfg


def _records_with_extra_topic(start_ns: int, end_ns: int) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for t_ns in (start_ns, end_ns):
        records.append(MessageRecord("/rgb", t_ns, t_ns, make_message_from_ns(t_ns, data=b"rgb")))
        records.append(MessageRecord("/depth", t_ns, t_ns, make_message_from_ns(t_ns, data=b"depth")))
        records.append(MessageRecord("/imu", t_ns, t_ns, make_message_from_ns(t_ns)))
        records.append(MessageRecord("/extra", t_ns, t_ns, make_message_from_ns(t_ns)))
    return records


def test_file_summary_metrics_include_all_topics_duration_and_bitrate(tmp_path: Path) -> None:
    cfg = _cfg()
    start_ns = 1_000_000_000
    end_ns = 3_000_000_000
    records = _records_with_extra_topic(start_ns=start_ns, end_ns=end_ns)
    source = InMemoryMessageSource(records=records)

    input_path = tmp_path / "sample.mcap"
    input_size_bytes = 4_000_000
    input_path.write_bytes(b"x" * input_size_bytes)

    result = analyze_file(
        input_path=input_path,
        output_dir=tmp_path,
        config=cfg,
        source=source,
    )
    metrics = result.report["metrics"]
    expected_duration_s = (end_ns - start_ns) / 1_000_000_000.0
    expected_bitrate_mbps = (input_size_bytes * 8.0) / (expected_duration_s * 1_000_000.0)

    assert metrics["file_total_messages"] == len(records)
    assert metrics["file_duration_s"] == pytest.approx(expected_duration_s)
    assert metrics["file_bitrate_mbps"] == pytest.approx(expected_bitrate_mbps)


def test_file_summary_bitrate_is_none_when_input_size_missing(tmp_path: Path) -> None:
    cfg = _cfg()
    records = _records_with_extra_topic(start_ns=1_000_000_000, end_ns=3_000_000_000)
    source = InMemoryMessageSource(records=records)

    result = analyze_file(
        input_path=tmp_path / "missing.mcap",
        output_dir=tmp_path,
        config=cfg,
        source=source,
    )
    metrics = result.report["metrics"]

    assert metrics["file_total_messages"] == len(records)
    assert metrics["file_duration_s"] == pytest.approx(2.0)
    assert metrics["file_bitrate_mbps"] is None


def test_file_summary_bitrate_is_none_when_duration_zero(tmp_path: Path) -> None:
    cfg = _cfg()
    records = _records_with_extra_topic(start_ns=1_000_000_000, end_ns=1_000_000_000)
    source = InMemoryMessageSource(records=records)

    input_path = tmp_path / "zero_duration.mcap"
    input_path.write_bytes(b"x" * 4096)

    result = analyze_file(
        input_path=input_path,
        output_dir=tmp_path,
        config=cfg,
        source=source,
    )
    metrics = result.report["metrics"]

    assert metrics["file_total_messages"] == len(records)
    assert metrics["file_duration_s"] == pytest.approx(0.0)
    assert metrics["file_bitrate_mbps"] is None
