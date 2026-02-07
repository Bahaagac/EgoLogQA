from __future__ import annotations

from pathlib import Path

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _cfg_with_depth_topic() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 5000
    return cfg


def test_missing_depth_timestamps_do_not_erase_integrity_segments(tmp_path: Path) -> None:
    cfg = _cfg_with_depth_topic()
    records: list[MessageRecord] = []
    base = 1_000_000_000
    step_ns = 33_333_333
    for i in range(360):
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
                topic="/imu",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t),
            )
        )
    source = InMemoryMessageSource(records=records)

    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=cfg,
        source=source,
    )

    assert result.report["segments"]
    assert "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH" not in result.report["gate"]["fail_reasons"]
    assert any(
        err.get("code") == "SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING"
        and err.get("severity") == "WARN"
        for err in result.report["errors"]
    )
