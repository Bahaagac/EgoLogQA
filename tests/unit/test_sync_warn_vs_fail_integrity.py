from __future__ import annotations

from pathlib import Path

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _base_cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 5000
    cfg.thresholds.sync_warn_ms = 16.0
    cfg.thresholds.sync_fail_ms = 33.0
    return cfg


def _build_offset_records(
    count: int = 360,
    rgb_step_ns: int = 33_333_333,
    depth_offset_ns: int = 16_600_000,
) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    base = 1_000_000_000
    for i in range(count):
        t_rgb = base + i * rgb_step_ns
        t_depth = t_rgb + depth_offset_ns
        records.append(
            MessageRecord(
                topic="/rgb",
                log_time_ns=t_rgb,
                publish_time_ns=t_rgb,
                msg=make_message_from_ns(t_rgb, data=b"rgb"),
            )
        )
        records.append(
            MessageRecord(
                topic="/depth",
                log_time_ns=t_depth,
                publish_time_ns=t_depth,
                msg=make_message_from_ns(t_depth, data=b"depth"),
            )
        )
        records.append(
            MessageRecord(
                topic="/imu",
                log_time_ns=t_rgb,
                publish_time_ns=t_rgb,
                msg=make_message_from_ns(t_rgb),
            )
        )
    return records


def test_sync_warn_threshold_does_not_shred_integrity_segments(tmp_path: Path) -> None:
    cfg = _base_cfg()
    source = InMemoryMessageSource(records=_build_offset_records())
    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=cfg,
        source=source,
    )

    warn_reasons = result.report["gate"]["warn_reasons"]
    fail_reasons = result.report["gate"]["fail_reasons"]
    assert "WARN_SYNC_P95_GT_WARN" in warn_reasons
    assert "FAIL_SYNC_P95_GT_FAIL" not in fail_reasons
    assert result.report["metrics"]["sync_p95_ms"] is not None
    assert result.report["metrics"]["sync_p95_ms"] > cfg.thresholds.sync_warn_ms
    assert result.report["metrics"]["sync_p95_ms"] < cfg.thresholds.sync_fail_ms
    assert result.report["metrics"]["integrity_ok_ratio"] is not None
    assert result.report["metrics"]["integrity_ok_ratio"] > 0.9
    assert result.report["segments"]
    assert max(seg["duration_s"] for seg in result.report["segments"]) >= 10.0
    assert result.report["gate"]["recommended_action"] == "FIX_TIME_ALIGNMENT"
