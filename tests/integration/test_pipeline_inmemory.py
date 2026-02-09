from __future__ import annotations

from pathlib import Path

from egologqa.models import MessageRecord, QAConfig
from egologqa.io.reader import InMemoryMessageSource
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def test_pipeline_inmemory_smoke(tmp_path: Path) -> None:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 3
    cfg.sampling.max_rgb_frames = 500

    records: list[MessageRecord] = []
    t0 = 1_000_000_000
    step = 50_000_000  # 50 ms
    for i in range(200):
        t = t0 + i * step
        records.append(
            MessageRecord(
                topic="/rgb",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t, data=b"not_an_image"),
            )
        )
        records.append(
            MessageRecord(
                topic="/depth",
                log_time_ns=t,
                publish_time_ns=t,
                msg=make_message_from_ns(t, data=b"not_a_depth"),
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

    report_path = tmp_path / "report.json"
    report_md_path = tmp_path / "report.md"
    assert report_path.exists()
    assert report_md_path.exists()
    assert result.report["gate"]["gate"] in {"PASS", "WARN", "FAIL"}
    assert set(result.report.keys()) == {
        "tool",
        "input",
        "streams",
        "time",
        "sampling",
        "metrics",
        "gate",
        "segments",
        "config_used",
        "errors",
    }
    assert "topic_stats" in result.report["streams"]
    assert "integrity_ok_ratio" in result.report["metrics"]
    assert "vision_ok_ratio" in result.report["metrics"]
    assert "exposure_bad_reason_counts" in result.report["metrics"]
    assert "segments_basis" in result.report["metrics"]
    assert "p50_mean" in result.report["metrics"]
    assert "p50_p05" in result.report["metrics"]
    assert "p50_p95" in result.report["metrics"]
    assert "dark_frame_ratio" in result.report["metrics"]
    assert "low_clip_when_dark_mean" in result.report["metrics"]
    assert "blur_debug_csv_path" in result.report["metrics"]
    assert "depth_debug_csv_path" in result.report["metrics"]
    assert "blur_fail_frames_dir" in result.report["metrics"]
    assert "blur_pass_frames_dir" in result.report["metrics"]
    assert "rgb_decode_attempt_count" in result.report["metrics"]
    assert "rgb_decode_success_count" in result.report["metrics"]
    assert "depth_decode_attempt_count" in result.report["metrics"]
    assert "depth_decode_success_count" in result.report["metrics"]
    assert "file_total_messages" in result.report["metrics"]
    assert "file_duration_s" in result.report["metrics"]
    assert "file_bitrate_mbps" in result.report["metrics"]
    assert "blur_valid_frame_count" in result.report["metrics"]
    assert "exposure_valid_frame_count" in result.report["metrics"]
    assert "depth_valid_frame_count" in result.report["metrics"]
    assert "blur_p10" in result.report["metrics"]
    assert "blur_p50" in result.report["metrics"]
    assert "blur_p90" in result.report["metrics"]
    assert "exposure_bad_first_sample_i" in result.report["metrics"]
    assert "exposure_bad_last_sample_i" in result.report["metrics"]
    assert result.report["metrics"]["file_total_messages"] == len(records)
    if result.report["errors"]:
        first = result.report["errors"][0]
        assert set(first.keys()) == {"severity", "code", "message", "context"}
