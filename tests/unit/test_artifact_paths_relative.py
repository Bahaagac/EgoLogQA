from __future__ import annotations

from pathlib import Path

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
    cfg.sampling.rgb_stride = 3
    cfg.sampling.max_rgb_frames = 200
    return cfg


def _records() -> list[MessageRecord]:
    out: list[MessageRecord] = []
    t0 = 1_000_000_000
    for i in range(100):
        t = t0 + i * 50_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def test_artifact_paths_are_output_relative_posix(tmp_path: Path) -> None:
    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=_cfg(),
        source=InMemoryMessageSource(_records()),
    )
    metrics = result.report["metrics"]
    for key in [
        "sync_histogram_path",
        "drop_timeline_path",
        "exposure_debug_csv_path",
        "exposure_low_clip_frames_dir",
        "exposure_high_clip_frames_dir",
        "exposure_flat_and_dark_frames_dir",
        "exposure_flat_and_bright_frames_dir",
        "exposure_evidence_error_path",
        "blur_debug_csv_path",
        "depth_debug_csv_path",
        "blur_fail_frames_dir",
        "blur_pass_frames_dir",
        "blur_fail_frames_annotated_dir",
        "blur_pass_frames_annotated_dir",
        "clean_segments_path",
        "clean_segments_nosync_path",
        "evidence_manifest_path",
        "benchmarks_path",
    ]:
        value = metrics.get(key)
        if not value:
            continue
        path = Path(value)
        assert not path.is_absolute()
        assert "\\" not in value
