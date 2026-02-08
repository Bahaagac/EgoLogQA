from __future__ import annotations

from pathlib import Path

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


CORE_METRIC_KEYS = [
    "sync_p50_ms",
    "sync_p95_ms",
    "sync_max_ms",
    "sync_fail_ratio",
    "expected_rgb_dt_ms",
    "drop_ratio",
    "imu_combined_missing_ratio",
    "blur_fail_ratio",
    "exposure_bad_ratio",
    "depth_invalid_mean",
    "integrity_ok_ratio",
    "vision_ok_ratio",
    "rgb_decode_attempt_count",
    "rgb_decode_success_count",
    "depth_decode_attempt_count",
    "depth_decode_success_count",
]


def _cfg(bench: bool) -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 2
    cfg.sampling.max_rgb_frames = 200
    cfg.debug.benchmarks_enabled = bench
    return cfg


def _records() -> list[MessageRecord]:
    out: list[MessageRecord] = []
    base = 1_000_000_000
    step = 40_000_000
    for i in range(120):
        t = base + i * step
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def _core_subset(report: dict) -> dict:
    metrics = report.get("metrics", {})
    return {key: metrics.get(key) for key in CORE_METRIC_KEYS}


def test_benchmarks_written_only_when_enabled_and_core_outputs_match(tmp_path: Path) -> None:
    no_bench_out = tmp_path / "no_bench"
    bench_out = tmp_path / "bench"

    no_bench = analyze_file(
        input_path="dummy.mcap",
        output_dir=no_bench_out,
        config=_cfg(False),
        source=InMemoryMessageSource(_records()),
    )
    with_bench = analyze_file(
        input_path="dummy.mcap",
        output_dir=bench_out,
        config=_cfg(True),
        source=InMemoryMessageSource(_records()),
    )

    assert no_bench.report["metrics"].get("benchmarks_path") is None
    assert (no_bench_out / "debug" / "benchmarks.json").exists() is False

    assert with_bench.report["metrics"].get("benchmarks_path") == "debug/benchmarks.json"
    assert (bench_out / "debug" / "benchmarks.json").exists()

    assert no_bench.report["gate"] == with_bench.report["gate"]
    assert no_bench.report["segments"] == with_bench.report["segments"]
    assert _core_subset(no_bench.report) == _core_subset(with_bench.report)
