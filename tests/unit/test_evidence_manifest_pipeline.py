from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = None
    cfg.topics.imu_accel_topic = None
    cfg.topics.imu_gyro_topic = None
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 64
    cfg.debug.export_evidence_frames = False
    cfg.debug.export_evidence_on_warn = False
    cfg.debug.write_evidence_manifest = True
    cfg.debug.write_annotated_evidence = True
    cfg.debug.evidence_frames_k = 4
    return cfg


def _records() -> list[MessageRecord]:
    base = 1_000_000_000
    step = 33_333_333
    out: list[MessageRecord] = []
    for i in range(6):
        t = base + i * step
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=f"rgb{i}".encode("utf-8"))))
    return out


def _fake_decode_rgb_message(msg):
    token = getattr(msg, "data", b"")
    level = int(token[-1:]) if token and token[-1:] in b"0123456789" else 0
    frame = np.full((24, 24, 3), level * 20, dtype=np.uint8)
    return frame, None


def _fake_compute_rgb_pixel_metrics(rgb_frames, thresholds, sample_indices=None, sample_times_ms=None):
    n = len(rgb_frames)
    sample_indices = sample_indices or list(range(n))
    sample_times_ms = sample_times_ms or [float(i) for i in range(n)]
    blur_values = [10.0 if i < n - 1 else 200.0 for i in range(n)]
    blur_ok = [v >= thresholds.blur_threshold_min for v in blur_values]
    exposure_ok = [True] * n

    rows = []
    for i in range(n):
        rows.append(
            {
                "sample_i": int(sample_indices[i]),
                "t_ms": float(sample_times_ms[i]),
                "roi_margin_ratio": float(thresholds.exposure_roi_margin_ratio),
                "blur_roi_margin_ratio": float(thresholds.blur_roi_margin_ratio),
                "blur_value": float(blur_values[i]),
                "low_clip": 0.0,
                "high_clip": 0.0,
                "p01": 0.0,
                "p05": 1.0,
                "p50": 100.0,
                "p95": 120.0,
                "p99": 130.0,
                "contrast": 130.0,
                "dynamic_range": 119.0,
                "exposure_bad": 0,
                "reasons": "",
            }
        )

    metrics = {
        "blur_median": float(np.median(np.asarray(blur_values, dtype=np.float64))) if blur_values else None,
        "blur_threshold": float(thresholds.blur_threshold_min),
        "blur_fail_ratio": float(np.mean([not x for x in blur_ok])) if blur_ok else None,
        "blur_p10": 10.0 if blur_values else None,
        "blur_p50": 10.0 if blur_values else None,
        "blur_p90": 200.0 if blur_values else None,
        "blur_valid_frame_count": n,
        "exposure_bad_ratio": 0.0 if rows else None,
        "exposure_valid_frame_count": n,
        "exposure_bad_first_sample_i": None,
        "exposure_bad_last_sample_i": None,
        "low_clip_mean": 0.0,
        "low_clip_p95": 0.0,
        "high_clip_mean": 0.0,
        "high_clip_p95": 0.0,
        "contrast_mean": 130.0,
        "contrast_p05": 130.0,
        "dynamic_range_mean": 119.0,
        "dynamic_range_p05": 119.0,
        "p50_mean": 100.0,
        "p50_p05": 100.0,
        "p50_p95": 100.0,
        "dark_frame_ratio": 0.0,
        "low_clip_when_dark_mean": None,
        "exposure_bad_reason_counts": {
            "low_clip": 0,
            "high_clip": 0,
            "flat_and_dark": 0,
            "flat_and_bright": 0,
        },
    }
    return metrics, blur_ok, exposure_ok, rows, []


def test_manifest_and_evidence_paths_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("egologqa.pipeline.decode_rgb_message", _fake_decode_rgb_message)
    monkeypatch.setattr("egologqa.pipeline.compute_rgb_pixel_metrics", _fake_compute_rgb_pixel_metrics)

    cfg = _cfg()
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    result1 = analyze_file(
        input_path="dummy.mcap",
        output_dir=out1,
        config=cfg,
        source=InMemoryMessageSource(_records()),
    )
    result2 = analyze_file(
        input_path="dummy.mcap",
        output_dir=out2,
        config=cfg,
        source=InMemoryMessageSource(_records()),
    )

    metrics = result1.report["metrics"]
    assert metrics["blur_fail_frames_dir"] == "debug/blur_fail_frames"
    assert metrics["blur_pass_frames_dir"] == "debug/blur_pass_frames"
    assert metrics["blur_fail_frames_annotated_dir"] == "debug/blur_fail_frames_annotated"
    assert metrics["blur_pass_frames_annotated_dir"] == "debug/blur_pass_frames_annotated"
    assert metrics["evidence_manifest_path"] == "debug/evidence_manifest.json"

    manifest1 = (out1 / "debug" / "evidence_manifest.json").read_bytes()
    manifest2 = (out2 / "debug" / "evidence_manifest.json").read_bytes()
    assert manifest1 == manifest2


def test_manifest_written_with_empty_sets_when_cv2_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("egologqa.pipeline.decode_rgb_message", _fake_decode_rgb_message)
    monkeypatch.setattr("egologqa.pipeline.compute_rgb_pixel_metrics", _fake_compute_rgb_pixel_metrics)
    monkeypatch.setattr("egologqa.pipeline._cv2_available", lambda: False)

    cfg = _cfg()
    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=cfg,
        source=InMemoryMessageSource(_records()),
    )

    metrics = result.report["metrics"]
    assert metrics["evidence_manifest_path"] == "debug/evidence_manifest.json"
    assert metrics["blur_fail_frames_dir"] is None
    assert metrics["blur_pass_frames_dir"] is None
    assert "blur_fail_frames_annotated_dir" not in metrics
    assert "blur_pass_frames_annotated_dir" not in metrics

    payload = json.loads((tmp_path / "debug" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert payload["selection_context"]["cv2_available"] is False
    assert payload["disabled_reason"] == "cv2_unavailable"
    assert payload["evidence_sets"]["blur_fail"] == []
    assert payload["evidence_sets"]["blur_pass"] == []
