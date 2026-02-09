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
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 100
    cfg.decode.warn_on_depth_pixel_decode_failure = True
    return cfg


def _records() -> list[MessageRecord]:
    out: list[MessageRecord] = []
    base = 1_000_000_000
    for i in range(4):
        t = base + i * 100_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def _error_context(report: dict[str, object], code: str) -> dict[str, object]:
    errors = report.get("errors", [])
    if not isinstance(errors, list):
        raise AssertionError("report.errors is not a list")
    for item in errors:
        if not isinstance(item, dict):
            continue
        if item.get("code") != code:
            continue
        context = item.get("context")
        if isinstance(context, dict):
            return context
    raise AssertionError(f"missing error context for code={code}")


def test_decode_warnings_include_cv2_probe_diagnostics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "egologqa.pipeline._cv2_probe",
        lambda: (False, "ImportError: No module named cv2"),
    )
    monkeypatch.setattr(
        "egologqa.pipeline.decode_rgb_message",
        lambda _msg: (None, "RGB_DECODE_FAIL"),
    )
    monkeypatch.setattr(
        "egologqa.pipeline.decode_depth_message",
        lambda _msg: (None, "DEPTH_PNG_IMDECODE_FAIL"),
    )

    result = analyze_file(
        input_path="dummy.mcap",
        output_dir=tmp_path,
        config=_cfg(),
        source=InMemoryMessageSource(_records()),
    )

    metrics = result.report["metrics"]
    assert metrics["cv2_available"] is False
    assert metrics["cv2_import_error"] == "ImportError: No module named cv2"

    rgb_ctx = _error_context(result.report, "RGB_DECODE_FAIL")
    assert rgb_ctx["cv2_available"] is False
    assert rgb_ctx["cv2_import_error"] == "ImportError: No module named cv2"
    assert rgb_ctx["rgb_decode_attempt_count"] == 4

    depth_ctx = _error_context(result.report, "DEPTH_PNG_IMDECODE_FAIL")
    assert depth_ctx["cv2_available"] is False
    assert depth_ctx["cv2_import_error"] == "ImportError: No module named cv2"
    assert depth_ctx["depth_decode_attempt_count"] == 4

    blur_ctx = _error_context(result.report, "BLUR_UNAVAILABLE_NO_DECODE")
    assert blur_ctx["cv2_available"] is False
    assert blur_ctx["cv2_import_error"] == "ImportError: No module named cv2"

    warn_reasons = result.report["gate"]["warn_reasons"]
    assert "WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED" in warn_reasons
    assert "WARN_RGB_PIXEL_DECODE_UNSUPPORTED" in warn_reasons
    assert warn_reasons.index("WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED") < warn_reasons.index(
        "WARN_RGB_PIXEL_DECODE_UNSUPPORTED"
    )
