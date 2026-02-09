from __future__ import annotations

import json
from pathlib import Path

from egologqa.ai_summary import (
    build_curated_context,
    deterministic_action_line,
    generate_summary_for_ui,
    resolve_gemini_api_key,
)


def _report_template() -> dict:
    return {
        "gate": {
            "gate": "WARN",
            "recommended_action": "USE_SEGMENTS_ONLY",
            "fail_reasons": [],
            "warn_reasons": ["WARN_SYNC_P95_GT_WARN"],
        },
        "streams": {
            "rgb_timestamps_present": True,
            "depth_topic_present": True,
            "depth_timestamps_present": True,
            "decode_status": {"rgb_pixels": "supported", "depth_pixels": "supported"},
            "rgb_topic": "/rgb",
            "depth_topic": "/depth",
            "imu_accel_topic": "/imu/accel",
            "imu_gyro_topic": "/imu/gyro",
        },
        "metrics": {
            "sync_p95_ms": 20.0,
            "drop_ratio": 0.02,
            "integrity_ok_ratio": 0.85,
            "vision_ok_ratio": 0.7,
            "blur_fail_ratio": 0.15,
            "exposure_bad_ratio": 0.11,
            "depth_fail_ratio": 0.08,
            "depth_invalid_mean": 0.05,
            "sync_histogram_path": "plots/sync_histogram.png",
            "clean_segments_path": "debug/clean_segments.json",
            "clean_segments_nosync_path": "debug/clean_segments_nosync.json",
            "exposure_bad_reason_counts": {"high_clip": 3},
            "out_of_order": {"rgb": {"count": 0, "ratio": 0.0}},
        },
        "segments": [
            {"start_ns": 1, "end_ns": 2, "duration_s": 1.2},
        ],
        "errors": [
            {
                "severity": "WARN",
                "code": "WARN_SYNC_P95_GT_WARN",
                "message": "sync warning",
                "context": {"sync_p95_ms": 20.0, "internal": "hidden"},
            }
        ],
    }


def _write_clean_segments(output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = [{"start_ns": 1, "end_ns": 2, "duration_s": 2.5}]
    (debug_dir / "clean_segments.json").write_text(json.dumps(payload), encoding="utf-8")
    (debug_dir / "clean_segments_nosync.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_gemini_api_key_env_precedence(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert resolve_gemini_api_key({"GOOGLE_API_KEY": "secret-google"}) == "google-key"


def test_resolve_gemini_api_key_uses_secrets_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_gemini_api_key({"GEMINI_API_KEY": "secret-gemini"}) == "secret-gemini"


def test_build_curated_context_includes_expected_and_excludes_artifacts(tmp_path: Path) -> None:
    report = _report_template()
    _write_clean_segments(tmp_path)

    context = build_curated_context(report, tmp_path)

    assert context["gate"]["recommended_action"] == "USE_SEGMENTS_ONLY"
    assert context["segments"]["integrity"]["count"] == 1
    assert context["segments"]["clean"]["count"] == 1
    assert context["segments"]["clean_nosync"]["count"] == 1
    assert "sync_p95_ms" in context["metrics"]
    assert "sync_histogram_path" not in context["metrics"]
    assert context["error_counts"]["warn_codes"][0]["code"] == "WARN_SYNC_P95_GT_WARN"
    assert "context" not in json.dumps(context["error_counts"])


def test_deterministic_action_line_mappings() -> None:
    assert deterministic_action_line("USE_FULL_SEQUENCE") == "Action: Use the full sequence."
    assert deterministic_action_line("USE_SEGMENTS_ONLY") == "Action: Use clean segments only."
    assert deterministic_action_line("FIX_TIME_ALIGNMENT") == "Action: Fix RGB-depth alignment, then re-run."
    assert deterministic_action_line("RECAPTURE_OR_SKIP") == "Action: Recapture or skip this recording."
    assert "conservative" in deterministic_action_line("UNKNOWN")


def test_generate_summary_without_api_key_returns_fallback(monkeypatch, tmp_path: Path) -> None:
    report = _report_template()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    out = generate_summary_for_ui(
        report=report,
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model="",
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_API_KEY_MISSING"
    assert out["action_line"] == "Action: Use clean segments only."


def test_generate_summary_request_failure_falls_back(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    monkeypatch.setattr(
        ai_summary,
        "_request_gemini_json_payload",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("AI_SUMMARY_SDK_IMPORT_FAILED")),
    )

    out = ai_summary.generate_summary_for_ui(
        report=_report_template(),
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model="gemini-2.5-flash",
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_SDK_IMPORT_FAILED"


def test_generate_summary_invalid_json_falls_back(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    monkeypatch.setattr(ai_summary, "_request_gemini_json_payload", lambda **_kwargs: "not-json")

    out = ai_summary.generate_summary_for_ui(
        report=_report_template(),
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model="gemini-2.5-flash",
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_JSON_PARSE_FAILED"


def test_generate_summary_normalizes_long_multiline_response(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    long_line = "First line\n" + ("very long detail " * 20)
    monkeypatch.setattr(
        ai_summary,
        "_request_gemini_json_payload",
        lambda **_kwargs: json.dumps({"summary_line": long_line}),
    )

    out = ai_summary.generate_summary_for_ui(
        report=_report_template(),
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model="gemini-2.5-flash",
    )

    assert out["source"] == "gemini"
    assert "\n" not in out["summary_line"]
    assert len(out["summary_line"]) <= 120
    assert out["action_line"] == "Action: Use clean segments only."


def test_generate_summary_disabled_returns_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")

    out = generate_summary_for_ui(
        report=_report_template(),
        output_dir=tmp_path,
        secrets=None,
        enabled=False,
        model="gemini-2.5-flash",
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_DISABLED"


def test_request_gemini_payload_omits_additional_properties(monkeypatch) -> None:
    import sys
    import types

    import egologqa.ai_summary as ai_summary

    captured: dict[str, object] = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            captured["config_kwargs"] = kwargs

    class FakeThinkingConfig:
        def __init__(self, **kwargs) -> None:
            captured["thinking_kwargs"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["generate_kwargs"] = kwargs

            class FakeResponse:
                text = '{"summary_line":"ok"}'

            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        ThinkingConfig=FakeThinkingConfig,
    )
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    payload = ai_summary._request_gemini_json_payload(
        context={"gate": {"gate": "PASS"}},
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    assert json.loads(payload)["summary_line"] == "ok"
    assert captured["api_key"] == "test-key"
    assert captured["thinking_kwargs"] == {"thinking_budget": 0}

    config_kwargs = captured["config_kwargs"]
    assert isinstance(config_kwargs, dict)
    assert config_kwargs["response_mime_type"] == "application/json"

    schema = config_kwargs["response_schema"]
    assert isinstance(schema, dict)
    assert "additionalProperties" not in schema


def test_streamlit_quick_summary_keeps_fallback_payload_wiring() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "ai_summary = generate_summary_for_ui(" in source
    assert "except Exception:\n            ai_summary = None" not in source


def test_streamlit_quick_summary_has_advanced_debug_caption() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert 'if ADVANCED_MODE and source_label == "Deterministic fallback" and summary_error_code:' in source
    assert "Debug:" in source
