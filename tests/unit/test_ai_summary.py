from __future__ import annotations

import json
from pathlib import Path

from egologqa.ai_summary import (
    DEFAULT_GEMINI_MODEL,
    build_curated_context,
    deterministic_action_line,
    generate_summary_for_ui,
    resolve_gemini_api_key,
    resolve_gemini_model_name,
)


def _report_template() -> dict:
    return {
        "gate": {
            "gate": "WARN",
            "recommended_action": "USE_SEGMENTS_ONLY",
            "fail_reasons": [],
            "warn_reasons": [
                "WARN_SYNC_P95_GT_WARN",
                "WARN_SYNC_JITTER_P95_GT_WARN",
            ],
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


def test_resolve_gemini_api_key_env_precedence(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert resolve_gemini_api_key({"GOOGLE_API_KEY": "secret-google"}) == "google-key"


def test_resolve_gemini_api_key_uses_secrets_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_gemini_api_key({"GEMINI_API_KEY": "secret-gemini"}) == "secret-gemini"


def test_resolve_gemini_model_name_explicit_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("EGOLOGQA_GEMINI_MODEL", "env-model")
    assert resolve_gemini_model_name("explicit-model") == "explicit-model"


def test_resolve_gemini_model_name_uses_env_when_no_override(monkeypatch) -> None:
    monkeypatch.setenv("EGOLOGQA_GEMINI_MODEL", "env-model")
    assert resolve_gemini_model_name(None) == "env-model"


def test_resolve_gemini_model_name_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("EGOLOGQA_GEMINI_MODEL", raising=False)
    assert resolve_gemini_model_name(None) == DEFAULT_GEMINI_MODEL


def test_build_curated_context_includes_project_brief_and_full_report(tmp_path: Path) -> None:
    report = _report_template()

    context = build_curated_context(report, tmp_path)

    assert "project_brief" in context
    assert "report" in context
    assert context["report"] == report
    assert "output_contract" in context
    assert "explanation_line" in json.dumps(context["output_contract"])
    assert "insight_line" in json.dumps(context["output_contract"])


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
        model=None,
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_API_KEY_MISSING"
    assert out["action_line"] == "Action: Use clean segments only."
    assert isinstance(out.get("summary_line"), str) and out["summary_line"].strip()
    assert isinstance(out.get("explanation_line"), str) and out["explanation_line"].strip()
    assert isinstance(out.get("insight_line"), str) and out["insight_line"].strip()


def test_generate_summary_sends_full_report_context(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    captured: dict[str, object] = {}

    def _fake_request(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "summary_line": "Warnings were detected.",
                "explanation_line": "Timing alignment variation drove the warning decision.",
                "insight_line": "Jitter remained elevated in sampled sync windows.",
            }
        )

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    monkeypatch.setenv("EGOLOGQA_GEMINI_MODEL", "env-model")
    monkeypatch.setattr(ai_summary, "_request_gemini_json_payload", _fake_request)

    report = _report_template()
    out = ai_summary.generate_summary_for_ui(
        report=report,
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model=None,
    )

    assert out["source"] == "gemini"
    assert captured["model"] == "env-model"
    context = captured["context"]
    assert isinstance(context, dict)
    assert context["report"] == report
    assert "project_brief" in context


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
    assert out["explanation_line"]
    assert out["insight_line"]


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


def test_generate_summary_schema_missing_insight_falls_back(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    monkeypatch.setattr(
        ai_summary,
        "_request_gemini_json_payload",
        lambda **_kwargs: json.dumps(
            {
                "summary_line": "Only one field",
                "explanation_line": "Missing insight",
            }
        ),
    )

    out = ai_summary.generate_summary_for_ui(
        report=_report_template(),
        output_dir=tmp_path,
        secrets=None,
        enabled=True,
        model="gemini-2.5-flash",
    )

    assert out["source"] == "fallback"
    assert out["error_code"] == "AI_SUMMARY_SCHEMA_INVALID"


def test_generate_summary_normalizes_long_multiline_response(monkeypatch, tmp_path: Path) -> None:
    import egologqa.ai_summary as ai_summary

    monkeypatch.setenv("GOOGLE_API_KEY", "x-key")
    long_summary = "Summary line\n" + ("very long detail " * 20)
    long_explanation = "Explanation line\n" + ("secondary detail " * 30)
    long_insight = "Insight line\n" + ("extra signal detail " * 30)
    monkeypatch.setattr(
        ai_summary,
        "_request_gemini_json_payload",
        lambda **_kwargs: json.dumps(
            {
                "summary_line": long_summary,
                "explanation_line": long_explanation,
                "insight_line": long_insight,
            }
        ),
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
    assert "\n" not in out["explanation_line"]
    assert "\n" not in out["insight_line"]
    assert len(out["summary_line"]) <= 150
    assert len(out["explanation_line"]) <= 220
    assert len(out["insight_line"]) <= 220
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


def test_request_gemini_payload_schema_includes_three_required_ai_lines(monkeypatch) -> None:
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
                text = '{"summary_line":"ok","explanation_line":"because","insight_line":"note"}'

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
        context={"report": {"gate": {"gate": "PASS"}}},
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    decoded = json.loads(payload)
    assert decoded["summary_line"] == "ok"
    assert decoded["explanation_line"] == "because"
    assert decoded["insight_line"] == "note"
    assert captured["api_key"] == "test-key"
    assert captured["thinking_kwargs"] == {"thinking_budget": 0}

    config_kwargs = captured["config_kwargs"]
    assert isinstance(config_kwargs, dict)
    assert config_kwargs["response_mime_type"] == "application/json"
    assert config_kwargs["max_output_tokens"] == 260

    schema = config_kwargs["response_schema"]
    assert isinstance(schema, dict)
    assert set(schema["required"]) == {"summary_line", "explanation_line", "insight_line"}
    assert "action_line" not in schema["properties"]
    assert "additionalProperties" not in schema


def test_streamlit_quick_summary_keeps_fallback_payload_wiring() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "ai_summary = generate_summary_for_ui(" in source
    assert "except Exception:\n            ai_summary = None" not in source


def test_streamlit_quick_summary_renders_three_ai_lines_and_debug_caption() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "explanation_line" in source
    assert "insight_line" in source
    assert 'if ADVANCED_MODE and source_label == "Deterministic fallback" and summary_error_code:' in source
    assert "Debug:" in source


def test_streamlit_uses_model_override_without_local_default_literal() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert 'AI_SUMMARY_MODEL_OVERRIDE = os.getenv("EGOLOGQA_GEMINI_MODEL")' in source
    assert 'AI_SUMMARY_MODEL = os.getenv("EGOLOGQA_GEMINI_MODEL", "gemini-2.5-flash")' not in source
