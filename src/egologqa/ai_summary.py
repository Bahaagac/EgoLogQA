from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_MAX_SUMMARY_CHARS = 150
_MAX_EXPLANATION_CHARS = 220
_MAX_INSIGHT_CHARS = 220

PROJECT_BRIEF = (
    "EgoLogQA is a deterministic quality-control analyzer for MicroAGI00-style ROS2 MCAP logs. "
    "It checks timestamp integrity, RGB-depth sync, drop ratio, IMU coverage, RGB blur and exposure, "
    "depth validity, and segment quality before recommending recording usage."
)

_REASON_HINTS = {
    "FAIL_NO_RGB_STREAM": "RGB stream is missing, so visual quality cannot be validated.",
    "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH": "No sufficiently long clean segment was found for reliable usage.",
    "FAIL_SYNC_P95_GT_FAIL": "RGB-depth timing misalignment is severe.",
    "FAIL_DROP_RATIO_GT_FAIL": "RGB frame-drop rate is too high for reliable continuity.",
    "FAIL_DEPTH_FAIL_RATIO_GT_FAIL": "Depth failures are too frequent.",
    "FAIL_DEPTH_INVALID_MEAN_GT_FAIL": "Depth invalid pixels are consistently too high.",
    "WARN_SYNC_P95_GT_WARN": "RGB-depth alignment is unstable.",
    "WARN_SYNC_JITTER_P95_GT_WARN": "Sync jitter is elevated, so alignment fluctuates between frames.",
    "WARN_SYNC_DRIFT_ABS_GT_WARN": "Sync drift indicates alignment changes over time.",
    "WARN_DROP_RATIO_GT_WARN": "RGB drops are above the warning threshold.",
    "WARN_IMU_MISSING_RATIO_GT_WARN": "IMU coverage is incomplete.",
    "WARN_BLUR_FAIL_RATIO_GT_WARN": "A noticeable portion of frames are blur-failed.",
    "WARN_EXPOSURE_BAD_RATIO_GT_WARN": "Exposure quality is unstable in sampled frames.",
    "WARN_DEPTH_INVALID_MEAN_GT_WARN": "Depth validity is degraded by invalid pixels.",
    "WARN_DEPTH_TIMESTAMP_MISSING": "Depth timestamp availability is limited, reducing sync confidence.",
    "WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED": "Depth pixel decode is unavailable in this runtime.",
    "WARN_RGB_PIXEL_DECODE_UNSUPPORTED": "RGB pixel decode is unavailable in this runtime.",
}


def resolve_gemini_api_key(secrets: Any | None) -> str | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        env_value = os.getenv(name)
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()

    if secrets is None:
        return None

    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            secret_value = secrets.get(name)
        except Exception:
            secret_value = None
        if isinstance(secret_value, str) and secret_value.strip():
            return secret_value.strip()
    return None


def resolve_gemini_model_name(model_override: str | None) -> str:
    if isinstance(model_override, str) and model_override.strip():
        return model_override.strip()

    env_value = os.getenv("EGOLOGQA_GEMINI_MODEL")
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()

    return DEFAULT_GEMINI_MODEL


def build_curated_context(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    del output_dir
    report_payload = report if isinstance(report, dict) else {}
    return {
        "project_brief": PROJECT_BRIEF,
        "output_contract": {
            "summary_line": "One short sentence: final quality outcome in plain language.",
            "explanation_line": "One short sentence: primary cause behind the gate outcome.",
            "insight_line": "One short sentence: secondary or unusual signal from the run.",
            "format_rules": [
                "No enum codes.",
                "No file paths.",
                "No action sentence.",
                "No metric dump.",
            ],
        },
        "report": report_payload,
    }


def deterministic_action_line(action_token: str) -> str:
    mapping = {
        "USE_FULL_SEQUENCE": "Action: Use the full sequence.",
        "USE_SEGMENTS_ONLY": "Action: Use clean segments only.",
        "FIX_TIME_ALIGNMENT": "Action: Fix RGB-depth alignment, then re-run.",
        "RECAPTURE_OR_SKIP": "Action: Recapture or skip this recording.",
    }
    return mapping.get(
        str(action_token or ""),
        "Action: Review quality sections and use conservative handling.",
    )


def fallback_summary(report: dict[str, Any]) -> dict[str, str]:
    gate = report.get("gate", {})
    gate = gate if isinstance(gate, dict) else {}
    gate_name = str(gate.get("gate") or "").upper()
    reason_codes = _reason_codes_from_gate(gate)

    if gate_name == "PASS":
        summary = "Quality checks passed and the recording is usable as a whole sequence."
    elif gate_name == "WARN":
        summary = "Quality checks raised warnings, so selective usage is safer than full-sequence use."
    elif gate_name == "FAIL":
        summary = "Critical quality checks failed, so this recording is not reliable as-is."
    else:
        summary = "Quality outcome is unavailable due to missing analysis context."

    explanation = _fallback_explanation(gate_name, reason_codes)
    insight = _fallback_insight(gate_name, reason_codes)

    return {
        "summary_line": _normalize_line(summary, _MAX_SUMMARY_CHARS),
        "explanation_line": _normalize_line(explanation, _MAX_EXPLANATION_CHARS),
        "insight_line": _normalize_line(insight, _MAX_INSIGHT_CHARS),
        "action_line": deterministic_action_line(str(gate.get("recommended_action") or "")),
    }


def generate_summary_for_ui(
    report: dict[str, Any],
    output_dir: Path,
    secrets: Any | None,
    enabled: bool,
    model: str | None,
) -> dict[str, Any]:
    selected_model = resolve_gemini_model_name(model)
    fallback = fallback_summary(report)
    result: dict[str, Any] = {
        "summary_line": fallback["summary_line"],
        "explanation_line": fallback["explanation_line"],
        "insight_line": fallback["insight_line"],
        "action_line": fallback["action_line"],
        "source": "fallback",
        "model": selected_model,
    }

    if not enabled:
        result["error_code"] = "AI_SUMMARY_DISABLED"
        return result

    api_key = resolve_gemini_api_key(secrets)
    if not api_key:
        result["error_code"] = "AI_SUMMARY_API_KEY_MISSING"
        return result

    context = build_curated_context(report, output_dir)

    try:
        response_text = _request_gemini_json_payload(
            context=context,
            api_key=api_key,
            model=selected_model,
        )
    except RuntimeError as exc:
        code = str(exc.args[0]) if exc.args else "AI_SUMMARY_REQUEST_FAILED"
        if not code.startswith("AI_SUMMARY_"):
            code = "AI_SUMMARY_REQUEST_FAILED"
        result["error_code"] = code
        return result
    except Exception:
        result["error_code"] = "AI_SUMMARY_REQUEST_FAILED"
        return result

    try:
        payload = json.loads(response_text)
    except Exception:
        result["error_code"] = "AI_SUMMARY_JSON_PARSE_FAILED"
        return result

    if not isinstance(payload, dict):
        result["error_code"] = "AI_SUMMARY_SCHEMA_INVALID"
        return result

    summary_raw = payload.get("summary_line")
    explanation_raw = payload.get("explanation_line")
    insight_raw = payload.get("insight_line")

    if not isinstance(summary_raw, str) or not summary_raw.strip():
        result["error_code"] = "AI_SUMMARY_SCHEMA_INVALID"
        return result
    if not isinstance(explanation_raw, str) or not explanation_raw.strip():
        result["error_code"] = "AI_SUMMARY_SCHEMA_INVALID"
        return result
    if not isinstance(insight_raw, str) or not insight_raw.strip():
        result["error_code"] = "AI_SUMMARY_SCHEMA_INVALID"
        return result

    summary_line = _normalize_line(summary_raw, _MAX_SUMMARY_CHARS)
    explanation_line = _normalize_line(explanation_raw, _MAX_EXPLANATION_CHARS)
    insight_line = _normalize_line(insight_raw, _MAX_INSIGHT_CHARS)

    if not summary_line or not explanation_line or not insight_line:
        result["error_code"] = "AI_SUMMARY_EMPTY"
        return result

    result["summary_line"] = summary_line
    result["explanation_line"] = explanation_line
    result["insight_line"] = insight_line
    result["source"] = "gemini"
    result.pop("error_code", None)
    return result


def _request_gemini_json_payload(context: dict[str, Any], api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("AI_SUMMARY_SDK_IMPORT_FAILED") from exc

    client = genai.Client(api_key=api_key)
    prompt = (
        "Summarize this EgoLogQA analysis for an operator.\\n"
        "Return JSON with exactly three keys: summary_line, explanation_line, insight_line.\\n"
        "Constraints:\\n"
        "- summary_line: one short sentence, outcome first.\\n"
        "- explanation_line: one short sentence with the primary cause.\\n"
        "- insight_line: one short sentence with secondary or unusual signal.\\n"
        "- plain language only, no enum names, no file paths, no action sentence, no metric dump.\\n"
        f"Context JSON: {json.dumps(context, sort_keys=True, separators=(',', ':'))}"
    )

    response = client.models.generate_content(
        model=resolve_gemini_model_name(model),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You explain robot-log quality outcomes for operators. "
                "Be concise, factual, and non-redundant."
            ),
            temperature=0.1,
            max_output_tokens=260,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "summary_line": {
                        "type": "string",
                        "description": "Short plain-language outcome summary.",
                    },
                    "explanation_line": {
                        "type": "string",
                        "description": "Short primary-cause explanation without action text.",
                    },
                    "insight_line": {
                        "type": "string",
                        "description": "Short secondary/unusual-signal note without action text.",
                    },
                },
                "required": ["summary_line", "explanation_line", "insight_line"],
            },
        ),
    )

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text

    raise RuntimeError("AI_SUMMARY_EMPTY_RESPONSE")


def _normalize_line(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _reason_codes_from_gate(gate: dict[str, Any]) -> list[str]:
    fail_reasons = gate.get("fail_reasons") if isinstance(gate.get("fail_reasons"), list) else []
    warn_reasons = gate.get("warn_reasons") if isinstance(gate.get("warn_reasons"), list) else []
    ordered = fail_reasons if fail_reasons else warn_reasons
    return [str(code) for code in ordered if code]


def _fallback_explanation(gate_name: str, reason_codes: list[str]) -> str:
    if reason_codes:
        return _reason_hint_for_code(reason_codes[0])

    if gate_name == "PASS":
        return "No primary anomalies were detected by integrity, timing, or pixel-quality checks."
    return "A primary cause could not be extracted from the current gate reason payload."


def _fallback_insight(gate_name: str, reason_codes: list[str]) -> str:
    if len(reason_codes) >= 2:
        insight = _reason_hint_for_code(reason_codes[1])
        if len(reason_codes) > 2:
            insight += " Additional checks also triggered."
        return insight

    if len(reason_codes) == 1:
        return "No secondary anomaly was reported beyond the primary flagged check."

    if gate_name == "PASS":
        return "No unusual warning or fail reasons were reported for this run."
    return "No secondary anomaly detail is available in this analysis output."


def _reason_hint_for_code(code: str) -> str:
    direct = _REASON_HINTS.get(code)
    if direct:
        return direct

    upper_code = code.upper()
    if "SYNC" in upper_code:
        return "RGB-depth timing alignment is unstable."
    if "DROP_RATIO" in upper_code:
        return "Frame-drop behavior is above target thresholds."
    if "IMU" in upper_code:
        return "IMU coverage or consistency is below expectations."
    if "BLUR" in upper_code:
        return "Image sharpness is below target on many sampled frames."
    if "EXPOSURE" in upper_code:
        return "Exposure quality indicates unstable lighting conditions."
    if "DEPTH" in upper_code:
        return "Depth quality checks indicate unreliable depth values."
    if "RGB" in upper_code:
        return "RGB stream quality is degraded in a gating check."
    return "A quality gate reason triggered outside normal operating thresholds."
