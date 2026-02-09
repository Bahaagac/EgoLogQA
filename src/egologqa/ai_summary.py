from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


_DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_SUMMARY_CHARS = 120
_KEY_METRIC_KEYS = [
    "file_total_messages",
    "file_duration_s",
    "file_bitrate_mbps",
    "sync_p95_ms",
    "sync_jitter_p95_ms",
    "sync_drift_ms_per_min",
    "sync_sample_count",
    "drop_ratio",
    "integrity_ok_ratio",
    "integrity_coverage_seconds_est",
    "vision_ok_ratio",
    "vision_coverage_seconds_est",
    "blur_fail_ratio",
    "exposure_bad_ratio",
    "depth_fail_ratio",
    "depth_invalid_mean",
    "rgb_decode_attempt_count",
    "rgb_decode_success_count",
    "depth_decode_attempt_count",
    "depth_decode_success_count",
    "blur_valid_frame_count",
    "exposure_valid_frame_count",
    "depth_valid_frame_count",
]


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


def build_curated_context(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    gate = report.get("gate", {})
    gate = gate if isinstance(gate, dict) else {}
    streams = report.get("streams", {})
    streams = streams if isinstance(streams, dict) else {}
    metrics = report.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    errors = report.get("errors", [])
    errors = errors if isinstance(errors, list) else []

    integrity_segments = report.get("segments", [])
    integrity_segments = integrity_segments if isinstance(integrity_segments, list) else []
    clean_segments = _load_segments_from_metric(metrics, output_dir, "clean_segments_path")
    clean_segments_nosync = _load_segments_from_metric(metrics, output_dir, "clean_segments_nosync_path")

    key_metrics: dict[str, Any] = {}
    for key in _KEY_METRIC_KEYS:
        key_metrics[key] = metrics.get(key)

    if isinstance(metrics.get("out_of_order"), dict):
        key_metrics["out_of_order"] = metrics.get("out_of_order")
    if isinstance(metrics.get("exposure_bad_reason_counts"), dict):
        key_metrics["exposure_bad_reason_counts"] = metrics.get("exposure_bad_reason_counts")

    return {
        "system_context": {
            "name": "EgoLogQA",
            "purpose": "Quality gate and segment extractor for ROS2 MCAP logs.",
            "checks": [
                "timestamp integrity",
                "rgb-depth sync and drop gaps",
                "imu coverage",
                "rgb blur and exposure",
                "depth validity",
                "integrity and clean segment extraction",
            ],
        },
        "gate": {
            "gate": gate.get("gate"),
            "recommended_action": gate.get("recommended_action"),
            "fail_reasons": gate.get("fail_reasons", []),
            "warn_reasons": gate.get("warn_reasons", []),
        },
        "segments": {
            "integrity": _segment_stats(integrity_segments),
            "clean": _segment_stats(clean_segments),
            "clean_nosync": _segment_stats(clean_segments_nosync),
        },
        "stream_status": {
            "rgb_timestamps_present": streams.get("rgb_timestamps_present"),
            "depth_topic_present": streams.get("depth_topic_present"),
            "depth_timestamps_present": streams.get("depth_timestamps_present"),
            "decode_status": streams.get("decode_status"),
            "topics": {
                "rgb_topic": streams.get("rgb_topic"),
                "depth_topic": streams.get("depth_topic"),
                "imu_accel_topic": streams.get("imu_accel_topic"),
                "imu_gyro_topic": streams.get("imu_gyro_topic"),
            },
        },
        "metrics": key_metrics,
        "error_counts": _error_counts(errors),
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

    if gate_name == "PASS":
        finding = "Quality checks passed with no warning or fail reasons."
    elif gate_name == "WARN":
        finding = "Quality checks found non-critical issues, so selective usage is safer."
    elif gate_name == "FAIL":
        finding = "Critical quality checks failed, so this recording is not reliable as-is."
    else:
        finding = "Quality outcome is unavailable due to missing analysis context."

    return {
        "summary_line": _normalize_line(finding, _MAX_SUMMARY_CHARS),
        "action_line": deterministic_action_line(str(gate.get("recommended_action") or "")),
    }


def generate_summary_for_ui(
    report: dict[str, Any],
    output_dir: Path,
    secrets: Any | None,
    enabled: bool,
    model: str,
) -> dict[str, Any]:
    selected_model = _normalize_model_name(model)
    fallback = fallback_summary(report)
    result: dict[str, Any] = {
        "summary_line": fallback["summary_line"],
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
    if not isinstance(summary_raw, str) or not summary_raw.strip():
        result["error_code"] = "AI_SUMMARY_SCHEMA_INVALID"
        return result

    summary_line = _normalize_line(summary_raw, _MAX_SUMMARY_CHARS)
    if not summary_line:
        result["error_code"] = "AI_SUMMARY_EMPTY"
        return result

    result["summary_line"] = summary_line
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
        "Summarize this EgoLogQA run in exactly one short sentence.\n"
        "Rules: plain language, no enum codes, no file paths, no metric dump.\n"
        f"Context JSON: {json.dumps(context, sort_keys=True, separators=(',', ':'))}"
    )

    response = client.models.generate_content(
        model=_normalize_model_name(model),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You summarize robot-log quality checks for operators. "
                "Return concise, actionable wording only."
            ),
            temperature=0.1,
            max_output_tokens=80,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "summary_line": {
                        "type": "string",
                        "description": "Single short sentence on the most important run outcome.",
                    }
                },
                "required": ["summary_line"],
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


def _normalize_model_name(model: str) -> str:
    value = str(model or "").strip()
    return value if value else _DEFAULT_MODEL


def _normalize_line(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _segment_stats(segments: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    total_duration_s = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        count += 1
        duration = _as_float(segment.get("duration_s"))
        if duration is not None and duration > 0.0:
            total_duration_s += duration
    return {
        "count": count,
        "total_duration_s": round(total_duration_s, 3),
    }


def _load_segments_from_metric(
    metrics: dict[str, Any], output_dir: Path, key: str
) -> list[dict[str, Any]]:
    relpath = metrics.get(key)
    if not relpath:
        return []
    path = output_dir / str(relpath)
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _error_counts(errors: list[Any]) -> dict[str, list[dict[str, Any]]]:
    warn_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for item in errors:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "UNKNOWN")
        severity = str(item.get("severity") or "").upper()
        if severity == "WARN":
            warn_counts[code] += 1
        elif severity == "ERROR":
            error_counts[code] += 1

    def _rows(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"code": code, "count": count}
            for code, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    return {
        "warn_codes": _rows(warn_counts),
        "error_codes": _rows(error_counts),
    }
