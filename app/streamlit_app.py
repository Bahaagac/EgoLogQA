from __future__ import annotations

import html
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from egologqa.config import load_config
from egologqa.io.hf_fetch import list_mcap_files, resolve_cached_file
from egologqa.io.local_fs import (
    LocalDirNotFound,
    LocalDirNotReadable,
    TooManyFiles,
    is_readable_file,
    list_mcap_files_in_dir,
)
from egologqa.kiosk_helpers import (
    allocate_run_dir,
    build_hf_display_label,
    build_timestamped_run_basename,
    ensure_writable_dir,
    make_local_option_label,
    map_error_bucket,
    resolve_runs_base_dir,
    write_latest_run_pointer,
)
from egologqa.pipeline import analyze_file
from egologqa.ui_text import recommended_action_copy


HF_REPO_ID = os.getenv("EGOLOGQA_HF_REPO_ID", "MicroAGI-Labs/MicroAGI00")
HF_REVISION = os.getenv("EGOLOGQA_HF_REVISION", "main")
HF_PREFIX = os.getenv("EGOLOGQA_HF_PREFIX", "raw_mcaps/")
HF_CACHE_DIR = Path(os.getenv("EGOLOGQA_HF_CACHE_DIR", "~/.cache/egologqa/hf_mcaps")).expanduser()
RUNS_BASE_DIR = resolve_runs_base_dir(os.getenv("EGOLOGQA_RUNS_DIR"))
CONFIG_PATH = "configs/microagi00_ros2.yaml"
ADVANCED_MODE = os.getenv("EGOLOGQA_UI_ADVANCED", "0") == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


LOCAL_MAX_FILES = _env_int("EGOLOGQA_LOCAL_MAX_FILES", 500)


st.set_page_config(page_title="EgoLogQA", layout="wide")
st.title("EgoLogQA")
st.caption("MicroAGI00 ROS2 MCAP quality assessment")
st.write("Choose an MCAP file to analyze.")
st.markdown(
    """
<style>
.egologqa-help {
    margin-top: -0.55rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    opacity: 0.88;
}
</style>
""",
    unsafe_allow_html=True,
)


def _read_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        secret_val = st.secrets.get("HF_TOKEN")
    except Exception:
        return None
    if isinstance(secret_val, str) and secret_val.strip():
        return secret_val.strip()
    return None


def _has_hf_dependency() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False, ttl=300)
def _cached_hf_file_list(
    repo_id: str,
    revision: str,
    prefix: str,
    token_digest: str,
    _token: str | None,
) -> list[dict[str, Any]]:
    del token_digest
    return list_mcap_files(
        repo_id=repo_id,
        revision=revision,
        token=_token,
        prefix=prefix,
    )


@st.cache_data(show_spinner=False, ttl=300)
def _cached_local_file_list(dir_path: str, nonce: int, max_files: int) -> list[dict[str, Any]]:
    del nonce
    return list_mcap_files_in_dir(dir_path=dir_path, max_files=max_files)


def _error_box(exc: Exception, default_msg: str) -> None:
    mapped = map_error_bucket(exc)
    st.error(mapped if mapped else default_msg)
    if ADVANCED_MODE:
        with st.expander("Error details", expanded=False):
            st.code(f"{exc.__class__.__name__}: {exc}")


def _phase_label(phase: str) -> str:
    mapping = {
        "scan": "scan",
        "pass1": "pass1",
        "pass2": "pass2",
        "done": "finalize",
        "error": "error",
    }
    return mapping.get(phase, phase)


def _render_gate_summary(report: dict[str, Any]) -> None:
    gate = report.get("gate", {})
    gate_name = gate.get("gate")
    display_gate = {"PASS": "PASS", "WARN": "WARNING", "FAIL": "FAIL"}.get(
        str(gate_name), str(gate_name)
    )
    fail_reasons = gate.get("fail_reasons", [])
    warn_reasons = gate.get("warn_reasons", [])
    if gate_name == "PASS":
        st.success(f"Overall Result: {display_gate}")
    elif gate_name == "WARN":
        st.warning(f"Overall Result: {display_gate}")
    else:
        st.error(f"Overall Result: {display_gate}")
    action_token = str(gate.get("recommended_action") or "UNKNOWN")
    action_copy = recommended_action_copy(
        action_token=action_token,
        gate=str(gate_name),
        fail_reasons=fail_reasons,
        warn_reasons=warn_reasons,
    )
    st.markdown("**Recommended action**")
    st.caption(f"Token: `{action_token}`")
    st.write(f"What to do now: {action_copy['what_to_do']}")
    st.caption(f"Why: {action_copy['why']}")
    if gate_name == "PASS":
        return

    if gate_name == "FAIL" and fail_reasons:
        _render_reason_table("Failure Reasons", "FAIL", fail_reasons, report)
    if gate_name in {"WARN", "FAIL"} and warn_reasons:
        _render_reason_table("Warning Reasons", "WARNING", warn_reasons, report)


def _on_progress_scaled(start: float, end: float):
    span = max(0.0, end - start)

    def _cb(event: dict) -> None:
        frac = min(max(float(event.get("progress", 0.0)), 0.0), 1.0)
        progress.progress(start + span * frac)
        phase = _phase_label(str(event.get("phase", "analysis")))
        _update_status(f"{phase}: {event.get('message', '...')}")

    return _cb


def _local_folder_presets(last_used: str | None) -> list[tuple[str, str]]:
    home = Path.home()
    presets: list[tuple[str, str]] = [
        ("~/Downloads", str(home / "Downloads")),
        ("~/Desktop", str(home / "Desktop")),
        ("~/.cache/egologqa", str(home / ".cache" / "egologqa")),
    ]
    if last_used and all(last_used != value for _, value in presets):
        presets.append(("Last used", last_used))
    presets.append(("Other...", "__other__"))
    return presets


def _normalized_local_dir_key(dir_path: str) -> str:
    expanded = os.path.expandvars(dir_path)
    return str(Path(expanded).expanduser().resolve())


def _resolve_input_to_local_path(
    source_mode: str,
    selected_local_path: str | None,
    selected_hf_path: str | None,
    hf_token: str | None,
) -> tuple[Path, float]:
    if source_mode == "Local disk":
        if selected_local_path is None:
            raise RuntimeError("No local file selected.")
        ok, reason = is_readable_file(selected_local_path)
        if not ok:
            raise RuntimeError(reason or "Selected local file is not readable.")
        return Path(selected_local_path), 0.05

    if selected_hf_path is None:
        raise RuntimeError("No Hugging Face file selected.")

    ensure_writable_dir(HF_CACHE_DIR, "HF cache directory")

    def _on_resolve(downloaded: int, total: int | None) -> None:
        if total is not None and total > 0:
            frac = min(max(downloaded / total, 0.0), 1.0)
            progress.progress(0.2 * frac)
            _update_status(
                f"download: {downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB"
            )
        else:
            progress.progress(0.2 if downloaded > 0 else 0.05)
            _update_status("cache: checking/downloading dataset file")

    path = resolve_cached_file(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        file_path=str(selected_hf_path),
        token=hf_token,
        cache_dir=HF_CACHE_DIR,
        progress_cb=_on_resolve,
    )
    return path, 0.2


def _render_metrics_table(
    title: str,
    metrics: dict[str, Any],
    keys: list[str],
    explanation: str,
    hide_none: bool = False,
) -> None:
    rows = [{"metric": key, "value": metrics.get(key)} for key in keys]
    if hide_none:
        rows = [row for row in rows if row["value"] is not None]
    _render_header(title, explanation, level=3)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No diagnostics available.")


def _show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists() and path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)


def _render_header(
    title: str,
    explanation: str,
    level: int = 2,
) -> None:
    if level == 2:
        st.header(title, anchor=False)
    else:
        st.subheader(title, anchor=False)
    escaped_explanation = html.escape(explanation, quote=True)
    st.markdown(
        f'<p class="egologqa-help">ℹ {escaped_explanation}</p>',
        unsafe_allow_html=True,
    )


def _load_segments_from_metric(metrics: dict[str, Any], output_dir: Path, key: str) -> list[dict[str, Any]]:
    rel = metrics.get(key)
    if not rel:
        return []
    path = output_dir / str(rel)
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _show_image_set(paths: list[Path], max_images: int | None = 12) -> None:
    images = sorted(paths, key=lambda p: p.name.lower())
    if max_images is not None:
        images = images[:max_images]
    if not images:
        return
    st.image([str(p) for p in images], caption=[p.name for p in images], width=220)
    st.caption("Filenames: " + ", ".join(p.name for p in images))


def _normalize_segment_rows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        rows.append(
            {
                "start_ns": seg.get("start_ns"),
                "end_ns": seg.get("end_ns"),
                "duration_s": seg.get("duration_s"),
            }
        )
    return rows


REASON_DESCRIPTIONS: dict[str, str] = {
    "FAIL_ANALYSIS_ERROR": "The analysis process failed internally and the result is not reliable.",
    "FAIL_NO_RGB_STREAM": "RGB timestamps are missing, so the run cannot be evaluated correctly.",
    "FAIL_SYNC_P95_GT_FAIL": "RGB and depth timing mismatch is too large for safe use.",
    "FAIL_DROP_RATIO_GT_FAIL": "Too many timing gaps were detected in the RGB stream.",
    "FAIL_DEPTH_FAIL_RATIO_GT_FAIL": "Too many sampled depth frames failed quality checks.",
    "FAIL_DEPTH_INVALID_MEAN_GT_FAIL": "Average invalid depth-pixel ratio is too high.",
    "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH": "No clean segment is long enough for reliable use.",
    "WARN_DEPTH_TIMESTAMP_MISSING": "Depth timestamps are missing for part of this run.",
    "WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED": "Depth pixel format could not be decoded for this run.",
    "WARN_RGB_PIXEL_DECODE_UNSUPPORTED": "RGB pixel format could not be decoded for this run.",
    "WARN_SYNC_P95_GT_WARN": "Timing mismatch is above warning level.",
    "WARN_SYNC_JITTER_P95_GT_WARN": "Timing jitter is above warning level.",
    "WARN_SYNC_DRIFT_ABS_GT_WARN": "Timing drift is above warning level.",
    "WARN_DROP_RATIO_GT_WARN": "Frame timing gaps are higher than expected.",
    "WARN_IMU_MISSING_RATIO_GT_WARN": "IMU coverage is lower than expected.",
    "WARN_BLUR_FAIL_RATIO_GT_WARN": "Many sampled frames are blur-fail.",
    "WARN_EXPOSURE_BAD_RATIO_GT_WARN": "Many sampled frames are exposure-fail.",
    "WARN_DEPTH_INVALID_MEAN_GT_WARN": "Average invalid depth-pixel ratio is above warning level.",
}


def _format_reason_context(code: str, report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    streams = report.get("streams", {})
    thresholds = report.get("config_used", {}).get("thresholds", {})
    errors = report.get("errors", [])

    if code == "FAIL_ANALYSIS_ERROR":
        for err in errors:
            if err.get("severity") == "ERROR":
                return f"Observed error code: {err.get('code', 'unknown')}"
        return "Observed an internal analysis error."
    if code == "FAIL_NO_RGB_STREAM":
        return f"rgb_timestamps_present={streams.get('rgb_timestamps_present')}"
    if code in {"FAIL_SYNC_P95_GT_FAIL", "WARN_SYNC_P95_GT_WARN"}:
        return (
            f"sync_p95_ms={metrics.get('sync_p95_ms')} "
            f"(warn={thresholds.get('sync_warn_ms')}, fail={thresholds.get('sync_fail_ms')})"
        )
    if code == "WARN_SYNC_JITTER_P95_GT_WARN":
        return (
            f"sync_jitter_p95_ms={metrics.get('sync_jitter_p95_ms')} "
            f"(warn={thresholds.get('sync_jitter_warn_ms')})"
        )
    if code == "WARN_SYNC_DRIFT_ABS_GT_WARN":
        return (
            f"sync_drift_ms_per_min={metrics.get('sync_drift_ms_per_min')} "
            f"(abs warn={thresholds.get('sync_drift_warn_ms_per_min')})"
        )
    if code == "FAIL_DROP_RATIO_GT_FAIL":
        return f"drop_ratio={metrics.get('drop_ratio')} (fail={thresholds.get('drop_fail_ratio')})"
    if code == "WARN_DROP_RATIO_GT_WARN":
        return f"drop_ratio={metrics.get('drop_ratio')} (warn={thresholds.get('drop_warn_ratio')})"
    if code == "WARN_IMU_MISSING_RATIO_GT_WARN":
        return (
            "imu_combined_missing_ratio="
            f"{metrics.get('imu_combined_missing_ratio')} "
            f"(warn={thresholds.get('imu_missing_warn_ratio')})"
        )
    if code == "WARN_BLUR_FAIL_RATIO_GT_WARN":
        return (
            f"blur_fail_ratio={metrics.get('blur_fail_ratio')} "
            f"(warn={thresholds.get('blur_fail_warn_ratio')})"
        )
    if code == "WARN_EXPOSURE_BAD_RATIO_GT_WARN":
        return (
            f"exposure_bad_ratio={metrics.get('exposure_bad_ratio')} "
            f"(warn={thresholds.get('exposure_bad_warn_ratio')})"
        )
    if code == "FAIL_DEPTH_FAIL_RATIO_GT_FAIL":
        return (
            f"depth_fail_ratio={metrics.get('depth_fail_ratio')} "
            f"(fail={thresholds.get('depth_fail_ratio_fail')})"
        )
    if code == "FAIL_DEPTH_INVALID_MEAN_GT_FAIL":
        return (
            f"depth_invalid_mean={metrics.get('depth_invalid_mean')} "
            f"(fail={thresholds.get('depth_invalid_mean_fail')})"
        )
    if code == "WARN_DEPTH_INVALID_MEAN_GT_WARN":
        return (
            f"depth_invalid_mean={metrics.get('depth_invalid_mean')} "
            f"(warn={thresholds.get('depth_invalid_mean_warn')})"
        )
    if code == "WARN_DEPTH_TIMESTAMP_MISSING":
        return f"depth_timestamps_present={streams.get('depth_timestamps_present')}"
    if code == "WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED":
        return f"depth_pixels={streams.get('decode_status', {}).get('depth_pixels')}"
    if code == "WARN_RGB_PIXEL_DECODE_UNSUPPORTED":
        return f"rgb_pixels={streams.get('decode_status', {}).get('rgb_pixels')}"
    if code == "FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH":
        return "No strict clean segment met the minimum required duration."
    return "See raw report for technical details."


def _render_reason_table(
    title: str,
    severity: str,
    reasons: list[str],
    report: dict[str, Any],
) -> None:
    _render_header(
        title,
        "These are the reasons behind the gate result, with plain-language explanations.",
        level=3,
    )
    rows: list[dict[str, Any]] = []
    for code in reasons:
        rows.append(
            {
                "severity": severity,
                "reason_code": code,
                "meaning": REASON_DESCRIPTIONS.get(code, "No description available for this code yet."),
                "What We Observed": _format_reason_context(code, report),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_artifacts(metrics: dict[str, Any], output_dir: Path) -> None:
    _render_header(
        "Artifacts",
        "These are generated files you can open for deeper review.",
        level=3,
    )
    artifact_keys = [
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
    ]
    rows = []
    for key in artifact_keys:
        rel = metrics.get(key)
        if rel:
            rows.append({"artifact": key, "path": str((output_dir / str(rel)).resolve())})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No artifact files were produced for this run.")

    for img_key, caption, help_text in [
        (
            "sync_histogram_path",
            "Sync delta histogram (ms)",
            "This chart shows how far RGB and depth timestamps are from each other. Smaller values are better.",
        ),
        (
            "drop_timeline_path",
            "Drop or gap timeline",
            "This timeline shows where frame timing gaps happened. More gaps usually means less usable data.",
        ),
    ]:
        rel = metrics.get(img_key)
        if rel:
            _render_header(caption, help_text, level=3)
            _show_image_if_exists(output_dir / str(rel), caption)

    preview_images: list[Path] = []
    preview_relpaths = metrics.get("preview_relpaths")
    if isinstance(preview_relpaths, list) and preview_relpaths:
        for rel in preview_relpaths:
            p = output_dir / str(rel)
            if p.exists() and p.is_file():
                preview_images.append(p)
    if not preview_images:
        preview_dir = output_dir / "previews"
        if preview_dir.exists() and preview_dir.is_dir():
            preview_images = [
                p
                for p in preview_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
    if preview_images:
        _render_header(
            "Preview Frames",
            "These are context samples. Evidence sections below focus on issue examples.",
            level=3,
        )
        _show_image_set(preview_images, max_images=12)

    _render_header(
        "Blur evidence",
        "These are example frames used to illustrate blur pass/fail outcomes.",
        level=3,
    )
    blur_groups = [
        ("Fail examples", "blur_fail_frames_annotated_dir", "blur_fail_frames_dir"),
        ("Pass examples", "blur_pass_frames_annotated_dir", "blur_pass_frames_dir"),
    ]
    any_blur = False
    for label, ann_key, raw_key in blur_groups:
        st.markdown(f"**{label}**")
        rel = metrics.get(ann_key) or metrics.get(raw_key)
        if rel:
            folder = output_dir / str(rel)
            if folder.exists() and folder.is_dir():
                images = [
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                if images:
                    _show_image_set(images, max_images=12)
                    any_blur = True
                    continue
        st.caption("No example frames in this section for this run.")
    if not any_blur:
        st.info("No blur example frames were exported for this run.")

    exposure_views = [
        (
            "low_clip",
            "Exposure evidence: low clip",
            "exposure_low_clip_frames_dir",
            "These frames are very dark in shadow regions and lose detail there.",
        ),
        (
            "high_clip",
            "Exposure evidence: high clip",
            "exposure_high_clip_frames_dir",
            "These frames are very bright in highlight regions and lose detail there.",
        ),
        (
            "flat_and_dark",
            "Exposure evidence: flat and dark",
            "exposure_flat_and_dark_frames_dir",
            "These frames are dark and low-contrast, so scene details are hard to see.",
        ),
        (
            "flat_and_bright",
            "Exposure evidence: flat and bright",
            "exposure_flat_and_bright_frames_dir",
            "These frames are bright and low-contrast, so scene details are washed out.",
        ),
    ]
    counts = metrics.get("exposure_bad_reason_counts", {})
    for reason, title, key, help_text in exposure_views:
        _render_header(
            title,
            help_text,
            level=3,
        )
        reason_count = int(counts.get(reason, 0)) if isinstance(counts, dict) else 0
        rel = metrics.get(key)
        if rel:
            folder = output_dir / str(rel)
            if folder.exists() and folder.is_dir():
                images = [
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                if images:
                    st.caption("Showing example frames for this issue.")
                    _show_image_set(images, max_images=None)
                    continue
        if reason_count == 0:
            st.info("No example frames for this issue in this run.")
        else:
            if int(metrics.get("rgb_decode_success_count") or 0) <= 0:
                st.info("This issue appears in this run, but RGB frames were not available for example export.")
            else:
                st.info("This issue appears in this run, but example images could not be exported.")

    st.markdown('<div style="height: 0.85rem;"></div>', unsafe_allow_html=True)

    exposure_error_rel = metrics.get("exposure_evidence_error_path")
    if exposure_error_rel:
        exposure_error_path = output_dir / str(exposure_error_rel)
        if exposure_error_path.exists():
            st.warning(exposure_error_path.read_text(encoding="utf-8"))

    if ADVANCED_MODE:
        manifest_rel = metrics.get("evidence_manifest_path")
        if manifest_rel:
            manifest_path = output_dir / str(manifest_rel)
            if manifest_path.exists() and manifest_path.is_file():
                _render_header("Evidence Manifest", "Raw deterministic selection manifest.", level=3)
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    rows = []
                    evidence_sets = payload.get("evidence_sets", {})
                    for bucket in ("blur_fail", "blur_pass"):
                        for item in evidence_sets.get(bucket, []):
                            rows.append(
                                {
                                    "decision": item.get("decision"),
                                    "rank": item.get("rank"),
                                    "sample_i": item.get("sample_i"),
                                    "timestamp_ms": item.get("timestamp_ms"),
                                    "blur_value": item.get("blur_value"),
                                    "blur_threshold": item.get("blur_threshold"),
                                    "source_image_relpath": item.get("source_image_relpath"),
                                    "annotated_image_relpath": item.get("annotated_image_relpath"),
                                }
                            )
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Manifest is present but contains no evidence entries.")
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.warning(f"Could not parse evidence manifest: {exc}")


def _render_full_results(report: dict[str, Any], output_dir: Path) -> None:
    st.markdown("---")
    st.header("Latest Analysis Results", anchor=False)
    _render_gate_summary(report)

    metrics = report.get("metrics", {})

    _render_metrics_table(
        "Core Metrics",
        metrics,
        [
            "sync_p50_ms",
            "sync_p95_ms",
            "sync_max_ms",
            "sync_fail_ratio",
            "drop_ratio",
            "imu_accel_missing_ratio",
            "imu_gyro_missing_ratio",
            "imu_combined_missing_ratio",
            "integrity_ok_ratio",
            "integrity_coverage_seconds_est",
            "vision_ok_ratio",
            "vision_coverage_seconds_est",
        ],
        explanation="Main quality numbers. In general, lower sync/drop problems and higher coverage are better.",
    )

    _render_metrics_table(
        "Sync Diagnostics",
        metrics,
        [
            "sync_signed_p50_ms",
            "sync_signed_mean_ms",
            "sync_signed_std_ms",
            "sync_drift_ms_per_min",
            "sync_jitter_p95_ms",
            "rgb_timebase_diff_signed_p50_ms",
            "rgb_timebase_diff_signed_mean_ms",
            "rgb_timebase_diff_abs_p95_ms",
            "rgb_timebase_diff_abs_max_ms",
            "rgb_timebase_diff_sample_count",
            "rgb_timebase_header_present_ratio",
        ],
        explanation="Timing details between RGB and depth. Positive offset means depth is later than RGB.",
        hide_none=True,
    )

    _render_metrics_table(
        "Exposure / Blur / Depth",
        metrics,
        [
            "blur_fail_ratio",
            "blur_threshold",
            "blur_p10",
            "blur_p50",
            "blur_p90",
            "exposure_bad_ratio",
            "p50_mean",
            "dynamic_range_mean",
            "depth_invalid_mean",
            "depth_invalid_p95",
            "depth_fail_ratio",
            "rgb_decode_attempt_count",
            "rgb_decode_success_count",
            "depth_decode_attempt_count",
            "depth_decode_success_count",
            "blur_valid_frame_count",
            "exposure_valid_frame_count",
            "depth_valid_frame_count",
        ],
        explanation="Image/depth quality summary from analyzed sampled frames (not all frames in the MCAP).",
    )

    _render_header(
        "Exposure Reason Counts",
        "How many analyzed sampled frames matched each exposure issue type.",
        level=3,
    )
    st.json(metrics.get("exposure_bad_reason_counts", {}))
    st.caption(
        "Bright-looking frames are counted as high-clip only when clipped highlight area is above the high-clip threshold."
    )

    _render_header(
        "Integrity Segments",
        "These are continuous time ranges where timestamps and core sensor timing stay healthy.",
        level=3,
    )
    integrity_segments = report.get("segments", [])
    if integrity_segments:
        st.dataframe(
            _normalize_segment_rows(integrity_segments), use_container_width=True, hide_index=True
        )
    else:
        st.info("No integrity segments produced.")

    _render_header(
        "Clean Segments",
        "These are stricter segments that pass timing checks plus blur, exposure, and depth quality checks.",
        level=3,
    )
    clean_segments = _load_segments_from_metric(metrics, output_dir, "clean_segments_path")
    if clean_segments:
        st.dataframe(_normalize_segment_rows(clean_segments), use_container_width=True, hide_index=True)
    else:
        st.info("No clean segments produced.")

    _render_header("Errors", "Warnings and errors recorded during analysis.", level=3)
    errors = report.get("errors", [])
    if errors:
        st.dataframe(errors, use_container_width=True, hide_index=True)
    else:
        st.info("No errors recorded.")

    _render_artifacts(metrics, output_dir)

    st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
    st.caption(f"Run directory: `{output_dir.resolve()}`")
    st.caption(f"Report path: `{(output_dir / 'report.json').resolve()}`")

    with st.expander("Raw report.json", expanded=False):
        st.json(report)


hf_available = _has_hf_dependency()
hf_token = _read_hf_token()
token_digest = hashlib.sha256((hf_token or "no-token").encode("utf-8")).hexdigest()

# Session state
st.session_state.setdefault("mcap_list", [])
st.session_state.setdefault("mcap_list_key", None)
st.session_state.setdefault("mcap_list_loaded", False)
st.session_state.setdefault("selected_hf_file", None)
st.session_state.setdefault("local_refresh_nonce", 0)
st.session_state.setdefault("local_dir", str(Path.home() / "Downloads"))
st.session_state.setdefault("local_dir_choice", "~/Downloads")
st.session_state.setdefault("local_custom_dir", "")
st.session_state.setdefault("selected_local_file", None)
st.session_state.setdefault("latest_report", None)
st.session_state.setdefault("latest_output_dir", None)

progress = st.progress(0.0)
status_slot = st.empty()
status_text = st.empty()
status_text.caption("Status: Idle")
status_ref: dict[str, Any] = {"box": None}


def _start_status(label: str) -> None:
    status_text.empty()
    status_ref["box"] = status_slot.status(label, state="running", expanded=True)


def _update_status(label: str) -> None:
    box = status_ref.get("box")
    if box is not None:
        box.update(label=label, state="running", expanded=True)


def _finish_status_ok(label: str = "Done") -> None:
    box = status_ref.get("box")
    if box is not None:
        box.update(label=label, state="complete", expanded=False)
    status_ref["box"] = None
    status_text.caption(f"Status: {label}")


def _finish_status_fail(label: str = "Failed") -> None:
    box = status_ref.get("box")
    if box is None:
        status_ref["box"] = status_slot.status(label, state="error", expanded=True)
    else:
        box.update(label=label, state="error", expanded=True)
    status_ref["box"] = None
    status_text.caption(f"Status: {label}")


def _run_analysis(source_mode: str, selected_hf_path: str | None, selected_local_path: str | None) -> None:
    try:
        ensure_writable_dir(RUNS_BASE_DIR, "Runs directory")
        cfg = load_config(CONFIG_PATH)

        chosen = selected_hf_path if source_mode == "Hugging Face" else selected_local_path
        if chosen is None:
            raise RuntimeError("No file selected.")

        run_base = build_timestamped_run_basename(Path(str(chosen)).name)
        output_dir = allocate_run_dir(RUNS_BASE_DIR, run_base)
        write_latest_run_pointer(RUNS_BASE_DIR, output_dir)

        if source_mode == "Hugging Face":
            _start_status("cache: resolving dataset file")
        else:
            _start_status("local: validating file")
            progress.progress(0.05)

        source_path, analysis_start = _resolve_input_to_local_path(
            source_mode=source_mode,
            selected_local_path=selected_local_path,
            selected_hf_path=selected_hf_path,
            hf_token=hf_token,
        )

        result = analyze_file(
            input_path=source_path,
            output_dir=output_dir,
            config=cfg,
            progress_cb=_on_progress_scaled(analysis_start, 1.0),
        )

        st.session_state["latest_report"] = result.report
        st.session_state["latest_output_dir"] = str(output_dir.resolve())
        _finish_status_ok("Done")
    except Exception as exc:
        _error_box(exc, "Analysis failed.")
        _finish_status_fail("Failed")


# HF listing lifecycle
force_hf_refresh = False
hf_key_changed = st.session_state["mcap_list_key"] != (HF_REPO_ID, HF_REVISION, HF_PREFIX, token_digest)

if ADVANCED_MODE:
    with st.expander("Advanced", expanded=False):
        st.caption("Developer-only controls")
        if st.button("Refresh HF file list", key="advanced_refresh_hf"):
            _cached_hf_file_list.clear()
            st.session_state["mcap_list_loaded"] = False
            force_hf_refresh = True

current_hf_key = (HF_REPO_ID, HF_REVISION, HF_PREFIX, token_digest)
needs_hf_load = (
    force_hf_refresh
    or not st.session_state["mcap_list_loaded"]
    or st.session_state["mcap_list_key"] != current_hf_key
)

if needs_hf_load:
    if not hf_available:
        st.session_state["mcap_list"] = []
        st.session_state["mcap_list_key"] = current_hf_key
        st.session_state["mcap_list_loaded"] = True
    else:
        with st.spinner("Loading MCAP list..."):
            try:
                hf_files = _cached_hf_file_list(
                    repo_id=HF_REPO_ID,
                    revision=HF_REVISION,
                    prefix=HF_PREFIX,
                    token_digest=token_digest,
                    _token=hf_token,
                )
                st.session_state["mcap_list"] = hf_files
                st.session_state["mcap_list_key"] = current_hf_key
                st.session_state["mcap_list_loaded"] = True
            except Exception as exc:
                st.session_state["mcap_list"] = []
                st.session_state["mcap_list_key"] = current_hf_key
                st.session_state["mcap_list_loaded"] = False
                _error_box(exc, "Could not load file list from Hugging Face.")

hf_files = list(st.session_state.get("mcap_list", []))
hf_values = [row["path"] for row in hf_files]

if force_hf_refresh:
    if st.session_state["selected_hf_file"] not in hf_values:
        st.session_state["selected_hf_file"] = None
elif hf_key_changed:
    st.session_state["selected_hf_file"] = None
elif st.session_state["selected_hf_file"] not in hf_values:
    st.session_state["selected_hf_file"] = None

hf_options: list[str | None] = [None] + hf_values
hf_labels: dict[str | None, str] = {None: "Select an MCAP file"}
for row in hf_files:
    path = row["path"]
    hf_labels[path] = build_hf_display_label(path, HF_PREFIX, row.get("size_bytes"))

# Local folder + listing lifecycle
local_presets = _local_folder_presets(st.session_state.get("local_dir"))
local_choice_labels = [label for label, _ in local_presets]
if st.session_state["local_dir_choice"] not in local_choice_labels:
    st.session_state["local_dir_choice"] = local_choice_labels[0]

# Layout tabs
hf_tab, local_tab = st.tabs(["Hugging Face", "Local disk"])

with hf_tab:
    if not hf_available:
        st.warning("`huggingface_hub` is not installed. Local disk mode is still available.")
    selected_hf_path = st.selectbox(
        "MCAP file",
        options=hf_options,
        key="selected_hf_file",
        format_func=lambda value: hf_labels.get(value, str(value)),
    )
    if not hf_values:
        st.info("No `.mcap` files found under configured prefix.")

    if selected_hf_path:
        st.caption(f"Using Hugging Face: `{hf_labels.get(selected_hf_path, selected_hf_path)}`")
    else:
        st.caption("Choose a Hugging Face file to enable analysis.")

    if st.button("Analyze Hugging Face file", type="primary", key="analyze_hf", disabled=selected_hf_path is None):
        _run_analysis("Hugging Face", selected_hf_path=selected_hf_path, selected_local_path=None)

with local_tab:
    folder_label = st.selectbox("Folder", options=local_choice_labels, key="local_dir_choice")
    selected_folder_value = dict(local_presets).get(folder_label, "")
    if selected_folder_value == "__other__":
        custom_value = st.text_input("Custom folder path", key="local_custom_dir")
        resolved_local_dir = custom_value.strip()
    else:
        resolved_local_dir = selected_folder_value

    previous_local_dir = st.session_state.get("local_dir")
    if resolved_local_dir and resolved_local_dir != previous_local_dir:
        st.session_state["local_dir"] = resolved_local_dir
        st.session_state["selected_local_file"] = None

    if st.button("Refresh local list", key="local_refresh"):
        st.session_state["local_refresh_nonce"] += 1

    local_rows: list[dict[str, Any]] = []
    local_error: str | None = None
    effective_local_dir = st.session_state.get("local_dir", "")
    if effective_local_dir:
        cache_dir_key = _normalized_local_dir_key(effective_local_dir)
        try:
            local_rows = _cached_local_file_list(
                dir_path=cache_dir_key,
                nonce=int(st.session_state["local_refresh_nonce"]),
                max_files=LOCAL_MAX_FILES,
            )
        except LocalDirNotFound as exc:
            local_error = str(exc)
        except LocalDirNotReadable as exc:
            local_error = str(exc)
        except TooManyFiles as exc:
            local_error = (
                f"{exc} Narrow the folder or set EGOLOGQA_LOCAL_MAX_FILES to a higher value."
            )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            local_error = f"Failed to list local files: {exc}"

    if local_error:
        st.error(local_error)

    local_values = [row["path"] for row in local_rows]
    local_options: list[str | None] = [None] + local_values
    local_labels: dict[str | None, str] = {None: "Select an MCAP file"}
    for row in local_rows:
        local_labels[row["path"]] = make_local_option_label(
            str(row["name"]), int(row.get("size_bytes", 0))
        )

    if st.session_state["selected_local_file"] not in local_values:
        st.session_state["selected_local_file"] = None

    selected_local_path = st.selectbox(
        "MCAP file",
        options=local_options,
        key="selected_local_file",
        format_func=lambda value: local_labels.get(value, str(value)),
    )

    if not local_rows and not local_error and effective_local_dir:
        st.info("No `.mcap` files found in this folder.")

    if selected_local_path:
        st.caption(f"Using local disk: `{Path(str(selected_local_path)).resolve()}`")
    else:
        st.caption("Choose a local file to enable analysis.")

    if st.button("Analyze local file", type="primary", key="analyze_local", disabled=selected_local_path is None):
        _run_analysis("Local disk", selected_hf_path=None, selected_local_path=selected_local_path)

latest_report = st.session_state.get("latest_report")
latest_output_dir = st.session_state.get("latest_output_dir")
if latest_report and latest_output_dir:
    _render_full_results(latest_report, Path(str(latest_output_dir)))
