from __future__ import annotations

import html
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

from egologqa.config import load_config
from egologqa.io.hf_fetch import list_mcap_files, resolve_cached_file
from egologqa.kiosk_helpers import (
    allocate_run_dir,
    build_hf_display_label,
    build_timestamped_run_basename,
    ensure_writable_dir,
    human_bytes,
    map_error_bucket,
    resolve_runs_base_dir,
    stage_uploaded_mcap,
    write_latest_run_pointer,
)

from egologqa.pipeline import analyze_file
from egologqa.ui_text import recommended_action_copy


HF_REPO_ID = os.getenv("EGOLOGQA_HF_REPO_ID", "MicroAGI-Labs/MicroAGI00")
HF_REVISION = os.getenv("EGOLOGQA_HF_REVISION", "main")
HF_PREFIX = os.getenv("EGOLOGQA_HF_PREFIX", "raw_mcaps/")
HF_CACHE_DIR = Path(os.getenv("EGOLOGQA_HF_CACHE_DIR", "~/.cache/EgoLogQA/hf_mcaps")).expanduser()
RUNS_BASE_DIR = resolve_runs_base_dir(os.getenv("EGOLOGQA_RUNS_DIR"))
CONFIG_PATH = "configs/microagi00_ros2.yaml"
ADVANCED_MODE = os.getenv("EGOLOGQA_UI_ADVANCED", "0") == "1"


# ───────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EgoLogQA", layout="wide")

st.markdown(
    """
<style>
/* ── Layout tightening ─────────────────────────────────────────────── */
.block-container { padding-top: 3.5rem; padding-bottom: 1.6rem; max-width: 72rem; }
section[data-testid="stSidebar"] { display: none; }

/* ── Section help text ─────────────────────────────────────────────── */
.eq-help {
    margin: -0.25rem 0 0.5rem 0;
    font-size: 0.82rem;
    color: #7a8194;
    line-height: 1.4;
}

/* ── Gate banner ───────────────────────────────────────────────────── */
.gate-banner {
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.65rem;
}
.gate-banner .g-icon { font-size: 1.35rem; line-height: 1; }
.gate-banner .g-label { font-size: 1.3rem; font-weight: 700; letter-spacing: 0.01em; }
.gate-pass { background: #d4edda; border: 1px solid #b1dfbb; color: #155724; }
.gate-warn { background: #fff3cd; border: 1px solid #ffc107; color: #664d03; }
.gate-fail { background: #f8d7da; border: 1px solid #dc3545; color: #58151c; }

/* ── Action row ────────────────────────────────────────────────────── */
.act-card {
    background: #f7f8fa; border: 1px solid #e2e6ea; border-radius: 7px;
    padding: 0.7rem 1rem; height: 100%;
}
.act-card .act-lbl {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #7a8194; font-weight: 600; margin-bottom: 0.15rem;
}
.act-card .act-txt { font-size: 0.9rem; color: #1a1a2e; line-height: 1.45; }

/* ── Reason cards ──────────────────────────────────────────────────── */
.r-card {
    border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.45rem;
    border-left: 4px solid; font-size: 0.85rem;
}
.r-fail { background: #fef2f2; border-left-color: #dc3545; }
.r-warn { background: #fffcf0; border-left-color: #ffc107; }
.r-card .r-code {
    font-family: 'SFMono-Regular','Consolas',monospace; font-size: 0.78rem;
    font-weight: 600; color: #495057; margin-bottom: 0.15rem;
}
.r-card .r-msg { color: #343a40; margin-bottom: 0.1rem; }
.r-card .r-obs {
    font-size: 0.76rem; color: #7a8194;
    font-family: 'SFMono-Regular','Consolas',monospace;
}

/* ── Metric cards ──────────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.04em; color: #7a8194 !important;
}

/* ── Segment badge ─────────────────────────────────────────────────── */
.seg-b {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 0.12rem 0.5rem; border-radius: 10px; margin-left: 0.4rem;
    vertical-align: middle; color: #fff;
}
.seg-b-ok { background: #4A90D9; }
.seg-b-zero { background: #adb5bd; }

/* ── Exposure count chips ──────────────────────────────────────────── */
.ec-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: #f0f2f5; border: 1px solid #e2e6ea; border-radius: 5px;
    padding: 0.2rem 0.55rem; margin: 0.15rem 0.2rem 0.15rem 0;
    font-size: 0.8rem;
}
.ec-lbl { color: #7a8194; font-weight: 500; }
.ec-val { color: #1a1a2e; font-weight: 700; font-family: 'SFMono-Regular','Consolas',monospace; }

/* ── Source pill ────────────────────────────────────────────────────── */
.src-pill {
    display: inline-block; background: #e8f0fe; color: #1a56db;
    font-size: 0.8rem; font-weight: 500; padding: 0.25rem 0.65rem;
    border-radius: 5px; margin: 0.2rem 0 0.4rem 0;
}

/* ── Artifact rows ─────────────────────────────────────────────────── */
.af-row {
    display: flex; align-items: baseline; gap: 0.4rem;
    padding: 0.25rem 0; border-bottom: 1px solid #f0f0f3; font-size: 0.8rem;
}
.af-row:last-child { border-bottom: none; }
.af-key { color: #7a8194; font-weight: 500; min-width: 12rem; flex-shrink: 0; }
.af-val {
    font-family: 'SFMono-Regular','Consolas',monospace;
    font-size: 0.76rem; color: #495057; word-break: break-all;
}

/* ── Tab inner padding ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-panel"] { padding-top: 0.75rem; }

/* ── Compact section label ─────────────────────────────────────────── */
.eq-sec {
    font-size: 0.95rem; font-weight: 700; color: #1a1a2e;
    margin: 0.6rem 0 0.2rem 0; letter-spacing: 0.01em;
}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="margin-bottom:0.1rem;">'
    '<span style="font-size:1.7rem;font-weight:800;color:#1a1a2e;">EgoLogQA</span>'
    '&nbsp;<span style="font-size:0.75rem;color:#adb5bd;vertical-align:middle;">v0.1.0</span>'
    '</div>'
    '<p style="margin:0 0 0.8rem 0;font-size:0.85rem;color:#7a8194;">'
    'MicroAGI00 ROS2 MCAP quality assessment &mdash; choose a file below to analyze.'
    '</p>',
    unsafe_allow_html=True,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers (unchanged logic)
# ───────────────────────────────────────────────────────────────────────────
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


def _error_box(exc: Exception, default_msg: str) -> None:
    mapped = map_error_bucket(exc)
    st.error(mapped if mapped else default_msg)
    if ADVANCED_MODE:
        with st.expander("Error details", expanded=False):
            st.code(f"{exc.__class__.__name__}: {exc}")


def _build_run_results_zip_safe(output_dir: Path) -> Path:
    # Local archive builder to avoid cross-module import skew during Cloud hot updates.
    output_dir = Path(output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        raise RuntimeError(f"Run directory does not exist: {output_dir}")

    zip_path = output_dir / "run_results.zip"
    files_to_include: list[Path] = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(output_dir)
        if rel.parts and rel.parts[0] == "input":
            continue
        if rel.as_posix() == "run_results.zip":
            continue
        files_to_include.append(file_path)

    try:
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_include:
                rel = file_path.relative_to(output_dir).as_posix()
                zf.write(file_path, arcname=rel)
    except Exception as exc:  # pragma: no cover - UI fallback
        raise RuntimeError(f"Failed to create results archive: {exc}") from exc
    return zip_path


def _first_decode_error_context(errors: list[Any], codes: set[str]) -> dict[str, Any] | None:
    for item in errors:
        if not isinstance(item, dict):
            continue
        if str(item.get("code") or "") not in codes:
            continue
        context = item.get("context")
        if isinstance(context, dict):
            return context
    return None


def _phase_label(phase: str) -> str:
    mapping = {"scan": "scan", "pass1": "pass1", "pass2": "pass2", "done": "finalize", "error": "error"}
    return mapping.get(phase, phase)


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.4f}" if (abs(v) < 0.001 and v != 0.0) else f"{v:.3f}"
    return str(v)


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _fmt_bytes_gib(size_bytes: Any) -> str:
    value = _as_float(size_bytes)
    if value is None or value < 0.0:
        return "--"
    return f"{value / (1024 ** 3):.2f}"


def _fmt_seconds(seconds: Any) -> str:
    value = _as_float(seconds)
    if value is None or value < 0.0:
        return "--"
    return f"{value:.3f}"


def _fmt_mbps(mbps: Any) -> str:
    value = _as_float(mbps)
    if value is None or value < 0.0:
        return "--"
    return f"{value:.3f}"


def _topic_rate_hz(stats: Any) -> float | None:
    if not isinstance(stats, dict):
        return None
    hz = _as_float(stats.get("approx_rate_hz"))
    if hz is not None and hz >= 0.0:
        return hz
    duration_s = _as_float(stats.get("duration_s"))
    try:
        msg_count = int(stats.get("message_count"))
    except (TypeError, ValueError):
        return None
    if duration_s is None or duration_s <= 0.0 or msg_count < 2:
        return None
    return float((msg_count - 1) / duration_s)


def _fmt_hz(hz: Any) -> str:
    value = _as_float(hz)
    if value is None or value < 0.0:
        return "--"
    return f"{value:.3f}"


def _sec_label(text: str) -> None:
    st.markdown(f'<p class="eq-sec">{html.escape(text)}</p>', unsafe_allow_html=True)


def _help(text: str) -> None:
    st.markdown(f'<p class="eq-help">{html.escape(text)}</p>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# Metric labels
# ───────────────────────────────────────────────────────────────────────────
_ML: dict[str, str] = {
    "sync_p50_ms": "Sync P50 (ms)", "sync_p95_ms": "Sync P95 (ms)",
    "sync_max_ms": "Sync Max (ms)", "sync_fail_ratio": "Sync Fail Ratio",
    "drop_ratio": "Drop Ratio",
    "imu_accel_missing_ratio": "IMU Accel Missing",
    "imu_gyro_missing_ratio": "IMU Gyro Missing",
    "imu_combined_missing_ratio": "IMU Combined Missing",
    "integrity_ok_ratio": "Integrity OK", "integrity_coverage_seconds_est": "Integrity Coverage (s)",
    "vision_ok_ratio": "Vision OK", "vision_coverage_seconds_est": "Vision Coverage (s)",
    "sync_signed_p50_ms": "Signed Sync P50 (ms)", "sync_signed_mean_ms": "Signed Sync Mean (ms)",
    "sync_signed_std_ms": "Signed Sync Std (ms)", "sync_drift_ms_per_min": "Drift (ms/min)",
    "sync_jitter_p95_ms": "Jitter P95 (ms)",
    "rgb_timebase_diff_signed_p50_ms": "Timebase Diff P50 (ms)",
    "rgb_timebase_diff_signed_mean_ms": "Timebase Diff Mean (ms)",
    "rgb_timebase_diff_abs_p95_ms": "Timebase Diff P95 abs (ms)",
    "rgb_timebase_diff_abs_max_ms": "Timebase Diff Max abs (ms)",
    "rgb_timebase_diff_sample_count": "Timebase Sample Count",
    "rgb_timebase_header_present_ratio": "Header Present Ratio",
    "blur_fail_ratio": "Blur Fail Ratio", "blur_threshold": "Blur Threshold",
    "blur_p10": "Blur P10", "blur_p50": "Blur P50", "blur_p90": "Blur P90",
    "exposure_bad_ratio": "Exposure Bad Ratio", "p50_mean": "Brightness Mean",
    "dynamic_range_mean": "Dynamic Range Mean",
    "depth_invalid_mean": "Depth Invalid Mean", "depth_invalid_p95": "Depth Invalid P95",
    "depth_fail_ratio": "Depth Fail Ratio",
    "rgb_decode_attempt_count": "RGB Decode Attempts", "rgb_decode_success_count": "RGB Decode OK",
    "depth_decode_attempt_count": "Depth Decode Attempts", "depth_decode_success_count": "Depth Decode OK",
    "blur_valid_frame_count": "Blur Valid Frames",
    "exposure_valid_frame_count": "Exposure Valid Frames",
    "depth_valid_frame_count": "Depth Valid Frames",
}


def _metrics_df(metrics: dict, keys: list[str], hide_none: bool = False):
    rows = [{"Metric": _ML.get(k, k), "Value": metrics.get(k)} for k in keys]
    if hide_none:
        rows = [r for r in rows if r["Value"] is not None]
    return rows


# ───────────────────────────────────────────────────────────────────────────
# Reason data (unchanged)
# ───────────────────────────────────────────────────────────────────────────
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


def _r2(v: Any) -> str:
    """Round a value to 2 decimal places for display; pass non-floats through."""
    if isinstance(v, float):
        return f"{v:.2f}"
    if v is None:
        return "N/A"
    return str(v)


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
            f"sync_p95_ms={_r2(metrics.get('sync_p95_ms'))} "
            f"(warn={_r2(thresholds.get('sync_warn_ms'))}, fail={_r2(thresholds.get('sync_fail_ms'))})"
        )
    if code == "WARN_SYNC_JITTER_P95_GT_WARN":
        return (
            f"sync_jitter_p95_ms={_r2(metrics.get('sync_jitter_p95_ms'))} "
            f"(warn={_r2(thresholds.get('sync_jitter_warn_ms'))})"
        )
    if code == "WARN_SYNC_DRIFT_ABS_GT_WARN":
        return (
            f"sync_drift_ms_per_min={_r2(metrics.get('sync_drift_ms_per_min'))} "
            f"(abs warn={_r2(thresholds.get('sync_drift_warn_ms_per_min'))})"
        )
    if code == "FAIL_DROP_RATIO_GT_FAIL":
        return f"drop_ratio={_r2(metrics.get('drop_ratio'))} (fail={_r2(thresholds.get('drop_fail_ratio'))})"
    if code == "WARN_DROP_RATIO_GT_WARN":
        return f"drop_ratio={_r2(metrics.get('drop_ratio'))} (warn={_r2(thresholds.get('drop_warn_ratio'))})"
    if code == "WARN_IMU_MISSING_RATIO_GT_WARN":
        return (
            f"imu_combined_missing_ratio={_r2(metrics.get('imu_combined_missing_ratio'))} "
            f"(warn={_r2(thresholds.get('imu_missing_warn_ratio'))})"
        )
    if code == "WARN_BLUR_FAIL_RATIO_GT_WARN":
        return (
            f"blur_fail_ratio={_r2(metrics.get('blur_fail_ratio'))} "
            f"(warn={_r2(thresholds.get('blur_fail_warn_ratio'))})"
        )
    if code == "WARN_EXPOSURE_BAD_RATIO_GT_WARN":
        return (
            f"exposure_bad_ratio={_r2(metrics.get('exposure_bad_ratio'))} "
            f"(warn={_r2(thresholds.get('exposure_bad_warn_ratio'))})"
        )
    if code == "FAIL_DEPTH_FAIL_RATIO_GT_FAIL":
        return (
            f"depth_fail_ratio={_r2(metrics.get('depth_fail_ratio'))} "
            f"(fail={_r2(thresholds.get('depth_fail_ratio_fail'))})"
        )
    if code == "FAIL_DEPTH_INVALID_MEAN_GT_FAIL":
        return (
            f"depth_invalid_mean={_r2(metrics.get('depth_invalid_mean'))} "
            f"(fail={_r2(thresholds.get('depth_invalid_mean_fail'))})"
        )
    if code == "WARN_DEPTH_INVALID_MEAN_GT_WARN":
        return (
            f"depth_invalid_mean={_r2(metrics.get('depth_invalid_mean'))} "
            f"(warn={_r2(thresholds.get('depth_invalid_mean_warn'))})"
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


# ───────────────────────────────────────────────────────────────────────────
# Image helpers
# ───────────────────────────────────────────────────────────────────────────
def _show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists() and path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)


def _show_image_set(paths: list[Path], max_images: int | None = 12) -> None:
    images = sorted(paths, key=lambda p: p.name.lower())
    if max_images is not None:
        images = images[:max_images]
    if not images:
        return
    cols_per_row = min(len(images), 4)
    for row_start in range(0, len(images), cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, img in enumerate(row_images):
            with cols[idx]:
                st.image(str(img), use_container_width=True)
                st.caption(img.name)


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


def _normalize_segment_rows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        rows.append({"Start (ns)": seg.get("start_ns"), "End (ns)": seg.get("end_ns"), "Duration (s)": seg.get("duration_s")})
    return rows


# ───────────────────────────────────────────────────────────────────────────
# Input resolution
# ───────────────────────────────────────────────────────────────────────────
def _resolve_input_to_local_path(
    source_mode: str,
    selected_hf_path: str | None,
    hf_token: str | None,
    uploaded_local_file: Any | None,
    output_dir: Path,
) -> tuple[Path, float]:
    if source_mode == "Local disk":
        if uploaded_local_file is None:
            raise RuntimeError("No uploaded file selected.")
        _update_status("upload: staging file")
        staged_path = stage_uploaded_mcap(uploaded_local_file=uploaded_local_file, output_dir=output_dir)
        return staged_path, 0.08

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


# ───────────────────────────────────────────────────────────────────────────
# Artifact labels
# ───────────────────────────────────────────────────────────────────────────
_AF: dict[str, str] = {
    "sync_histogram_path": "Sync Histogram", "drop_timeline_path": "Drop Timeline",
    "exposure_debug_csv_path": "Exposure CSV",
    "exposure_low_clip_frames_dir": "Exposure: Low Clip",
    "exposure_high_clip_frames_dir": "Exposure: High Clip",
    "exposure_flat_and_dark_frames_dir": "Exposure: Flat & Dark",
    "exposure_flat_and_bright_frames_dir": "Exposure: Flat & Bright",
    "exposure_evidence_error_path": "Exposure Evidence Errors",
    "blur_debug_csv_path": "Blur CSV", "depth_debug_csv_path": "Depth CSV",
    "blur_fail_frames_dir": "Blur Fail Frames", "blur_pass_frames_dir": "Blur Pass Frames",
    "blur_fail_frames_annotated_dir": "Blur Fail (Annotated)",
    "blur_pass_frames_annotated_dir": "Blur Pass (Annotated)",
    "clean_segments_path": "Clean Segments JSON",
    "clean_segments_nosync_path": "Clean Segments (No-Sync)",
    "evidence_manifest_path": "Evidence Manifest", "benchmarks_path": "Benchmarks",
}

_AF_KEYS = list(_AF.keys())


# ───────────────────────────────────────────────────────────────────────────
# Full results rendering  (restructured layout, same data)
# ───────────────────────────────────────────────────────────────────────────
def _render_full_results(report: dict[str, Any], output_dir: Path) -> None:
    st.markdown("---")
    metrics = report.get("metrics", {})
    streams = report.get("streams", {})
    gate = report.get("gate", {})
    gate_name = gate.get("gate")
    fail_reasons = gate.get("fail_reasons", [])
    warn_reasons = gate.get("warn_reasons", [])

    # ━━ 1. GATE BANNER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if gate_name == "PASS":
        bcls, icon, dlabel = "gate-pass", "&#10003;", "PASS"
    elif gate_name == "WARN":
        bcls, icon, dlabel = "gate-warn", "&#9888;", "WARNING"
    else:
        bcls, icon, dlabel = "gate-fail", "&#10007;", "FAIL"

    st.markdown(
        f'<div class="gate-banner {bcls}">'
        f'<span class="g-icon">{icon}</span><span class="g-label">{dlabel}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    action_token = str(gate.get("recommended_action") or "UNKNOWN")
    action_copy = recommended_action_copy(
        action_token=action_token, gate=str(gate_name),
        fail_reasons=fail_reasons, warn_reasons=warn_reasons,
    )
    st.markdown(
        '<div class="act-card">'
        f'<div class="act-txt"><strong>What to do:</strong> {html.escape(action_copy["what_to_do"])}</div>'
        f'<div style="margin-top:0.25rem;font-size:0.82rem;color:#7a8194;">'
        f'<strong>Why:</strong> {html.escape(action_copy["why"])}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Reason cards (compact, only when present)
    if gate_name == "FAIL" and fail_reasons:
        _sec_label("Failure Reasons")
        _render_reason_cards("FAIL", fail_reasons, report)
    if gate_name in {"WARN", "FAIL"} and warn_reasons:
        _sec_label("Warning Reasons")
        _render_reason_cards("WARNING", warn_reasons, report)

    topic_stats = streams.get("topic_stats", {})
    topic_stats_map = topic_stats if isinstance(topic_stats, dict) else {}
    file_size_bytes = report.get("input", {}).get("file_size_bytes")
    file_duration_s = metrics.get("file_duration_s")
    if file_duration_s is None:
        file_duration_s = report.get("time", {}).get("duration_s")

    file_bitrate_mbps = metrics.get("file_bitrate_mbps")
    if file_bitrate_mbps is None:
        file_size_float = _as_float(file_size_bytes)
        file_duration_float = _as_float(file_duration_s)
        if (
            file_size_float is not None
            and file_duration_float is not None
            and file_duration_float > 0.0
        ):
            file_bitrate_mbps = (file_size_float * 8.0) / (file_duration_float * 1_000_000.0)

    rgb_topic = streams.get("rgb_topic")
    depth_topic = streams.get("depth_topic")
    imu_accel_topic = streams.get("imu_accel_topic")
    imu_gyro_topic = streams.get("imu_gyro_topic")
    rgb_stats = topic_stats_map.get(rgb_topic) if isinstance(rgb_topic, str) else None
    depth_stats = topic_stats_map.get(depth_topic) if isinstance(depth_topic, str) else None
    rgb_rate_hz = _topic_rate_hz(rgb_stats)
    depth_rate_hz = _topic_rate_hz(depth_stats)

    imu_rates: list[float] = []
    imu_topics_seen: set[str] = set()
    for imu_topic in [imu_accel_topic, imu_gyro_topic]:
        if not isinstance(imu_topic, str) or not imu_topic or imu_topic in imu_topics_seen:
            continue
        imu_topics_seen.add(imu_topic)
        rate = _topic_rate_hz(topic_stats_map.get(imu_topic))
        if rate is not None:
            imu_rates.append(rate)
    imu_rate_hz = float(sum(imu_rates) / len(imu_rates)) if imu_rates else None

    # ━━ 2. FILE KEY NUMBERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("File Key Numbers")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    with f1:
        st.metric("FILE SIZE (GiB)", _fmt_bytes_gib(file_size_bytes))
    with f2:
        st.metric("DURATION (s)", _fmt_seconds(file_duration_s))
    with f3:
        st.metric("AVG BITRATE (Mbps)", _fmt_mbps(file_bitrate_mbps))
    with f4:
        st.metric("RGB RATE (Hz)", _fmt_hz(rgb_rate_hz))
    with f5:
        st.metric("DEPTH RATE (Hz)", _fmt_hz(depth_rate_hz))
    with f6:
        st.metric("IMU RATE (Hz)", _fmt_hz(imu_rate_hz))

    # ━━ 3. KEY NUMBERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Analysis Key Numbers")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Sync P95", _fmt(metrics.get("sync_p95_ms")))
    with m2:
        st.metric("Drop Ratio", _fmt(metrics.get("drop_ratio")))
    with m3:
        st.metric("Integrity", _fmt(metrics.get("integrity_ok_ratio")))
    with m4:
        st.metric("Vision", _fmt(metrics.get("vision_ok_ratio")))
    with m5:
        st.metric("Blur Fail", _fmt(metrics.get("blur_fail_ratio")))
    with m6:
        st.metric("Exposure Bad", _fmt(metrics.get("exposure_bad_ratio")))

    # ━━ 4. SEGMENTS (side by side) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Segments")
    _help("Integrity segments use core timing; clean segments add blur, exposure, and depth checks.")

    integrity_segments = report.get("segments", [])
    clean_segments = _load_segments_from_metric(metrics, output_dir, "clean_segments_path")
    seg_col1, seg_col2 = st.columns(2)

    with seg_col1:
        ic = len(integrity_segments) if isinstance(integrity_segments, list) else 0
        bc = "seg-b seg-b-ok" if ic > 0 else "seg-b seg-b-zero"
        st.markdown(f'Integrity Segments <span class="{bc}">{ic}</span>', unsafe_allow_html=True)
        if integrity_segments:
            st.dataframe(_normalize_segment_rows(integrity_segments), use_container_width=True, hide_index=True)
        else:
            st.caption("None produced.")

    with seg_col2:
        cc = len(clean_segments)
        bc2 = "seg-b seg-b-ok" if cc > 0 else "seg-b seg-b-zero"
        st.markdown(f'Clean Segments <span class="{bc2}">{cc}</span>', unsafe_allow_html=True)
        if clean_segments:
            st.dataframe(_normalize_segment_rows(clean_segments), use_container_width=True, hide_index=True)
        else:
            st.caption("None produced.")

    # ━━ 5. CORE METRICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Core Metrics")
    _help("Main quality numbers. Lower sync/drop and higher coverage is better.")
    rows = _metrics_df(metrics, [
        "sync_p50_ms", "sync_p95_ms", "sync_max_ms", "sync_fail_ratio",
        "drop_ratio",
        "imu_accel_missing_ratio", "imu_gyro_missing_ratio", "imu_combined_missing_ratio",
        "integrity_ok_ratio", "integrity_coverage_seconds_est",
        "vision_ok_ratio", "vision_coverage_seconds_est",
    ])
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No data.")

    # ━━ 6. SYNC DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Sync Diagnostics")
    _help("Timing details between RGB and depth. Positive offset = depth later than RGB.")
    rows = _metrics_df(metrics, [
        "sync_signed_p50_ms", "sync_signed_mean_ms", "sync_signed_std_ms",
        "sync_drift_ms_per_min", "sync_jitter_p95_ms",
        "rgb_timebase_diff_signed_p50_ms", "rgb_timebase_diff_signed_mean_ms",
        "rgb_timebase_diff_abs_p95_ms", "rgb_timebase_diff_abs_max_ms",
        "rgb_timebase_diff_sample_count", "rgb_timebase_header_present_ratio",
    ], hide_none=True)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No diagnostics available.")

    # ━━ 7. EXPOSURE / BLUR / DEPTH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Exposure / Blur / Depth")
    _help("Image/depth quality from sampled frames (not every frame in the MCAP).")
    rows = _metrics_df(metrics, [
        "blur_fail_ratio", "blur_threshold", "blur_p10", "blur_p50", "blur_p90",
        "exposure_bad_ratio", "p50_mean", "dynamic_range_mean",
        "depth_invalid_mean", "depth_invalid_p95", "depth_fail_ratio",
        "rgb_decode_attempt_count", "rgb_decode_success_count",
        "depth_decode_attempt_count", "depth_decode_success_count",
        "blur_valid_frame_count", "exposure_valid_frame_count", "depth_valid_frame_count",
    ])
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No data.")

    # ━━ 8. EXPOSURE REASON COUNTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Exposure Reason Counts")
    _help("How many sampled frames matched each exposure issue type.")
    exp_counts = metrics.get("exposure_bad_reason_counts", {})
    if isinstance(exp_counts, dict) and exp_counts:
        chips = "".join(
            f'<span class="ec-chip"><span class="ec-lbl">{html.escape(str(k))}</span>'
            f'<span class="ec-val">{v}</span></span>'
            for k, v in exp_counts.items()
        )
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.caption("No exposure issues detected.")
    st.caption(
        "Bright-looking frames are counted as high-clip only when "
        "clipped highlight area is above the high-clip threshold."
    )

    # ━━ 9. ERRORS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Errors")
    _help("Warnings and errors recorded during analysis.")
    errors = report.get("errors", [])
    if errors:
        st.dataframe(errors, use_container_width=True, hide_index=True)
    else:
        st.info("No errors recorded.")

    # ━━ 10. ARTIFACTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _sec_label("Artifacts")
    _help("Generated files you can open for deeper review.")
    af_parts: list[str] = []
    for key in _AF_KEYS:
        rel = metrics.get(key)
        if rel:
            lbl = html.escape(_AF.get(key, key))
            fp = html.escape(str((output_dir / str(rel)).resolve()))
            af_parts.append(f'<div class="af-row"><span class="af-key">{lbl}</span><span class="af-val">{fp}</span></div>')
    if af_parts:
        with st.expander("File Paths", expanded=False):
            st.markdown("".join(af_parts), unsafe_allow_html=True)
    else:
        st.info("No artifact files were produced for this run.")

    # Plot images
    for img_key, caption, ht in [
        ("sync_histogram_path", "Sync delta histogram (ms)",
         "How far RGB and depth timestamps are from each other. Smaller is better."),
        ("drop_timeline_path", "Drop / gap timeline",
         "Where frame timing gaps happened. More gaps = less usable data."),
    ]:
        rel = metrics.get(img_key)
        if rel:
            _sec_label(caption)
            _help(ht)
            _show_image_if_exists(output_dir / str(rel), caption)

    rgb_decode_success_count = int(metrics.get("rgb_decode_success_count") or 0)
    if rgb_decode_success_count == 0:
        st.warning("No preview/evidence images were generated because RGB decoding failed.")
        decode_context = _first_decode_error_context(
            errors, {"RGB_DECODE_FAIL", "BLUR_UNAVAILABLE_NO_DECODE"}
        )
        details: list[str] = []
        cv2_import_error = None
        if decode_context is not None:
            cv2_import_error = decode_context.get("cv2_import_error")
            cv2_available = decode_context.get("cv2_available")
            if isinstance(cv2_available, bool):
                details.append(f"cv2_available={cv2_available}")
            attempts = decode_context.get("rgb_decode_attempt_count")
            if attempts is not None:
                details.append(f"rgb_decode_attempt_count={attempts}")
        if not isinstance(cv2_import_error, str) or not cv2_import_error.strip():
            metric_hint = metrics.get("cv2_import_error")
            if isinstance(metric_hint, str) and metric_hint.strip():
                cv2_import_error = metric_hint
        if isinstance(cv2_import_error, str) and cv2_import_error.strip():
            details.insert(0, f"cv2_import_error={cv2_import_error}")
        if details:
            st.caption("Decode diagnostics: " + " | ".join(details[:3]))

    # Preview frames
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
                p for p in preview_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
    if preview_images:
        _sec_label("Preview Frames")
        _help("Context samples from the recording. Evidence sections below focus on specific issues.")
        _show_image_set(preview_images, max_images=12)

    # Blur evidence
    _sec_label("Blur Evidence")
    _help("Example frames illustrating blur pass/fail outcomes.")
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
                images = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if images:
                    _show_image_set(images, max_images=12)
                    any_blur = True
                    continue
        st.caption("No example frames in this section for this run.")
    if not any_blur:
        st.info("No blur example frames were exported for this run.")

    # Exposure evidence
    exposure_views = [
        ("low_clip", "Exposure Evidence: Low Clip", "exposure_low_clip_frames_dir",
         "Frames very dark in shadow regions, losing detail."),
        ("high_clip", "Exposure Evidence: High Clip", "exposure_high_clip_frames_dir",
         "Frames very bright in highlights, losing detail."),
        ("flat_and_dark", "Exposure Evidence: Flat & Dark", "exposure_flat_and_dark_frames_dir",
         "Dark and low-contrast frames where scene details are hard to see."),
        ("flat_and_bright", "Exposure Evidence: Flat & Bright", "exposure_flat_and_bright_frames_dir",
         "Bright and low-contrast frames where scene details are washed out."),
    ]
    counts = metrics.get("exposure_bad_reason_counts", {})
    for reason, title, key, ht in exposure_views:
        _sec_label(title)
        _help(ht)
        reason_count = int(counts.get(reason, 0)) if isinstance(counts, dict) else 0
        rel = metrics.get(key)
        if rel:
            folder = output_dir / str(rel)
            if folder.exists() and folder.is_dir():
                images = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
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

    # ━━ Advanced: evidence manifest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ADVANCED_MODE:
        manifest_rel = metrics.get("evidence_manifest_path")
        if manifest_rel:
            manifest_path = output_dir / str(manifest_rel)
            if manifest_path.exists() and manifest_path.is_file():
                with st.expander("Evidence Manifest (advanced)", expanded=False):
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                        rows = []
                        evidence_sets = payload.get("evidence_sets", {})
                        for bucket in ("blur_fail", "blur_pass"):
                            for item in evidence_sets.get(bucket, []):
                                rows.append({
                                    "decision": item.get("decision"),
                                    "rank": item.get("rank"),
                                    "sample_i": item.get("sample_i"),
                                    "timestamp_ms": item.get("timestamp_ms"),
                                    "blur_value": item.get("blur_value"),
                                    "blur_threshold": item.get("blur_threshold"),
                                    "source_image_relpath": item.get("source_image_relpath"),
                                    "annotated_image_relpath": item.get("annotated_image_relpath"),
                                })
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        else:
                            st.caption("Manifest is present but contains no evidence entries.")
                    except Exception as exc:  # pragma: no cover - UI fallback
                        st.warning(f"Could not parse evidence manifest: {exc}")

    # ━━ RESULTS DOWNLOADS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
    _sec_label("Results Download")

    report_path = output_dir / "report.json"
    if report_path.exists() and report_path.is_file():
        report_bytes = report_path.read_bytes()
    else:
        report_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )

    key_suffix = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "Download report.json",
            data=report_bytes,
            file_name="report.json",
            mime="application/json",
            key=f"download_report_{key_suffix}",
            use_container_width=True,
        )

    zip_path = output_dir / "run_results.zip"
    zip_error: str | None = None
    try:
        if not zip_path.exists():
            zip_path = _build_run_results_zip_safe(output_dir)
    except Exception as exc:  # pragma: no cover - UI fallback
        zip_error = str(exc)

    with dcol2:
        if zip_error:
            st.warning(f"Could not prepare run archive: {zip_error}")
        elif zip_path.exists() and zip_path.is_file():
            st.download_button(
                "Download Run Results (.zip)",
                data=zip_path.read_bytes(),
                file_name=zip_path.name,
                mime="application/zip",
                key=f"download_zip_{key_suffix}",
                use_container_width=True,
            )
        else:
            st.warning("Run archive is not available for download.")

    st.caption("ZIP includes generated report and artifacts, and excludes source uploads.")

    with st.expander("Raw report.json", expanded=False):
        st.json(report)


def _render_reason_cards(severity: str, reasons: list[str], report: dict[str, Any]) -> None:
    card_cls = "r-fail" if severity == "FAIL" else "r-warn"
    for code in reasons:
        meaning = REASON_DESCRIPTIONS.get(code, "No description available.")
        observed = _format_reason_context(code, report)
        st.markdown(
            f'<div class="r-card {card_cls}">'
            f'<div class="r-code">{html.escape(code)}</div>'
            f'<div class="r-msg">{html.escape(meaning)}</div>'
            f'<div class="r-obs">{html.escape(observed)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────────────────
# Session state (unchanged)
# ───────────────────────────────────────────────────────────────────────────
hf_available = _has_hf_dependency()
hf_token = _read_hf_token()
token_digest = hashlib.sha256((hf_token or "no-token").encode("utf-8")).hexdigest()

st.session_state.setdefault("mcap_list", [])
st.session_state.setdefault("mcap_list_key", None)
st.session_state.setdefault("mcap_list_loaded", False)
st.session_state.setdefault("selected_hf_file", None)
st.session_state.setdefault("latest_report", None)
st.session_state.setdefault("latest_output_dir", None)

# ───────────────────────────────────────────────────────────────────────────
# Progress & status widgets
# ───────────────────────────────────────────────────────────────────────────
progress = st.progress(0.0)
status_slot = st.empty()
status_text = st.empty()
status_text.empty()
status_ref: dict[str, Any] = {"box": None}


def _on_progress_scaled(start: float, end: float):
    span = max(0.0, end - start)

    def _cb(event: dict) -> None:
        frac = min(max(float(event.get("progress", 0.0)), 0.0), 1.0)
        progress.progress(start + span * frac)
        phase = _phase_label(str(event.get("phase", "analysis")))
        _update_status(f"{phase}: {event.get('message', '...')}")

    return _cb


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
    status_text.empty()


def _finish_status_fail(label: str = "Failed") -> None:
    box = status_ref.get("box")
    if box is None:
        status_ref["box"] = status_slot.status(label, state="error", expanded=True)
    else:
        box.update(label=label, state="error", expanded=True)
    status_ref["box"] = None
    status_text.empty()


# ───────────────────────────────────────────────────────────────────────────
# Run analysis
# ───────────────────────────────────────────────────────────────────────────
def _run_analysis(
    source_mode: str,
    selected_hf_path: str | None,
    uploaded_local_file: Any | None,
) -> None:
    try:
        ensure_writable_dir(RUNS_BASE_DIR, "Runs directory")
        cfg = load_config(CONFIG_PATH)

        chosen = selected_hf_path if source_mode == "Hugging Face" else getattr(uploaded_local_file, "name", None)
        if chosen is None:
            raise RuntimeError("No file selected.")

        run_base = build_timestamped_run_basename(Path(str(chosen)).name)
        output_dir = allocate_run_dir(RUNS_BASE_DIR, run_base)
        write_latest_run_pointer(RUNS_BASE_DIR, output_dir)

        if source_mode == "Hugging Face":
            _start_status("cache: resolving dataset file")
        else:
            _start_status("upload: staging file")
            progress.progress(0.05)

        source_path, analysis_start = _resolve_input_to_local_path(
            source_mode=source_mode,
            selected_hf_path=selected_hf_path,
            hf_token=hf_token,
            uploaded_local_file=uploaded_local_file,
            output_dir=output_dir,
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


# ───────────────────────────────────────────────────────────────────────────
# HF listing lifecycle (unchanged logic)
# ───────────────────────────────────────────────────────────────────────────
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
                    repo_id=HF_REPO_ID, revision=HF_REVISION,
                    prefix=HF_PREFIX, token_digest=token_digest, _token=hf_token,
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

# ───────────────────────────────────────────────────────────────────────────
# Source tabs
# ───────────────────────────────────────────────────────────────────────────
hf_tab, local_tab = st.tabs(["Hugging Face", "Local disk"])

with hf_tab:
    if not hf_available:
        st.warning("`huggingface_hub` is not installed. Upload mode is still available.")
    selected_hf_path = st.selectbox(
        "MCAP file", options=hf_options, key="selected_hf_file",
        format_func=lambda value: hf_labels.get(value, str(value)),
    )
    if not hf_values:
        st.info("No `.mcap` files found under configured prefix.")

    if not selected_hf_path:
        st.caption("Choose a Hugging Face file to enable analysis.")

    if st.button("Analyze Hugging Face file", type="primary", key="analyze_hf", disabled=selected_hf_path is None):
        _run_analysis("Hugging Face", selected_hf_path=selected_hf_path, uploaded_local_file=None)

with local_tab:
    uploaded_local_file = st.file_uploader(
        "MCAP file",
        type=["mcap"],
        accept_multiple_files=False,
        key="uploaded_local_file",
    )
    if uploaded_local_file is not None:
        upload_size = getattr(uploaded_local_file, "size", None)
        size_text = human_bytes(upload_size if isinstance(upload_size, int) else None)
        st.caption(f"Selected upload: {uploaded_local_file.name} ({size_text})")
    else:
        st.caption("Upload a local `.mcap` file to enable analysis.")

    if st.button("Analyze local file", type="primary", key="analyze_local", disabled=uploaded_local_file is None):
        _run_analysis("Local disk", selected_hf_path=None, uploaded_local_file=uploaded_local_file)

# ───────────────────────────────────────────────────────────────────────────
# Render results if available
# ───────────────────────────────────────────────────────────────────────────
latest_report = st.session_state.get("latest_report")
latest_output_dir = st.session_state.get("latest_output_dir")
if latest_report and latest_output_dir:
    _render_full_results(latest_report, Path(str(latest_output_dir)))
