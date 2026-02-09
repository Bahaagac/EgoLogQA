from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from egologqa import __version__


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit_or_unknown(cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_report_json(report: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    sanitized = sanitize_json_value(report)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    return report_path


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    return value


def empty_report(input_path: str, file_size_bytes: int | None = None, commit: str = "unknown") -> dict[str, Any]:
    return {
        "tool": {
            "name": "egologqa",
            "version": __version__,
            "git_commit": commit,
        },
        "input": {
            "file_path": input_path,
            "file_size_bytes": file_size_bytes,
            "analyzed_at_utc": now_utc_iso(),
        },
        "streams": {
            "rgb_topic": None,
            "depth_topic": None,
            "imu_accel_topic": None,
            "imu_gyro_topic": None,
            "imu_mode": "none",
            "topic_stats": {},
            "depth_topic_present": False,
            "depth_timestamps_present": False,
            "rgb_timestamps_present": False,
            "decode_status": {
                "rgb_pixels": "unsupported",
                "depth_pixels": "unsupported",
                "depth_timestamps": "missing",
            },
        },
        "time": {
            "time_base": "ns",
            "duration_s": 0.0,
        },
        "sampling": {
            "rgb_stride": None,
            "max_rgb_frames": None,
            "frames_analyzed": 0,
        },
        "metrics": {
            "file_total_messages": None,
            "file_duration_s": None,
            "file_bitrate_mbps": None,
            "sync_p50_ms": None,
            "sync_p95_ms": None,
            "sync_max_ms": None,
            "sync_fail_ratio": None,
            "sync_sample_count": 0,
            "sync_pattern": None,
            "sync_offset_estimate_ms": None,
            "expected_rgb_dt_ms": None,
            "drop_ratio": None,
            "imu_accel_missing_ratio": None,
            "imu_gyro_missing_ratio": None,
            "imu_combined_missing_ratio": None,
            "blur_median": None,
            "blur_threshold": None,
            "blur_fail_ratio": None,
            "blur_p10": None,
            "blur_p50": None,
            "blur_p90": None,
            "blur_valid_frame_count": 0,
            "exposure_bad_ratio": None,
            "exposure_valid_frame_count": 0,
            "exposure_bad_first_sample_i": None,
            "exposure_bad_last_sample_i": None,
            "low_clip_mean": None,
            "low_clip_p95": None,
            "high_clip_mean": None,
            "high_clip_p95": None,
            "contrast_mean": None,
            "contrast_p05": None,
            "dynamic_range_mean": None,
            "dynamic_range_p05": None,
            "p50_mean": None,
            "p50_p05": None,
            "p50_p95": None,
            "dark_frame_ratio": None,
            "low_clip_when_dark_mean": None,
            "exposure_bad_reason_counts": {},
            "exposure_debug_csv_path": None,
            "exposure_low_clip_frames_dir": None,
            "exposure_high_clip_frames_dir": None,
            "exposure_flat_and_dark_frames_dir": None,
            "exposure_flat_and_bright_frames_dir": None,
            "exposure_evidence_error_path": None,
            "blur_debug_csv_path": None,
            "depth_debug_csv_path": None,
            "blur_fail_frames_dir": None,
            "blur_pass_frames_dir": None,
            "clean_segments_basis": None,
            "clean_segments_path": None,
            "clean_segments_nosync_path": None,
            "rgb_decode_attempt_count": 0,
            "rgb_decode_success_count": 0,
            "depth_decode_attempt_count": 0,
            "depth_decode_success_count": 0,
            "depth_valid_frame_count": 0,
            "preview_count": 0,
            "preview_relpaths": [],
            "integrity_ok_ratio": None,
            "integrity_coverage_seconds_est": None,
            "vision_ok_ratio": None,
            "vision_coverage_seconds_est": None,
            "segments_basis": None,
            "depth_invalid_mean": None,
            "depth_invalid_p95": None,
            "depth_fail_ratio": None,
            "out_of_order": {},
        },
        "gate": {
            "gate": "FAIL",
            "recommended_action": "RECAPTURE_OR_SKIP",
            "fail_reasons": ["FAIL_ANALYSIS_ERROR"],
            "warn_reasons": [],
        },
        "segments": [],
        "config_used": {},
        "errors": [],
    }
