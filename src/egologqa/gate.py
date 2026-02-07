from __future__ import annotations

from typing import Any

from egologqa.constants import FAIL_REASON_ORDER, RECOMMENDED_ACTION, WARN_REASON_ORDER
from egologqa.models import QAConfig


def evaluate_gate(
    config: QAConfig,
    metrics: dict[str, Any],
    streams: dict[str, Any],
    duration_s: float,
    segments: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    fail_set: set[str] = set()
    warn_set: set[str] = set()

    if not streams.get("rgb_timestamps_present", False):
        fail_set.add("FAIL_NO_RGB_STREAM")

    if any(err.get("severity") == "ERROR" for err in errors):
        fail_set.add("FAIL_ANALYSIS_ERROR")

    sync_p95 = metrics.get("sync_p95_ms")
    drop_ratio = metrics.get("drop_ratio")
    if sync_p95 is not None and sync_p95 > config.thresholds.sync_fail_ms:
        fail_set.add("FAIL_SYNC_P95_GT_FAIL")
    if drop_ratio is not None and drop_ratio > config.thresholds.drop_fail_ratio:
        fail_set.add("FAIL_DROP_RATIO_GT_FAIL")
    if len(segments) == 0 and duration_s >= config.gate.fail_if_no_segments_min_duration_s:
        fail_set.add("FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH")

    if not streams.get("depth_timestamps_present", False):
        warn_set.add("WARN_DEPTH_TIMESTAMP_MISSING")
    if (
        config.decode.warn_on_depth_pixel_decode_failure
        and streams.get("decode_status", {}).get("depth_pixels") == "unsupported"
    ):
        warn_set.add("WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED")
    if streams.get("decode_status", {}).get("rgb_pixels") == "unsupported":
        warn_set.add("WARN_RGB_PIXEL_DECODE_UNSUPPORTED")
    if sync_p95 is not None and sync_p95 > config.thresholds.sync_warn_ms:
        warn_set.add("WARN_SYNC_P95_GT_WARN")
    if drop_ratio is not None and drop_ratio > config.thresholds.drop_warn_ratio:
        warn_set.add("WARN_DROP_RATIO_GT_WARN")
    imu_missing = metrics.get("imu_combined_missing_ratio")
    if imu_missing is not None and imu_missing > config.thresholds.imu_missing_warn_ratio:
        warn_set.add("WARN_IMU_MISSING_RATIO_GT_WARN")
    blur_fail_ratio = metrics.get("blur_fail_ratio")
    if blur_fail_ratio is not None and blur_fail_ratio > config.thresholds.blur_fail_warn_ratio:
        warn_set.add("WARN_BLUR_FAIL_RATIO_GT_WARN")
    exposure_bad_ratio = metrics.get("exposure_bad_ratio")
    if (
        exposure_bad_ratio is not None
        and exposure_bad_ratio > config.thresholds.exposure_bad_warn_ratio
    ):
        warn_set.add("WARN_EXPOSURE_BAD_RATIO_GT_WARN")
    depth_invalid_mean = metrics.get("depth_invalid_mean")
    if (
        depth_invalid_mean is not None
        and depth_invalid_mean > config.thresholds.depth_invalid_mean_warn
    ):
        warn_set.add("WARN_DEPTH_INVALID_MEAN_GT_WARN")

    fail_reasons = [code for code in FAIL_REASON_ORDER if code in fail_set]
    warn_reasons = [code for code in WARN_REASON_ORDER if code in warn_set]

    if fail_reasons:
        gate = "FAIL"
    elif warn_reasons:
        gate = "WARN"
    else:
        gate = "PASS"

    if gate == "PASS":
        floor_codes = set(config.gate.gate_warn_floor_error_codes)
        has_warn_floor = any(
            err.get("severity") == "WARN" and err.get("code") in floor_codes for err in errors
        )
        if has_warn_floor:
            gate = "WARN"

    return {
        "gate": gate,
        "recommended_action": RECOMMENDED_ACTION[gate],
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
    }
