from __future__ import annotations

from typing import Any

from egologqa.constants import FAIL_REASON_ORDER, RECOMMENDED_ACTION, WARN_REASON_ORDER
from egologqa.models import QAConfig


FIXABLE_SYNC_REASONS = {
    "WARN_SYNC_P95_GT_WARN",
    "FAIL_SYNC_P95_GT_FAIL",
}

BLOCKING_FAIL_REASONS = {
    "FAIL_ANALYSIS_ERROR",
    "FAIL_NO_RGB_STREAM",
    "FAIL_DROP_RATIO_GT_FAIL",
    "FAIL_DEPTH_FAIL_RATIO_GT_FAIL",
    "FAIL_DEPTH_INVALID_MEAN_GT_FAIL",
}
# Future required-stream FAIL reasons should be added here to avoid
# suggesting FIX_TIME_ALIGNMENT when critical streams are missing.


def sync_offset_estimate_ms(metrics: dict[str, Any]) -> float | None:
    signed_p50 = metrics.get("sync_signed_p50_ms")
    if signed_p50 is not None:
        return float(signed_p50)
    signed_mean = metrics.get("sync_signed_mean_ms")
    if signed_mean is not None:
        return float(signed_mean)
    return None


def classify_sync_pattern(metrics: dict[str, Any], config: QAConfig) -> str:
    sample_count = int(metrics.get("sync_sample_count") or 0)
    if sample_count < config.thresholds.sync_min_samples:
        return "unavailable"

    sync_p95 = metrics.get("sync_p95_ms")
    if sync_p95 is None:
        return "unavailable"
    if float(sync_p95) <= config.thresholds.sync_warn_ms:
        return "ok"

    jitter = metrics.get("sync_jitter_p95_ms")
    drift = metrics.get("sync_drift_ms_per_min")
    if (jitter is not None and float(jitter) > config.thresholds.sync_jitter_warn_ms) or (
        drift is not None
        and abs(float(drift)) > config.thresholds.sync_drift_warn_ms_per_min
    ):
        return "unstable_timing"

    signed_p50 = metrics.get("sync_signed_p50_ms")
    signed_std = metrics.get("sync_signed_std_ms")
    if (
        signed_p50 is not None
        and signed_std is not None
        and jitter is not None
        and drift is not None
        and abs(float(signed_p50)) >= config.thresholds.sync_warn_ms
        and float(signed_std) <= config.thresholds.sync_stable_std_max_ms
        and float(jitter) <= config.thresholds.sync_stable_jitter_p95_max_ms
        and abs(float(drift)) <= config.thresholds.sync_stable_drift_abs_max_ms_per_min
    ):
        return "stable_offset"
    return "mixed_or_unclear"


def _has_segment_longer_than(
    segments: list[dict[str, Any]],
    threshold_s: float,
) -> bool:
    return any(float(seg.get("duration_s", 0.0)) >= threshold_s for seg in segments)


def evaluate_gate(
    config: QAConfig,
    metrics: dict[str, Any],
    streams: dict[str, Any],
    duration_s: float,
    segments: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    clean_segments: list[dict[str, Any]] | None = None,
    clean_segments_nosync: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean_segments = clean_segments or []
    clean_segments_nosync = clean_segments_nosync or []
    fail_set: set[str] = set()
    warn_set: set[str] = set()

    if any(err.get("severity") == "ERROR" for err in errors):
        fail_set.add("FAIL_ANALYSIS_ERROR")
    if not streams.get("rgb_timestamps_present", False):
        fail_set.add("FAIL_NO_RGB_STREAM")

    sync_p95 = metrics.get("sync_p95_ms")
    sync_sample_count = int(metrics.get("sync_sample_count") or 0)
    sync_evaluable = sync_sample_count >= config.thresholds.sync_min_samples
    sync_drift = metrics.get("sync_drift_ms_per_min")
    sync_jitter = metrics.get("sync_jitter_p95_ms")
    drop_ratio = metrics.get("drop_ratio")
    if sync_evaluable and sync_p95 is not None and float(sync_p95) > config.thresholds.sync_fail_ms:
        fail_set.add("FAIL_SYNC_P95_GT_FAIL")
    if drop_ratio is not None and drop_ratio > config.thresholds.drop_fail_ratio:
        fail_set.add("FAIL_DROP_RATIO_GT_FAIL")

    depth_eligible_for_fail = bool(streams.get("depth_topic_present", False)) and (
        int(metrics.get("depth_decode_success_count") or 0) > 0
    ) and (int(metrics.get("depth_valid_frame_count") or 0) > 0)
    depth_fail_ratio = metrics.get("depth_fail_ratio")
    if (
        depth_eligible_for_fail
        and depth_fail_ratio is not None
        and float(depth_fail_ratio) >= config.thresholds.depth_fail_ratio_fail
    ):
        fail_set.add("FAIL_DEPTH_FAIL_RATIO_GT_FAIL")
    depth_invalid_mean = metrics.get("depth_invalid_mean")
    if (
        depth_eligible_for_fail
        and depth_invalid_mean is not None
        and float(depth_invalid_mean) >= config.thresholds.depth_invalid_mean_fail
    ):
        fail_set.add("FAIL_DEPTH_INVALID_MEAN_GT_FAIL")

    if len(clean_segments) == 0 and duration_s >= config.gate.fail_if_no_segments_min_duration_s:
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
    if sync_evaluable and sync_p95 is not None and float(sync_p95) > config.thresholds.sync_warn_ms:
        warn_set.add("WARN_SYNC_P95_GT_WARN")
    if (
        sync_evaluable
        and sync_jitter is not None
        and float(sync_jitter) > config.thresholds.sync_jitter_warn_ms
    ):
        warn_set.add("WARN_SYNC_JITTER_P95_GT_WARN")
    if (
        sync_evaluable
        and sync_drift is not None
        and abs(float(sync_drift)) > config.thresholds.sync_drift_warn_ms_per_min
    ):
        warn_set.add("WARN_SYNC_DRIFT_ABS_GT_WARN")
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
        depth_eligible_for_fail
        and depth_invalid_mean is not None
        and float(depth_invalid_mean) > config.thresholds.depth_invalid_mean_warn
    ):
        warn_set.add("WARN_DEPTH_INVALID_MEAN_GT_WARN")

    sync_pattern = classify_sync_pattern(metrics, config)

    fail_reasons = [code for code in FAIL_REASON_ORDER if code in fail_set]
    warn_reasons = [code for code in WARN_REASON_ORDER if code in warn_set]

    if fail_reasons:
        gate = "FAIL"
    elif warn_reasons:
        gate = "WARN"
    else:
        gate = "PASS"

    # Downgrade a pure no-clean-segments failure when the counterfactual
    # demonstrates a stable-offset-only sync issue.
    if (
        gate == "FAIL"
        and fail_reasons == ["FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH"]
        and sync_pattern == "stable_offset"
        and len(clean_segments) == 0
        and len(clean_segments_nosync) > 0
    ):
        gate = "WARN"
        fail_reasons = []

    if gate == "PASS":
        floor_codes = set(config.gate.gate_warn_floor_error_codes)
        has_warn_floor = any(
            err.get("severity") == "WARN" and err.get("code") in floor_codes for err in errors
        )
        if has_warn_floor:
            gate = "WARN"

    strict_clean_exists = _has_segment_longer_than(
        clean_segments, config.segments.min_segment_seconds
    )
    integrity_exists = _has_segment_longer_than(segments, config.segments.min_segment_seconds)
    reasons = set(fail_reasons + warn_reasons)
    has_fixable_sync_reason = bool(reasons.intersection(FIXABLE_SYNC_REASONS))
    has_blocking_fail_reason = bool(set(fail_reasons).intersection(BLOCKING_FAIL_REASONS))
    if (
        sync_pattern == "stable_offset"
        and has_fixable_sync_reason
        and not has_blocking_fail_reason
    ):
        recommended_action = "FIX_TIME_ALIGNMENT"
    elif gate == "PASS":
        recommended_action = RECOMMENDED_ACTION["PASS"]
    elif strict_clean_exists:
        recommended_action = RECOMMENDED_ACTION["WARN"]
    elif fail_reasons == ["FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH"] and integrity_exists:
        recommended_action = RECOMMENDED_ACTION["WARN"]
    else:
        recommended_action = RECOMMENDED_ACTION["FAIL"]

    return {
        "gate": gate,
        "recommended_action": recommended_action,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
    }
