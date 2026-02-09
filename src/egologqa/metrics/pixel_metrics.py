from __future__ import annotations

import numpy as np

from egologqa.models import ThresholdsConfig


def compute_rgb_pixel_metrics(
    rgb_frames: list[np.ndarray],
    thresholds: ThresholdsConfig,
    sample_indices: list[int] | None = None,
    sample_times_ms: list[float] | None = None,
) -> tuple[
    dict[str, float | None | dict[str, int]],
    list[bool],
    list[bool],
    list[dict[str, float | int | str]],
    list[dict[str, str | int | float]],
]:
    try:
        import cv2
    except Exception:
        return (
            {
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
                "exposure_bad_reason_counts": {
                    "low_clip": 0,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [],
            [],
            [],
            [],
        )
    if not rgb_frames:
        return (
            {
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
                "exposure_bad_reason_counts": {
                    "low_clip": 0,
                    "high_clip": 0,
                    "flat_and_dark": 0,
                    "flat_and_bright": 0,
                },
            },
            [],
            [],
            [],
            [],
        )

    sample_indices = sample_indices or list(range(len(rgb_frames)))
    sample_times_ms = sample_times_ms or [float(i) for i in range(len(rgb_frames))]
    blur_scores: list[float | None] = []
    exposure_ok: list[bool] = [True] * len(rgb_frames)
    exposure_rows: list[dict[str, float | int | str]] = []
    exposure_compute_errors: list[dict[str, str | int | float]] = []
    reason_counts: dict[str, int] = {
        "low_clip": 0,
        "high_clip": 0,
        "flat_and_dark": 0,
        "flat_and_bright": 0,
    }
    low_clip_values: list[float] = []
    high_clip_values: list[float] = []
    contrast_values: list[float] = []
    dynamic_values: list[float] = []
    p50_values: list[float] = []
    dark_frame_mask: list[bool] = []
    exposure_bad_values: list[bool] = []

    for i, frame in enumerate(rgb_frames):
        sample_i = sample_indices[i] if i < len(sample_indices) else i
        t_ms = sample_times_ms[i] if i < len(sample_times_ms) else float(i)
        blur_score: float | None = None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_roi, blur_margin_used = _blur_roi(gray, thresholds.blur_roi_margin_ratio)
            blur_score = float(cv2.Laplacian(blur_roi, cv2.CV_64F).var())
            gray_roi, margin_used = _exposure_roi(gray, thresholds.exposure_roi_margin_ratio)
            bgr_roi, _ = _exposure_roi(frame, thresholds.exposure_roi_margin_ratio)

            low_clip = float(np.mean(gray_roi <= thresholds.low_clip_pixel_value))
            high_clip_luma = float(
                np.mean(gray_roi >= thresholds.high_clip_pixel_value)
            )
            # Channel-aware clipping catches strongly tinted highlight saturation
            # cases that grayscale-only clipping can miss.
            high_clip_any_channel = float(
                np.mean(np.max(bgr_roi, axis=2) >= thresholds.high_clip_pixel_value)
            )
            high_clip = max(high_clip_luma, high_clip_any_channel)
            p01 = float(np.percentile(gray_roi, 1))
            p05 = float(np.percentile(gray_roi, 5))
            p50 = float(np.percentile(gray_roi, 50))
            p95 = float(np.percentile(gray_roi, 95))
            p99 = float(np.percentile(gray_roi, 99))
            contrast = p99 - p01
            dynamic_range = p95 - p05

            bad_flat_and_dark = (
                dynamic_range < thresholds.dynamic_range_min
                and p50 < thresholds.median_dark
            )
            bad_flat_and_bright = (
                dynamic_range < thresholds.dynamic_range_min
                and p50 > thresholds.median_bright
            )
            bad_saturation_dark = (
                low_clip > thresholds.low_clip_warn
                and p50 < thresholds.median_dark
                and p95 < thresholds.low_clip_p95_max
            )
            bad_saturation_bright = (
                high_clip > thresholds.high_clip_warn
                and p50 > thresholds.median_bright
            )
            bad = (
                bad_flat_and_dark
                or bad_flat_and_bright
                or bad_saturation_dark
                or bad_saturation_bright
            )
            reasons: list[str] = []
            if bad_saturation_dark:
                reasons.append("low_clip")
                reason_counts["low_clip"] += 1
            if bad_saturation_bright:
                reasons.append("high_clip")
                reason_counts["high_clip"] += 1
            if bad_flat_and_dark:
                reasons.append("flat_and_dark")
                reason_counts["flat_and_dark"] += 1
            if bad_flat_and_bright:
                reasons.append("flat_and_bright")
                reason_counts["flat_and_bright"] += 1

            exposure_ok[i] = not bad
            exposure_bad_values.append(bad)
            low_clip_values.append(low_clip)
            high_clip_values.append(high_clip)
            contrast_values.append(contrast)
            dynamic_values.append(dynamic_range)
            p50_values.append(p50)
            dark_frame_mask.append(p50 < thresholds.median_dark)
            exposure_rows.append(
                {
                    "sample_i": int(sample_i),
                    "t_ms": float(t_ms),
                    "roi_margin_ratio": float(margin_used),
                    "blur_roi_margin_ratio": float(blur_margin_used),
                    "blur_value": blur_score,
                    "low_clip": low_clip,
                    "high_clip": high_clip,
                    "high_clip_luma": high_clip_luma,
                    "high_clip_any_channel": high_clip_any_channel,
                    "p01": p01,
                    "p05": p05,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "contrast": contrast,
                    "dynamic_range": dynamic_range,
                    "exposure_bad": int(bad),
                    "reasons": ";".join(reasons),
                }
            )
        except Exception as exc:
            exposure_ok[i] = True
            exposure_compute_errors.append(
                {
                    "sample_i": int(sample_i),
                    "t_ms": float(t_ms),
                    "message": str(exc),
                }
            )
        blur_scores.append(blur_score)

    valid_blur_scores = [x for x in blur_scores if x is not None]
    if valid_blur_scores:
        blur_arr = np.asarray(valid_blur_scores, dtype=np.float64)
        blur_threshold = float(thresholds.blur_threshold_min)
        blur_bad = [x < blur_threshold for x in valid_blur_scores]
        blur_fail_ratio = float(np.mean(blur_bad))
        blur_median = float(np.median(blur_arr))
        blur_p10 = float(np.percentile(blur_arr, 10))
        blur_p50 = float(np.percentile(blur_arr, 50))
        blur_p90 = float(np.percentile(blur_arr, 90))
    else:
        blur_threshold = None
        blur_fail_ratio = None
        blur_median = None
        blur_p10 = None
        blur_p50 = None
        blur_p90 = None
    blur_ok: list[bool] = []
    for score in blur_scores:
        if score is None or blur_threshold is None:
            blur_ok.append(True)
        else:
            blur_ok.append(score >= blur_threshold)

    exposure_bad_sample_indices = [
        int(row["sample_i"])
        for row in exposure_rows
        if int(row.get("exposure_bad", 0)) == 1
    ]

    metrics = {
        "blur_median": blur_median,
        "blur_threshold": blur_threshold,
        "blur_fail_ratio": blur_fail_ratio,
        "blur_p10": blur_p10,
        "blur_p50": blur_p50,
        "blur_p90": blur_p90,
        "blur_valid_frame_count": len(valid_blur_scores),
        "exposure_bad_ratio": (
            float(np.mean(exposure_bad_values)) if exposure_bad_values else None
        ),
        "exposure_valid_frame_count": len(exposure_bad_values),
        "exposure_bad_first_sample_i": (
            min(exposure_bad_sample_indices) if exposure_bad_sample_indices else None
        ),
        "exposure_bad_last_sample_i": (
            max(exposure_bad_sample_indices) if exposure_bad_sample_indices else None
        ),
        "low_clip_mean": (
            float(np.mean(low_clip_values)) if low_clip_values else None
        ),
        "low_clip_p95": (
            float(np.percentile(low_clip_values, 95)) if low_clip_values else None
        ),
        "high_clip_mean": (
            float(np.mean(high_clip_values)) if high_clip_values else None
        ),
        "high_clip_p95": (
            float(np.percentile(high_clip_values, 95)) if high_clip_values else None
        ),
        "contrast_mean": (
            float(np.mean(contrast_values)) if contrast_values else None
        ),
        "contrast_p05": (
            float(np.percentile(contrast_values, 5)) if contrast_values else None
        ),
        "dynamic_range_mean": (
            float(np.mean(dynamic_values)) if dynamic_values else None
        ),
        "dynamic_range_p05": (
            float(np.percentile(dynamic_values, 5)) if dynamic_values else None
        ),
        "p50_mean": float(np.mean(p50_values)) if p50_values else None,
        "p50_p05": (
            float(np.percentile(p50_values, 5)) if p50_values else None
        ),
        "p50_p95": (
            float(np.percentile(p50_values, 95)) if p50_values else None
        ),
        "dark_frame_ratio": (
            float(np.mean(dark_frame_mask)) if dark_frame_mask else None
        ),
        "low_clip_when_dark_mean": (
            float(np.mean([v for v, is_dark in zip(low_clip_values, dark_frame_mask) if is_dark]))
            if any(dark_frame_mask)
            else None
        ),
        "exposure_bad_reason_counts": reason_counts,
    }
    return metrics, blur_ok, exposure_ok, exposure_rows, exposure_compute_errors


def compute_depth_pixel_metrics(
    depth_frames: list[np.ndarray],
    thresholds: ThresholdsConfig,
) -> tuple[dict[str, float | None], list[bool]]:
    if not depth_frames:
        return (
            {
                "depth_invalid_mean": None,
                "depth_invalid_p95": None,
                "depth_fail_ratio": None,
            },
            [],
        )
    invalid_ratios: list[float] = []
    for depth in depth_frames:
        invalid_ratios.append(float(np.mean(depth == 0)))
    arr = np.asarray(invalid_ratios, dtype=np.float64)
    fail_mask = arr > thresholds.depth_invalid_threshold
    metrics = {
        "depth_invalid_mean": float(np.mean(arr)),
        "depth_invalid_p95": float(np.percentile(arr, 95)),
        "depth_fail_ratio": float(np.mean(fail_mask)),
    }
    depth_ok = [not bool(x) for x in fail_mask]
    return metrics, depth_ok


def _exposure_roi(frame: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    margin = int(min(h, w) * margin_ratio)
    y0 = margin
    y1 = h - margin
    x0 = margin
    x1 = w - margin
    if y1 <= y0 or x1 <= x0:
        return frame, 0.0
    return frame[y0:y1, x0:x1], float(margin_ratio)


def _blur_roi(gray: np.ndarray, margin_ratio: float) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    margin = int(min(h, w) * margin_ratio)
    y0 = margin
    y1 = h - margin
    x0 = margin
    x1 = w - margin
    if y1 <= y0 or x1 <= x0:
        return gray, 0.0
    return gray[y0:y1, x0:x1], float(margin_ratio)
