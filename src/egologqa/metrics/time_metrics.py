from __future__ import annotations

from typing import Optional

import numpy as np


def compute_out_of_order_ratio(times_ms: list[float]) -> tuple[int, float]:
    if len(times_ms) < 2:
        return 0, 0.0
    arr = np.asarray(times_ms, dtype=np.float64)
    inversions = int(np.sum(arr[1:] < arr[:-1]))
    ratio = inversions / max(1, len(times_ms) - 1)
    return inversions, float(ratio)


def compute_stream_gaps(
    times_ms: list[float], gap_factor: float
) -> dict[str, float | int | list[tuple[float, float]] | None]:
    if len(times_ms) < 2:
        return {
            "expected_dt_ms": None,
            "gap_count": 0,
            "gap_ratio": 0.0,
            "gap_intervals_ms": [],
        }
    arr = np.asarray(times_ms, dtype=np.float64)
    dt = np.diff(arr)
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return {
            "expected_dt_ms": None,
            "gap_count": 0,
            "gap_ratio": 0.0,
            "gap_intervals_ms": [],
        }
    p10 = float(np.percentile(positive_dt, 10))
    p90 = float(np.percentile(positive_dt, 90))
    inner = positive_dt[(positive_dt >= p10) & (positive_dt <= p90)]
    expected_dt = float(np.median(inner if len(inner) > 0 else positive_dt))
    if expected_dt <= 0:
        gap_mask = np.zeros(len(dt), dtype=bool)
    else:
        gap_mask = dt > (gap_factor * expected_dt)
    intervals: list[tuple[float, float]] = []
    for i, is_gap in enumerate(gap_mask, start=1):
        if is_gap:
            intervals.append((float(arr[i - 1]), float(arr[i])))
    gap_count = int(np.sum(gap_mask))
    return {
        "expected_dt_ms": expected_dt,
        "gap_count": gap_count,
        "gap_ratio": gap_count / max(1, len(times_ms) - 1),
        "gap_intervals_ms": intervals,
    }


def compute_sync_metrics(
    rgb_times_ms: list[float],
    depth_times_ms_for_index: list[float],
    sync_fail_ms: float,
) -> dict[str, float | None]:
    if len(rgb_times_ms) == 0 or len(depth_times_ms_for_index) == 0:
        return {
            "sync_p50_ms": None,
            "sync_p95_ms": None,
            "sync_max_ms": None,
            "sync_fail_ratio": None,
        }
    depth_sorted = np.asarray(depth_times_ms_for_index, dtype=np.float64)
    rgb_arr = np.asarray(rgb_times_ms, dtype=np.float64)
    deltas = nearest_abs_delta(rgb_arr, depth_sorted)
    return {
        "sync_p50_ms": float(np.percentile(deltas, 50)),
        "sync_p95_ms": float(np.percentile(deltas, 95)),
        "sync_max_ms": float(np.max(deltas)),
        "sync_fail_ratio": float(np.mean(deltas > sync_fail_ms)),
    }


def nearest_abs_delta(query_times_ms: np.ndarray, sorted_ref_times_ms: np.ndarray) -> np.ndarray:
    if len(sorted_ref_times_ms) == 0:
        return np.array([], dtype=np.float64)
    pos = np.searchsorted(sorted_ref_times_ms, query_times_ms, side="left")
    right_idx = np.clip(pos, 0, len(sorted_ref_times_ms) - 1)
    left_idx = np.clip(pos - 1, 0, len(sorted_ref_times_ms) - 1)
    right_delta = np.abs(sorted_ref_times_ms[right_idx] - query_times_ms)
    left_delta = np.abs(sorted_ref_times_ms[left_idx] - query_times_ms)
    return np.minimum(left_delta, right_delta)


def nearest_indices(query_times_ms: np.ndarray, sorted_ref_times_ms: np.ndarray) -> np.ndarray:
    if len(sorted_ref_times_ms) == 0:
        return np.array([], dtype=np.int64)
    pos = np.searchsorted(sorted_ref_times_ms, query_times_ms, side="left")
    right_idx = np.clip(pos, 0, len(sorted_ref_times_ms) - 1)
    left_idx = np.clip(pos - 1, 0, len(sorted_ref_times_ms) - 1)
    right_delta = np.abs(sorted_ref_times_ms[right_idx] - query_times_ms)
    left_delta = np.abs(sorted_ref_times_ms[left_idx] - query_times_ms)
    choose_left = left_delta <= right_delta
    nearest = np.where(choose_left, left_idx, right_idx)
    return nearest.astype(np.int64)


def compute_imu_coverage(
    rgb_times_ms: list[float],
    imu_times_ms: list[float],
    window_ms: float,
) -> list[bool]:
    if len(rgb_times_ms) == 0:
        return []
    if len(imu_times_ms) == 0:
        return [False] * len(rgb_times_ms)
    imu_sorted = np.asarray(sorted(imu_times_ms), dtype=np.float64)
    coverage: list[bool] = []
    for t in rgb_times_ms:
        left = np.searchsorted(imu_sorted, t - window_ms, side="left")
        right = np.searchsorted(imu_sorted, t + window_ms, side="right")
        coverage.append(right > left)
    return coverage
