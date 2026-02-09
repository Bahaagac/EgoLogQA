from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np

from egologqa.artifacts import (
    write_blur_debug_csv,
    write_blur_annotated_evidence_frames,
    write_blur_evidence_frames,
    write_benchmarks_json,
    write_clean_segments_json,
    write_depth_debug_csv,
    write_drop_timeline,
    write_evidence_manifest_json,
    write_exposure_evidence_error,
    write_exposure_evidence_frames,
    write_exposure_debug_csv,
    write_report_markdown,
    write_rgb_previews,
    select_exposure_evidence_rows,
    select_blur_evidence_rows,
    write_sync_histogram,
)
from egologqa.config import apply_topic_overrides, config_to_dict
from egologqa.decoders.depth import decode_depth_message
from egologqa.decoders.rgb import decode_rgb_message
from egologqa.drop_regions import DropRegions
from egologqa.frame_flags import build_frame_flags
from egologqa.gate import classify_sync_pattern, evaluate_gate, sync_offset_estimate_ms
from egologqa.io.reader import MCapMessageSource, MessageSource
from egologqa.metrics.pixel_metrics import compute_depth_pixel_metrics, compute_rgb_pixel_metrics
from egologqa.metrics.time_metrics import (
    compute_sync_diagnostics,
    compute_imu_coverage,
    compute_stream_gaps,
    compute_sync_metrics,
    nearest_abs_delta,
    nearest_indices,
)
from egologqa.models import AnalysisResult, QAConfig, ThresholdsConfig, TopicOverrides
from egologqa.report import empty_report, git_commit_or_unknown, write_report_json
from egologqa.sampling import sample_rgb_indices
from egologqa.segments import extract_segments
from egologqa.topic_select import select_topics
from egologqa.time import extract_header_stamp_ns, extract_stamp_ns


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class StreamCollector:
    name: str
    times_ns: list[int] = field(default_factory=list)
    times_ms: list[float] = field(default_factory=list)
    total_messages: int = 0
    invalid_timestamp_messages: int = 0
    out_of_order_count: int = 0
    inversion_indices: list[int] = field(default_factory=list)
    _last_ns: Optional[int] = None

    def add_timestamp(self, t_ns: int) -> None:
        self.total_messages += 1
        if t_ns <= 0:
            self.invalid_timestamp_messages += 1
            return
        if self._last_ns is not None and t_ns < self._last_ns:
            self.out_of_order_count += 1
            self.inversion_indices.append(len(self.times_ns))
        self._last_ns = t_ns
        self.times_ns.append(t_ns)
        self.times_ms.append(t_ns / 1_000_000.0)

    @property
    def has_timestamps(self) -> bool:
        return len(self.times_ns) >= 2

    @property
    def out_of_order_ratio(self) -> float:
        if len(self.times_ns) < 2:
            return 0.0
        return self.out_of_order_count / max(1, len(self.times_ns) - 1)


def analyze_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: QAConfig,
    overrides: TopicOverrides | None = None,
    source: MessageSource | None = None,
    progress_cb: ProgressCallback | None = None,
) -> AnalysisResult:
    input_path = str(input_path)
    output_dir = Path(output_dir)
    cfg = apply_topic_overrides(config, overrides)
    file_size = Path(input_path).stat().st_size if Path(input_path).exists() else None
    report = empty_report(
        input_path=input_path,
        file_size_bytes=file_size,
        commit=git_commit_or_unknown(Path.cwd()),
    )
    report["config_used"] = config_to_dict(cfg)
    errors: list[dict[str, Any]] = []
    _append_legacy_exposure_keys_warning(cfg, errors)
    bench_enabled = bool(cfg.debug.benchmarks_enabled)
    phase_durations_s: dict[str, float] = {
        "scan": 0.0,
        "pass1": 0.0,
        "pass2": 0.0,
        "artifacts": 0.0,
        "report_write": 0.0,
    }
    total_start = perf_counter()

    def _timed_artifact_write(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        out = fn(*args, **kwargs)
        phase_durations_s["artifacts"] += perf_counter() - start
        return out

    _emit(progress_cb, "scan", 0.05, "Scanning topics")

    try:
        scan_start = perf_counter()
        source_obj: MessageSource = source if source is not None else MCapMessageSource(input_path)
        topic_stats = source_obj.scan_topics()
        selected = select_topics(cfg, topic_stats)
        report["streams"]["rgb_topic"] = selected.rgb_topic
        report["streams"]["depth_topic"] = selected.depth_topic
        report["streams"]["imu_accel_topic"] = selected.imu_accel_topic
        report["streams"]["imu_gyro_topic"] = selected.imu_gyro_topic
        report["streams"]["imu_mode"] = selected.imu_mode
        report["streams"]["topic_stats"] = {}
        for topic in [
            selected.rgb_topic,
            selected.depth_topic,
            selected.imu_accel_topic,
            selected.imu_gyro_topic,
        ]:
            if topic is None or topic in report["streams"]["topic_stats"]:
                continue
            info = topic_stats.get(topic)
            report["streams"]["topic_stats"][topic] = {
                "message_count": info.message_count if info else 0,
                "approx_rate_hz": info.approx_rate_hz if info else None,
                "duration_s": info.duration_s if info else 0.0,
            }

        stream_map: dict[str, list[StreamCollector]] = {}
        rgb_col = StreamCollector("rgb")
        depth_col = StreamCollector("depth")
        imu_acc_col = StreamCollector("imu_accel")
        imu_gyro_col = StreamCollector("imu_gyro")

        def bind(topic: str | None, collector: StreamCollector) -> None:
            if topic is None:
                return
            stream_map.setdefault(topic, []).append(collector)

        bind(selected.rgb_topic, rgb_col)
        bind(selected.depth_topic, depth_col)
        bind(selected.imu_accel_topic, imu_acc_col)
        bind(selected.imu_gyro_topic, imu_gyro_col)
        selected_topics = set(stream_map.keys())
        phase_durations_s["scan"] = perf_counter() - scan_start

        rgb_total_messages_seen = 0
        rgb_header_valid_count = 0
        rgb_timebase_diffs_ms: list[float] = []

        # Pass 1: timestamps only.
        pass1_start = perf_counter()
        _emit(progress_cb, "pass1", 0.12, "Pass 1: collecting timestamps")
        for rec in source_obj.iter_messages(topics=selected_topics):
            collectors = stream_map.get(rec.topic, [])
            if not collectors:
                continue
            if rec.topic == selected.rgb_topic:
                rgb_total_messages_seen += 1
                header_ns = extract_header_stamp_ns(rec.msg)
                if header_ns > 0:
                    rgb_header_valid_count += 1
                    if rec.log_time_ns > 0:
                        rgb_timebase_diffs_ms.append((header_ns - rec.log_time_ns) / 1_000_000.0)
            t_ns, _, _ = extract_stamp_ns(rec.msg, rec.log_time_ns)
            for col in collectors:
                col.add_timestamp(t_ns)

        report["streams"]["rgb_timestamps_present"] = rgb_col.has_timestamps
        depth_topic_present = selected.depth_topic is not None and (
            topic_stats.get(selected.depth_topic, None) is not None
            and topic_stats[selected.depth_topic].message_count > 0
        )
        depth_timestamps_present = depth_col.has_timestamps
        report["streams"]["depth_topic_present"] = depth_topic_present
        report["streams"]["depth_timestamps_present"] = depth_timestamps_present
        report["streams"]["decode_status"]["depth_timestamps"] = (
            "present" if depth_timestamps_present else "missing"
        )

        for col in [rgb_col, depth_col, imu_acc_col, imu_gyro_col]:
            if col.total_messages > 0 and not col.has_timestamps:
                errors.append(
                    _error(
                        "WARN",
                        "STREAM_TIMESTAMPS_MISSING",
                        f"Stream '{col.name}' has fewer than 2 valid timestamps.",
                        {
                            "stream": col.name,
                            "total_messages": col.total_messages,
                            "valid_timestamps": len(col.times_ns),
                        },
                    )
                )

        analyzed_duration_s = _duration_seconds(rgb_col.times_ns)
        report["time"]["duration_s"] = analyzed_duration_s
        report["sampling"]["rgb_stride"] = cfg.sampling.rgb_stride
        report["sampling"]["max_rgb_frames"] = cfg.sampling.max_rgb_frames

        out_of_order = {
            "rgb": {
                "count": rgb_col.out_of_order_count,
                "ratio": rgb_col.out_of_order_ratio,
            },
            "depth": {
                "count": depth_col.out_of_order_count,
                "ratio": depth_col.out_of_order_ratio,
            },
            "imu_accel": {
                "count": imu_acc_col.out_of_order_count,
                "ratio": imu_acc_col.out_of_order_ratio,
            },
            "imu_gyro": {
                "count": imu_gyro_col.out_of_order_count,
                "ratio": imu_gyro_col.out_of_order_ratio,
            },
        }
        report["metrics"]["out_of_order"] = out_of_order

        high_ooo_streams = []
        for name, payload in out_of_order.items():
            if payload["ratio"] > cfg.integrity.out_of_order_warn_ratio:
                high_ooo_streams.append(name)
                errors.append(
                    _error(
                        "WARN",
                        "TIMESTAMP_OUT_OF_ORDER_HIGH",
                        "Out-of-order timestamp ratio exceeded threshold.",
                        {
                            "stream": name,
                            "ratio": payload["ratio"],
                            "threshold": cfg.integrity.out_of_order_warn_ratio,
                        },
                    )
                )

        rgb_gap = compute_stream_gaps(rgb_col.times_ms, cfg.thresholds.image_gap_factor)
        report["metrics"]["expected_rgb_dt_ms"] = rgb_gap["expected_dt_ms"]
        report["metrics"]["drop_ratio"] = rgb_gap["gap_ratio"]
        drop_regions = DropRegions(rgb_gap["gap_intervals_ms"])

        # Alignment metrics. Sorting is allowed only for alignment tasks.
        sync_rgb_times = rgb_col.times_ms
        if "rgb" in high_ooo_streams:
            sync_rgb_times = sorted(sync_rgb_times)
        sync_depth_times = depth_col.times_ms
        if "depth" in high_ooo_streams:
            sync_depth_times = sorted(sync_depth_times)
        else:
            sync_depth_times = sorted(sync_depth_times)
        sync_metrics = compute_sync_metrics(
            rgb_times_ms=sync_rgb_times,
            depth_times_ms_for_index=sync_depth_times,
            sync_fail_ms=cfg.thresholds.sync_fail_ms,
        )
        report["metrics"].update(sync_metrics)
        sync_diag_metrics = compute_sync_diagnostics(
            rgb_times_ms=sync_rgb_times,
            depth_times_ms_for_index=sync_depth_times,
        )
        if sync_diag_metrics:
            report["metrics"].update(sync_diag_metrics)

        if rgb_timebase_diffs_ms:
            diffs_arr = np.asarray(rgb_timebase_diffs_ms, dtype=np.float64)
            abs_arr = np.abs(diffs_arr)
            report["metrics"]["rgb_timebase_diff_signed_p50_ms"] = _percentile50(diffs_arr)
            report["metrics"]["rgb_timebase_diff_signed_mean_ms"] = float(np.mean(diffs_arr))
            report["metrics"]["rgb_timebase_diff_abs_p95_ms"] = _percentile95(abs_arr)
            report["metrics"]["rgb_timebase_diff_abs_max_ms"] = float(np.max(abs_arr))
            report["metrics"]["rgb_timebase_diff_sample_count"] = int(diffs_arr.size)
            report["metrics"]["rgb_timebase_header_present_ratio"] = (
                float(rgb_header_valid_count / rgb_total_messages_seen)
                if rgb_total_messages_seen > 0
                else None
            )

        imu_acc_cov = None
        imu_gyro_cov = None
        imu_acc_missing_ratio = None
        imu_gyro_missing_ratio = None
        imu_combined_missing_ratio = None
        if selected.imu_mode != "none" and rgb_col.times_ms:
            imu_acc_cov = compute_imu_coverage(
                rgb_col.times_ms, imu_acc_col.times_ms, cfg.thresholds.imu_window_ms
            )
            imu_gyro_cov = compute_imu_coverage(
                rgb_col.times_ms, imu_gyro_col.times_ms, cfg.thresholds.imu_window_ms
            )
            imu_acc_missing_ratio = float(np.mean(np.logical_not(imu_acc_cov)))
            imu_gyro_missing_ratio = float(np.mean(np.logical_not(imu_gyro_cov)))
            combined = [not (a and g) for a, g in zip(imu_acc_cov, imu_gyro_cov)]
            imu_combined_missing_ratio = float(np.mean(combined))
        report["metrics"]["imu_accel_missing_ratio"] = imu_acc_missing_ratio
        report["metrics"]["imu_gyro_missing_ratio"] = imu_gyro_missing_ratio
        report["metrics"]["imu_combined_missing_ratio"] = imu_combined_missing_ratio

        sampled_rgb_indices = sample_rgb_indices(
            total_frames=len(rgb_col.times_ns),
            stride=cfg.sampling.rgb_stride,
            max_frames=cfg.sampling.max_rgb_frames,
        )
        report["sampling"]["frames_analyzed"] = len(sampled_rgb_indices)
        sampled_rgb_times_ns = [rgb_col.times_ns[i] for i in sampled_rgb_indices]
        sampled_rgb_times_ms = [rgb_col.times_ms[i] for i in sampled_rgb_indices]

        # Build inversion-adjacent force-break positions for sampled frames.
        force_bad_sample_positions: set[int] = set()
        if "rgb" in high_ooo_streams:
            sampled_arr = np.asarray(sampled_rgb_indices, dtype=np.int64)
            for inv_idx in rgb_col.inversion_indices:
                if sampled_arr.size == 0:
                    break
                for candidate in (max(0, inv_idx - 1), inv_idx):
                    ins = int(np.searchsorted(sampled_arr, candidate, side="left"))
                    for cpos in (ins - 1, ins):
                        if 0 <= cpos < sampled_arr.size:
                            force_bad_sample_positions.add(int(cpos))

        # Compute nearest depth indices for sampled RGB frames.
        depth_index_by_sample_pos: dict[int, int] = {}
        sampled_sync_deltas: list[float] | None = None
        if depth_col.times_ms and sampled_rgb_times_ms:
            depth_arr = np.asarray(depth_col.times_ms, dtype=np.float64)
            depth_order = np.argsort(depth_arr)
            depth_sorted = depth_arr[depth_order]
            sampled_arr = np.asarray(sampled_rgb_times_ms, dtype=np.float64)
            nearest_sorted_idx = nearest_indices(sampled_arr, depth_sorted)
            nearest_depth_idx = depth_order[nearest_sorted_idx]
            deltas = nearest_abs_delta(sampled_arr, depth_sorted)
            sampled_sync_deltas = deltas.tolist()
            for pos, depth_idx in enumerate(nearest_depth_idx.tolist()):
                depth_index_by_sample_pos[pos] = depth_idx

        # Pass 2: pixel metrics on sampled frames.
        phase_durations_s["pass1"] = perf_counter() - pass1_start
        pass2_start = perf_counter()
        _emit(progress_cb, "pass2", 0.55, "Pass 2: decoding sampled frames")
        rgb_frames_by_pos: dict[int, Any] = {}
        depth_frames_by_pos: dict[int, Any] = {}
        rgb_decode_errors: dict[str, int] = {}
        depth_decode_errors: dict[str, int] = {}
        rgb_decode_error_by_pos: dict[int, str] = {}
        depth_decode_error_by_pos: dict[int, str] = {}
        depth_non_uint16_positions: list[int] = []
        rgb_targets = set(sampled_rgb_indices)
        sample_pos_by_rgb_idx: dict[int, list[int]] = {}
        for pos, idx in enumerate(sampled_rgb_indices):
            sample_pos_by_rgb_idx.setdefault(idx, []).append(pos)
        depth_targets = set(depth_index_by_sample_pos.values())
        sample_pos_by_depth_idx: dict[int, list[int]] = {}
        for pos, idx in depth_index_by_sample_pos.items():
            sample_pos_by_depth_idx.setdefault(idx, []).append(pos)

        rgb_valid_idx = -1
        depth_valid_idx = -1
        pixel_topics = {t for t in [selected.rgb_topic, selected.depth_topic] if t}
        for rec in source_obj.iter_messages(topics=pixel_topics):
            if rec.topic == selected.rgb_topic:
                t_ns, _, _ = extract_stamp_ns(rec.msg, rec.log_time_ns)
                if t_ns > 0:
                    rgb_valid_idx += 1
                    if rgb_valid_idx in rgb_targets:
                        frame, err = decode_rgb_message(rec.msg)
                        if err is None and frame is not None:
                            for pos in sample_pos_by_rgb_idx.get(rgb_valid_idx, []):
                                rgb_frames_by_pos[pos] = frame
                        else:
                            code = err or "RGB_DECODE_FAIL"
                            rgb_decode_errors[code] = rgb_decode_errors.get(code, 0) + 1
                            for pos in sample_pos_by_rgb_idx.get(rgb_valid_idx, []):
                                rgb_decode_error_by_pos[pos] = code
            elif rec.topic == selected.depth_topic:
                t_ns, _, _ = extract_stamp_ns(rec.msg, rec.log_time_ns)
                if t_ns > 0:
                    depth_valid_idx += 1
                    if depth_valid_idx in depth_targets:
                        frame, err = decode_depth_message(rec.msg)
                        if err is None and frame is not None:
                            for pos in sample_pos_by_depth_idx.get(depth_valid_idx, []):
                                depth_frames_by_pos[pos] = frame
                        else:
                            code = err or "DEPTH_PNG_IMDECODE_FAIL"
                            depth_decode_errors[code] = depth_decode_errors.get(code, 0) + 1
                            for pos in sample_pos_by_depth_idx.get(depth_valid_idx, []):
                                depth_decode_error_by_pos[pos] = code
                                if code == "DEPTH_UNEXPECTED_DTYPE":
                                    depth_non_uint16_positions.append(pos)

        sampled_count = len(sampled_rgb_indices)
        decoded_rgb_positions = sorted(rgb_frames_by_pos.keys())
        decoded_rgb_frames = [rgb_frames_by_pos[pos] for pos in decoded_rgb_positions]
        decoded_rgb_times_ms = [sampled_rgb_times_ms[pos] for pos in decoded_rgb_positions]
        decoded_depth_positions = sorted(depth_frames_by_pos.keys())
        decoded_depth_frames = [depth_frames_by_pos[pos] for pos in decoded_depth_positions]

        rgb_metrics, blur_ok_partial, exposure_ok_partial, exposure_rows, exposure_compute_errors = compute_rgb_pixel_metrics(
            decoded_rgb_frames,
            cfg.thresholds,
            sample_indices=decoded_rgb_positions,
            sample_times_ms=decoded_rgb_times_ms,
        )
        depth_metrics, depth_ok_partial = compute_depth_pixel_metrics(
            decoded_depth_frames, cfg.thresholds
        )

        blur_ok = [True] * sampled_count
        exposure_ok = [True] * sampled_count
        depth_ok = [True] * sampled_count
        for idx, pos in enumerate(decoded_rgb_positions):
            blur_ok[pos] = blur_ok_partial[idx]
            exposure_ok[pos] = exposure_ok_partial[idx]
        for idx, pos in enumerate(decoded_depth_positions):
            depth_ok[pos] = depth_ok_partial[idx]

        report["metrics"].update(rgb_metrics)
        report["metrics"].update(depth_metrics)
        if "blur_valid_frame_count" not in rgb_metrics:
            report["metrics"]["blur_valid_frame_count"] = len(decoded_rgb_frames)
        if "exposure_valid_frame_count" not in rgb_metrics:
            report["metrics"]["exposure_valid_frame_count"] = len(exposure_rows)

        rgb_decode_attempt_count = sampled_count
        rgb_decode_success_count = len(decoded_rgb_positions)
        depth_decode_attempt_count = len(depth_index_by_sample_pos)
        depth_decode_success_count = len(decoded_depth_positions)

        report["metrics"]["rgb_decode_attempt_count"] = rgb_decode_attempt_count
        report["metrics"]["rgb_decode_success_count"] = rgb_decode_success_count
        report["metrics"]["depth_decode_attempt_count"] = depth_decode_attempt_count
        report["metrics"]["depth_decode_success_count"] = depth_decode_success_count
        report["metrics"]["depth_valid_frame_count"] = len(decoded_depth_frames)

        rgb_decode_supported = len(decoded_rgb_positions) > 0
        depth_decode_supported = len(decoded_depth_positions) > 0
        report["streams"]["decode_status"]["rgb_pixels"] = (
            "supported" if rgb_decode_supported else "unsupported"
        )
        report["streams"]["decode_status"]["depth_pixels"] = (
            "supported" if depth_decode_supported else "unsupported"
        )

        if not rgb_decode_supported:
            code = _max_error_code(rgb_decode_errors, "RGB_DECODE_FAIL")
            errors.append(
                _error(
                    "WARN",
                    code,
                    "RGB pixel decode unavailable for sampled frames.",
                    {"error_counts": rgb_decode_errors},
                )
            )
        if not depth_decode_supported and depth_col.times_ns:
            code = _max_error_code(depth_decode_errors, "DEPTH_PNG_IMDECODE_FAIL")
            errors.append(
                _error(
                    "WARN",
                    code,
                    "Depth pixel decode unavailable for sampled frames.",
                    {"error_counts": depth_decode_errors},
                )
            )
        if depth_non_uint16_positions:
            errors.append(
                _error(
                    "WARN",
                    "DEPTH_DTYPE_NON_UINT16_SEEN",
                    "Decoded depth samples included non-uint16 dtype.",
                    {
                        "count": len(depth_non_uint16_positions),
                        "first_sample_i": min(depth_non_uint16_positions),
                    },
                )
            )
        for compute_error in exposure_compute_errors:
            errors.append(
                _error(
                    "ERROR",
                    "EXPOSURE_COMPUTE_FAILED",
                    "Exposure feature computation failed for a decoded RGB frame.",
                    compute_error,
                )
            )
        if report["metrics"].get("blur_valid_frame_count", 0) == 0:
            errors.append(
                _error(
                    "WARN",
                    "BLUR_UNAVAILABLE_NO_DECODE",
                    "Blur metrics unavailable because no valid decoded RGB frames were available.",
                    {
                        "rgb_decode_attempt_count": rgb_decode_attempt_count,
                        "rgb_decode_success_count": rgb_decode_success_count,
                    },
                )
            )

        if cfg.debug.export_exposure_csv and rgb_decode_supported:
            if exposure_rows:
                exposure_csv_path = _timed_artifact_write(
                    write_exposure_debug_csv, exposure_rows, output_dir
                )
                report["metrics"]["exposure_debug_csv_path"] = _relative_path(
                    exposure_csv_path, output_dir
                )
                if report["metrics"]["exposure_debug_csv_path"] is None:
                    errors.append(
                        _error(
                            "WARN",
                            "RGB_EXPOSURE_DEBUG_UNAVAILABLE",
                            "Exposure debug CSV export failed.",
                            {"rows": len(exposure_rows)},
                        )
                    )
            else:
                errors.append(
                    _error(
                        "WARN",
                        "RGB_EXPOSURE_DEBUG_UNAVAILABLE",
                        "Exposure debug CSV could not be generated because no exposure rows were available.",
                        {
                            "rgb_decode_supported": rgb_decode_supported,
                            "decoded_rgb_frames": len(decoded_rgb_frames),
                        },
                    )
                )
                report["metrics"]["exposure_debug_csv_path"] = None

        blur_value_by_pos: dict[int, float] = {}
        blur_roi_margin_by_pos: dict[int, float] = {}
        for row in exposure_rows:
            pos = int(row.get("sample_i", -1))
            if pos < 0:
                continue
            if row.get("blur_value") is not None:
                blur_value_by_pos[pos] = float(row["blur_value"])
            if row.get("blur_roi_margin_ratio") is not None:
                blur_roi_margin_by_pos[pos] = float(row["blur_roi_margin_ratio"])

        blur_rows: list[dict[str, Any]] = []
        blur_threshold = report["metrics"].get("blur_threshold")
        for pos in range(sampled_count):
            decode_ok = pos in rgb_frames_by_pos
            blur_value = blur_value_by_pos.get(pos)
            row: dict[str, Any] = {
                "sample_i": pos,
                "t_ms": float(sampled_rgb_times_ms[pos]),
                "roi_margin_ratio": float(
                    blur_roi_margin_by_pos.get(pos, cfg.thresholds.blur_roi_margin_ratio)
                ),
                "decode_ok": int(decode_ok),
                "blur_value": blur_value,
                "blur_threshold": blur_threshold,
                "blur_ok": None,
                "decode_error_code": rgb_decode_error_by_pos.get(pos, ""),
            }
            if decode_ok and blur_value is not None and blur_threshold is not None:
                row["blur_ok"] = int(bool(blur_ok[pos]))
            blur_rows.append(row)

        depth_rows: list[dict[str, Any]] = []
        for pos in sorted(depth_index_by_sample_pos.keys()):
            decode_ok = pos in depth_frames_by_pos
            frame = depth_frames_by_pos.get(pos)
            depth_row: dict[str, Any] = {
                "sample_i": pos,
                "t_ms": float(sampled_rgb_times_ms[pos]),
                "decode_ok": int(decode_ok),
                "invalid_ratio": None,
                "min_depth": None,
                "max_depth": None,
                "dtype": "",
                "error_code": depth_decode_error_by_pos.get(pos, ""),
            }
            if decode_ok and frame is not None:
                depth_row["invalid_ratio"] = float(np.mean(frame == 0))
                depth_row["min_depth"] = int(np.min(frame))
                depth_row["max_depth"] = int(np.max(frame))
                depth_row["dtype"] = str(frame.dtype)
            depth_rows.append(depth_row)

        if cfg.debug.export_blur_csv:
            blur_csv_path = _timed_artifact_write(write_blur_debug_csv, blur_rows, output_dir)
            report["metrics"]["blur_debug_csv_path"] = _relative_path(blur_csv_path, output_dir)
            depth_csv_path = _timed_artifact_write(write_depth_debug_csv, depth_rows, output_dir)
            report["metrics"]["depth_debug_csv_path"] = _relative_path(depth_csv_path, output_dir)

        blur_fail_evidence_rows: list[dict[str, Any]] = []
        blur_pass_evidence_rows: list[dict[str, Any]] = []
        for row in blur_rows:
            if int(row.get("decode_ok", 0)) != 1:
                continue
            if row.get("blur_value") is None or row.get("blur_ok") is None:
                continue
            pos = int(row["sample_i"])
            frame = rgb_frames_by_pos.get(pos)
            if frame is None:
                continue
            evidence_row = {
                "sample_i": pos,
                "t_ms": float(row["t_ms"]),
                "blur_value": float(row["blur_value"]),
                "frame": frame,
            }
            if int(row["blur_ok"]) == 0:
                blur_fail_evidence_rows.append(evidence_row)
            else:
                blur_pass_evidence_rows.append(evidence_row)

        sampled_imu_acc_cov = (
            [imu_acc_cov[idx] for idx in sampled_rgb_indices] if imu_acc_cov is not None else None
        )
        sampled_imu_gyro_cov = (
            [imu_gyro_cov[idx] for idx in sampled_rgb_indices] if imu_gyro_cov is not None else None
        )
        sync_available_globally = depth_timestamps_present and sampled_sync_deltas is not None
        sync_sample_count = len(sampled_sync_deltas) if sampled_sync_deltas is not None else 0
        report["metrics"]["sync_sample_count"] = sync_sample_count
        if not sync_available_globally and sampled_count > 0:
            errors.append(
                _error(
                    "WARN",
                    "SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING",
                    "Sync evaluation unavailable because depth timestamps are missing.",
                    {
                        "depth_topic_present": depth_topic_present,
                        "depth_timestamps_present": depth_timestamps_present,
                        "sampled_rgb_count": sampled_count,
                    },
                )
            )
        elif sync_sample_count < cfg.thresholds.sync_min_samples:
            errors.append(
                _error(
                    "WARN",
                    "SYNC_INSUFFICIENT_SAMPLES",
                    "Sync evaluation sample count is below threshold.",
                    {
                        "sync_sample_count": sync_sample_count,
                        "sync_min_samples": cfg.thresholds.sync_min_samples,
                    },
                )
            )
        elif sampled_sync_deltas is not None and len(sampled_sync_deltas) < sampled_count:
            missing_positions = list(range(len(sampled_sync_deltas), sampled_count))
            errors.append(
                _error(
                    "WARN",
                    "SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING",
                    "Sync evaluation unavailable for a subset of sampled frames.",
                    {
                        "sampled_rgb_count": sampled_count,
                        "sync_delta_count": len(sampled_sync_deltas),
                        "missing_count": len(missing_positions),
                        "first_missing_sample_i": missing_positions[0] if missing_positions else None,
                    },
                )
            )
        frame_flags = build_frame_flags(
            sampled_rgb_times_ms=sampled_rgb_times_ms,
            sampled_rgb_indices=sampled_rgb_indices,
            sync_deltas_ms=sampled_sync_deltas,
            sync_fail_ms=cfg.thresholds.sync_fail_ms,
            sync_warn_ms=cfg.thresholds.sync_warn_ms,
            sync_available_globally=sync_available_globally,
            drop_regions=drop_regions,
            imu_accel_coverage=sampled_imu_acc_cov,
            imu_gyro_coverage=sampled_imu_gyro_cov,
            blur_ok=blur_ok if sampled_count > 0 else None,
            exposure_ok=exposure_ok if sampled_count > 0 else None,
            depth_ok=depth_ok if sampled_count > 0 else None,
            rgb_pixels_supported=rgb_decode_supported,
            depth_pixels_supported=depth_decode_supported,
            depth_timestamps_present=depth_timestamps_present,
            imu_exists=selected.imu_mode != "none",
            forced_bad_sample_positions=force_bad_sample_positions,
        )

        report["metrics"]["integrity_ok_ratio"] = (
            float(np.mean(frame_flags.frame_ok_integrity))
            if frame_flags.frame_ok_integrity
            else None
        )
        report["metrics"]["vision_ok_ratio"] = (
            float(np.mean(frame_flags.frame_ok_vision))
            if frame_flags.frame_ok_vision
            else None
        )
        if report["metrics"]["integrity_ok_ratio"] is not None:
            report["metrics"]["integrity_coverage_seconds_est"] = (
                float(report["metrics"]["integrity_ok_ratio"]) * analyzed_duration_s
            )
        else:
            report["metrics"]["integrity_coverage_seconds_est"] = None
        if report["metrics"]["vision_ok_ratio"] is not None:
            report["metrics"]["vision_coverage_seconds_est"] = (
                float(report["metrics"]["vision_ok_ratio"]) * analyzed_duration_s
            )
        else:
            report["metrics"]["vision_coverage_seconds_est"] = None
        report["metrics"]["segments_basis"] = "integrity"

        integrity_segments = extract_segments(
            sampled_times_ns=sampled_rgb_times_ns,
            frame_ok=frame_flags.frame_ok_integrity,
            max_gap_fill_ms=cfg.segments.max_gap_fill_ms,
            min_segment_seconds=cfg.segments.min_segment_seconds,
            forced_break_positions=force_bad_sample_positions,
        )
        clean_ok_strict = [
            bool(
                frame_flags.sync_ok_warn[i]
                and frame_flags.rgb_drop_ok[i]
                and frame_flags.imu_ok[i]
                and frame_flags.blur_ok[i]
                and frame_flags.exposure_ok[i]
                and frame_flags.depth_ok[i]
            )
            for i in range(sampled_count)
        ]
        clean_ok_nosync = [
            bool(
                frame_flags.rgb_drop_ok[i]
                and frame_flags.imu_ok[i]
                and frame_flags.blur_ok[i]
                and frame_flags.exposure_ok[i]
                and frame_flags.depth_ok[i]
            )
            for i in range(sampled_count)
        ]
        for forced_pos in force_bad_sample_positions:
            if 0 <= forced_pos < sampled_count:
                clean_ok_strict[forced_pos] = False
                clean_ok_nosync[forced_pos] = False

        clean_segments = extract_segments(
            sampled_times_ns=sampled_rgb_times_ns,
            frame_ok=clean_ok_strict,
            max_gap_fill_ms=cfg.segments.max_gap_fill_ms,
            min_segment_seconds=cfg.segments.min_segment_seconds,
            forced_break_positions=force_bad_sample_positions,
        )
        clean_segments_nosync = extract_segments(
            sampled_times_ns=sampled_rgb_times_ns,
            frame_ok=clean_ok_nosync,
            max_gap_fill_ms=cfg.segments.max_gap_fill_ms,
            min_segment_seconds=cfg.segments.min_segment_seconds,
            forced_break_positions=force_bad_sample_positions,
        )
        clean_segments_path = _timed_artifact_write(
            write_clean_segments_json, clean_segments, output_dir, "clean_segments.json"
        )
        clean_segments_nosync_path = _timed_artifact_write(
            write_clean_segments_json, clean_segments_nosync, output_dir, "clean_segments_nosync.json"
        )
        report["metrics"]["clean_segments_basis"] = "warn_strict_quality_mask"
        report["metrics"]["clean_segments_path"] = _relative_path(clean_segments_path, output_dir)
        report["metrics"]["clean_segments_nosync_path"] = _relative_path(
            clean_segments_nosync_path, output_dir
        )

        report["segments"] = integrity_segments
        report["errors"] = errors
        report["metrics"]["sync_pattern"] = classify_sync_pattern(report["metrics"], cfg)
        report["metrics"]["sync_offset_estimate_ms"] = sync_offset_estimate_ms(report["metrics"])

        gate = evaluate_gate(
            config=cfg,
            metrics=report["metrics"],
            streams=report["streams"],
            duration_s=analyzed_duration_s,
            segments=integrity_segments,
            errors=errors,
            clean_segments=clean_segments,
            clean_segments_nosync=clean_segments_nosync,
        )
        report["gate"] = gate

        blur_warn_trigger = "WARN_BLUR_FAIL_RATIO_GT_WARN" in gate["warn_reasons"]
        should_export_evidence = cfg.debug.export_evidence_frames or (
            cfg.debug.export_evidence_on_warn and blur_warn_trigger
        )
        blur_evidence_requested = (
            should_export_evidence
            or cfg.debug.write_annotated_evidence
            or cfg.debug.write_evidence_manifest
        )
        exposure_reason_counts = report["metrics"].get("exposure_bad_reason_counts", {})
        exposure_reason_total = 0
        if isinstance(exposure_reason_counts, dict):
            exposure_reason_total = int(sum(int(v) for v in exposure_reason_counts.values()))
        exposure_auto_trigger = exposure_reason_total > 0 and rgb_decode_success_count > 0
        exposure_evidence_requested = should_export_evidence or exposure_auto_trigger

        cv2_available = _cv2_available()
        evidence_sample_positions: set[int] = set()
        fail_dir = None
        pass_dir = None
        if blur_evidence_requested and cv2_available:
            fail_sel, pass_sel = select_blur_evidence_rows(
                blur_fail_evidence_rows, blur_pass_evidence_rows, cfg.debug.evidence_frames_k
            )
            fail_dir, pass_dir = _timed_artifact_write(
                write_blur_evidence_frames,
                blur_fail_evidence_rows,
                blur_pass_evidence_rows,
                output_dir,
                cfg.debug.evidence_frames_k,
            )
            evidence_sample_positions.update(
                int(row.get("sample_i", -1))
                for row in (fail_sel + pass_sel)
                if int(row.get("sample_i", -1)) >= 0
            )
        if blur_evidence_requested:
            report["metrics"]["blur_fail_frames_dir"] = _relative_path(fail_dir, output_dir)
            report["metrics"]["blur_pass_frames_dir"] = _relative_path(pass_dir, output_dir)

        annotated_fail_dir = None
        annotated_pass_dir = None
        if cfg.debug.write_annotated_evidence and cv2_available:
            annotated_fail_dir, annotated_pass_dir = _timed_artifact_write(
                write_blur_annotated_evidence_frames,
                blur_fail_evidence_rows,
                blur_pass_evidence_rows,
                output_dir,
                cfg.debug.evidence_frames_k,
                blur_threshold if blur_threshold is not None else None,
            )
            if annotated_fail_dir:
                report["metrics"]["blur_fail_frames_annotated_dir"] = _relative_path(
                    annotated_fail_dir, output_dir
                )
            if annotated_pass_dir:
                report["metrics"]["blur_pass_frames_annotated_dir"] = _relative_path(
                    annotated_pass_dir, output_dir
                )

        if cfg.debug.write_evidence_manifest:
            manifest_path = _timed_artifact_write(
                write_evidence_manifest_json,
                output_dir,
                blur_fail_evidence_rows,
                blur_pass_evidence_rows,
                cfg.debug.evidence_frames_k,
                blur_threshold if blur_threshold is not None else None,
                cfg.sampling.rgb_stride,
                cfg.sampling.max_rgb_frames,
                cv2_available,
                bool(annotated_fail_dir or annotated_pass_dir),
            )
            if manifest_path:
                report["metrics"]["evidence_manifest_path"] = _relative_path(
                    manifest_path, output_dir
                )

        if exposure_evidence_requested and cv2_available:
            exposure_k = cfg.debug.evidence_frames_k
            if exposure_auto_trigger and not should_export_evidence:
                exposure_k = max(cfg.thresholds.pass_exposure_evidence_k, exposure_reason_total)
            selected_by_reason, selection_warnings = select_exposure_evidence_rows(
                exposure_rows, exposure_k
            )
            for rows in selected_by_reason.values():
                evidence_sample_positions.update(
                    int(row.get("sample_i", -1)) for row in rows if int(row.get("sample_i", -1)) >= 0
                )
            exposure_dirs = _timed_artifact_write(
                write_exposure_evidence_frames,
                selected_by_reason,
                rgb_frames_by_pos,
                output_dir,
            )
            report["metrics"]["exposure_low_clip_frames_dir"] = _relative_path(
                exposure_dirs.get("low_clip"), output_dir
            )
            report["metrics"]["exposure_high_clip_frames_dir"] = _relative_path(
                exposure_dirs.get("high_clip"), output_dir
            )
            report["metrics"]["exposure_flat_and_dark_frames_dir"] = _relative_path(
                exposure_dirs.get("flat_and_dark"), output_dir
            )
            report["metrics"]["exposure_flat_and_bright_frames_dir"] = _relative_path(
                exposure_dirs.get("flat_and_bright"), output_dir
            )
            exposure_error_path = _timed_artifact_write(
                write_exposure_evidence_error, selection_warnings, output_dir
            )
            report["metrics"]["exposure_evidence_error_path"] = _relative_path(
                exposure_error_path, output_dir
            )

        pass2_total = perf_counter() - pass2_start
        phase_durations_s["pass2"] = max(0.0, pass2_total - phase_durations_s["artifacts"])

        preview_paths: list[str] = []
        if cfg.debug.export_preview_frames:
            preview_paths = _timed_artifact_write(
                write_rgb_previews,
                rgb_frames_by_pos,
                output_dir,
                12,
                evidence_sample_positions,
            )
        preview_relpaths = sorted(
            [
                rel
                for rel in (_relative_path(path, output_dir) for path in preview_paths)
                if rel is not None
            ]
        )
        report["metrics"]["preview_relpaths"] = preview_relpaths
        report["metrics"]["preview_count"] = len(preview_relpaths)
        plot_path = _timed_artifact_write(write_sync_histogram, sampled_sync_deltas, output_dir)
        drop_timeline_path = _timed_artifact_write(
            write_drop_timeline, rgb_col.times_ms, rgb_gap["gap_intervals_ms"], output_dir
        )
        if plot_path:
            report["metrics"]["sync_histogram_path"] = _relative_path(plot_path, output_dir)
        if drop_timeline_path:
            report["metrics"]["drop_timeline_path"] = _relative_path(drop_timeline_path, output_dir)
        _emit(progress_cb, "done", 1.0, f"Done ({gate['gate']})", {"gate": gate["gate"]})
    except Exception as exc:
        errors.append(
            _error(
                "ERROR",
                "ANALYSIS_EXCEPTION",
                str(exc),
                {"type": exc.__class__.__name__},
            )
        )
        report["errors"] = errors
        report["gate"]["gate"] = "FAIL"
        report["gate"]["recommended_action"] = "RECAPTURE_OR_SKIP"
        report["gate"]["fail_reasons"] = ["FAIL_ANALYSIS_ERROR"]
        report["gate"]["warn_reasons"] = []
        _emit(progress_cb, "error", 1.0, "Analysis failed", {"error": str(exc)})

    if bench_enabled:
        report["metrics"]["benchmarks_path"] = "debug/benchmarks.json"

    report_write_start = perf_counter()
    try:
        write_report_markdown(report, output_dir)
    except Exception:
        pass
    report_path = write_report_json(report, output_dir)
    phase_durations_s["report_write"] = perf_counter() - report_write_start

    if bench_enabled:
        total_s = perf_counter() - total_start
        benchmarks = {
            "schema_version": 1,
            "phase_durations_s": phase_durations_s,
            "total_s": total_s,
            "counts": {
                "frames_analyzed": report.get("sampling", {}).get("frames_analyzed", 0),
                "sync_sample_count": report.get("metrics", {}).get("sync_sample_count", 0),
                "rgb_decode_attempt_count": report.get("metrics", {}).get(
                    "rgb_decode_attempt_count", 0
                ),
                "rgb_decode_success_count": report.get("metrics", {}).get(
                    "rgb_decode_success_count", 0
                ),
                "depth_decode_attempt_count": report.get("metrics", {}).get(
                    "depth_decode_attempt_count", 0
                ),
                "depth_decode_success_count": report.get("metrics", {}).get(
                    "depth_decode_success_count", 0
                ),
            },
        }
        try:
            bench_path = _timed_artifact_write(write_benchmarks_json, benchmarks, output_dir)
            if not bench_path:
                report["metrics"].pop("benchmarks_path", None)
        except Exception:
            report["metrics"].pop("benchmarks_path", None)
        report_path = write_report_json(report, output_dir)

    return AnalysisResult(
        report=report,
        gate=report["gate"]["gate"],
        output_path=report_path,
        recommended_action=report["gate"]["recommended_action"],
        fail_reasons=list(report["gate"]["fail_reasons"]),
        warn_reasons=list(report["gate"]["warn_reasons"]),
        errors=list(report["errors"]),
        report_path=report_path,
    )


def _duration_seconds(times_ns: list[int]) -> float:
    if len(times_ns) < 2:
        return 0.0
    return (max(times_ns) - min(times_ns)) / 1_000_000_000.0


def _max_error_code(counts: dict[str, int], default: str) -> str:
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _error(severity: str, code: str, message: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "context": context,
    }


def _relative_path(path: str | None, output_dir: str | Path) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(Path(output_dir).resolve())
        return rel.as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def _percentile50(values: np.ndarray) -> float:
    try:
        return float(np.percentile(values, 50, method="linear"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 fallback
        return float(np.percentile(values, 50, interpolation="linear"))


def _percentile95(values: np.ndarray) -> float:
    try:
        return float(np.percentile(values, 95, method="linear"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 fallback
        return float(np.percentile(values, 95, interpolation="linear"))


def _append_legacy_exposure_keys_warning(
    cfg: QAConfig,
    errors: list[dict[str, Any]],
) -> None:
    defaults = ThresholdsConfig()
    ignored = {
        "low_clip_threshold": cfg.thresholds.low_clip_threshold,
        "high_clip_threshold": cfg.thresholds.high_clip_threshold,
        "contrast_min": cfg.thresholds.contrast_min,
    }
    default_ignored = {
        "low_clip_threshold": defaults.low_clip_threshold,
        "high_clip_threshold": defaults.high_clip_threshold,
        "contrast_min": defaults.contrast_min,
    }
    if ignored == default_ignored:
        return
    errors.append(
        _error(
            "WARN",
            "LEGACY_EXPOSURE_KEYS_IGNORED",
            "Legacy exposure keys are ignored by the v1.3 exposure classifier.",
            {
                "ignored_values": ignored,
                "default_values": default_ignored,
            },
        )
    )


def _emit(
    progress_cb: ProgressCallback | None,
    phase: str,
    progress: float,
    message: str,
    partial: dict[str, Any] | None = None,
) -> None:
    if progress_cb is None:
        return
    event = {
        "phase": phase,
        "progress": progress,
        "message": message,
    }
    if partial is not None:
        event["partial"] = partial
    progress_cb(event)
