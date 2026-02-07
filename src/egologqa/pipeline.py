from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from egologqa.artifacts import (
    write_drop_timeline,
    write_exposure_debug_csv,
    write_report_markdown,
    write_rgb_previews,
    write_sync_histogram,
)
from egologqa.config import apply_topic_overrides, config_to_dict
from egologqa.decoders.depth import decode_depth_message
from egologqa.decoders.rgb import decode_rgb_message
from egologqa.drop_regions import DropRegions
from egologqa.frame_flags import build_frame_flags
from egologqa.gate import evaluate_gate
from egologqa.io.reader import MCapMessageSource, MessageSource
from egologqa.metrics.pixel_metrics import compute_depth_pixel_metrics, compute_rgb_pixel_metrics
from egologqa.metrics.time_metrics import (
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
from egologqa.time import extract_stamp_ns


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
    _emit(progress_cb, "scan", 0.05, "Scanning topics")

    try:
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

        # Pass 1: timestamps only.
        _emit(progress_cb, "pass1", 0.12, "Pass 1: collecting timestamps")
        for rec in source_obj.iter_messages(topics=selected_topics):
            collectors = stream_map.get(rec.topic, [])
            if not collectors:
                continue
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
        _emit(progress_cb, "pass2", 0.55, "Pass 2: decoding sampled frames")
        rgb_frames_by_pos: dict[int, Any] = {}
        depth_frames_by_pos: dict[int, Any] = {}
        rgb_decode_errors: dict[str, int] = {}
        depth_decode_errors: dict[str, int] = {}
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
                            rgb_decode_errors[err or "RGB_DECODE_FAIL"] = (
                                rgb_decode_errors.get(err or "RGB_DECODE_FAIL", 0) + 1
                            )
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
                            depth_decode_errors[err or "DEPTH_PNG_IMDECODE_FAIL"] = (
                                depth_decode_errors.get(err or "DEPTH_PNG_IMDECODE_FAIL", 0) + 1
                            )

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
        for compute_error in exposure_compute_errors:
            errors.append(
                _error(
                    "ERROR",
                    "EXPOSURE_COMPUTE_FAILED",
                    "Exposure feature computation failed for a decoded RGB frame.",
                    compute_error,
                )
            )

        if cfg.debug.export_exposure_csv and rgb_decode_supported:
            if exposure_rows:
                exposure_csv_path = write_exposure_debug_csv(exposure_rows, output_dir)
                report["metrics"]["exposure_debug_csv_path"] = _relative_path(exposure_csv_path, output_dir)
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

        sampled_imu_acc_cov = (
            [imu_acc_cov[idx] for idx in sampled_rgb_indices] if imu_acc_cov is not None else None
        )
        sampled_imu_gyro_cov = (
            [imu_gyro_cov[idx] for idx in sampled_rgb_indices] if imu_gyro_cov is not None else None
        )
        frame_flags = build_frame_flags(
            sampled_rgb_times_ms=sampled_rgb_times_ms,
            sampled_rgb_indices=sampled_rgb_indices,
            sync_deltas_ms=sampled_sync_deltas,
            sync_warn_ms=cfg.thresholds.sync_warn_ms,
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

        segments = extract_segments(
            sampled_times_ns=sampled_rgb_times_ns,
            frame_ok=frame_flags.frame_ok_integrity,
            max_gap_fill_ms=cfg.segments.max_gap_fill_ms,
            min_segment_seconds=cfg.segments.min_segment_seconds,
            forced_break_positions=force_bad_sample_positions,
        )
        report["segments"] = segments
        report["errors"] = errors
        if sampled_sync_deltas is not None:
            report["metrics"]["sync_sample_count"] = len(sampled_sync_deltas)

        gate = evaluate_gate(
            config=cfg,
            metrics=report["metrics"],
            streams=report["streams"],
            duration_s=analyzed_duration_s,
            segments=segments,
            errors=errors,
        )
        report["gate"] = gate

        preview_paths = write_rgb_previews(rgb_frames_by_pos, output_dir)
        plot_path = write_sync_histogram(sampled_sync_deltas, output_dir)
        drop_timeline_path = write_drop_timeline(rgb_col.times_ms, rgb_gap["gap_intervals_ms"], output_dir)
        if preview_paths:
            report["metrics"]["preview_count"] = len(preview_paths)
        if plot_path:
            report["metrics"]["sync_histogram_path"] = plot_path
        if drop_timeline_path:
            report["metrics"]["drop_timeline_path"] = drop_timeline_path
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

    try:
        write_report_markdown(report, output_dir)
    except Exception:
        pass
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
        return str(Path(path).resolve().relative_to(Path(output_dir).resolve()))
    except Exception:
        return str(path)


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
