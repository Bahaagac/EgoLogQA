from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from egologqa.report import sanitize_json_value


def write_report_markdown(report: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.md"
    gate = report.get("gate", {}).get("gate", "UNKNOWN")
    lines = [
        "# EgoLogQA Report",
        "",
        f"- Gate: **{gate}**",
        f"- Recommended action: `{report.get('gate', {}).get('recommended_action')}`",
        f"- Input: `{report.get('input', {}).get('file_path')}`",
        f"- Duration (s): `{report.get('time', {}).get('duration_s')}`",
        "",
        "## Reasons",
        "",
        f"- FAIL: {', '.join(report.get('gate', {}).get('fail_reasons', [])) or '(none)'}",
        f"- WARN: {', '.join(report.get('gate', {}).get('warn_reasons', [])) or '(none)'}",
        "",
        "## Key Metrics",
        "",
        f"- sync_p95_ms: `{report.get('metrics', {}).get('sync_p95_ms')}`",
        f"- drop_ratio: `{report.get('metrics', {}).get('drop_ratio')}`",
        f"- imu_combined_missing_ratio: `{report.get('metrics', {}).get('imu_combined_missing_ratio')}`",
        f"- integrity_ok_ratio: `{report.get('metrics', {}).get('integrity_ok_ratio')}`",
        f"- vision_ok_ratio: `{report.get('metrics', {}).get('vision_ok_ratio')}`",
        "",
        f"- integrity_segments: `{len(report.get('segments', []))}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_rgb_previews(
    frames_by_pos: dict[int, np.ndarray],
    output_dir: str | Path,
    max_previews: int = 12,
    exclude_positions: set[int] | None = None,
) -> list[str]:
    if not frames_by_pos:
        return []
    try:
        import cv2
    except Exception:
        return []
    preview_dir = Path(output_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    excluded = exclude_positions or set()
    ordered_positions = sorted(frames_by_pos.keys())
    preferred_positions = [pos for pos in ordered_positions if pos not in excluded]
    fallback_positions = [pos for pos in ordered_positions if pos in excluded]
    selected_positions = (preferred_positions + fallback_positions)[:max_previews]
    for idx, pos in enumerate(selected_positions):
        frame = frames_by_pos[pos]
        file_name = f"rgb_{idx:04d}_sample_{pos:06d}.png"
        path = preview_dir / file_name
        ok = cv2.imwrite(str(path), frame)
        if ok:
            saved.append(str(path))
    return saved


def write_sync_histogram(sync_deltas_ms: list[float] | None, output_dir: str | Path) -> str | None:
    if not sync_deltas_ms:
        return None
    try:
        import cv2
    except Exception:
        return None

    deltas = np.asarray(sync_deltas_ms, dtype=np.float64)
    if deltas.size == 0:
        return None

    bins = 20
    hist, bin_edges = np.histogram(deltas, bins=bins)
    width, height = 920, 420
    left_margin = 72
    right_margin = 36
    top_margin = 44
    bottom_margin = 86
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    max_count = max(1, int(hist.max()))
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    for i, count in enumerate(hist):
        x0 = plot_left + int(i * plot_w / bins)
        x1 = plot_left + int((i + 1) * plot_w / bins) - 2
        h = int((count / max_count) * plot_h)
        y1 = plot_bottom - h
        cv2.rectangle(canvas, (x0, y1), (max(x0 + 1, x1), plot_bottom), (70, 120, 220), thickness=-1)

    for tick_i in range(5):
        y_val = int(round((max_count * tick_i) / 4))
        y = plot_bottom - int((y_val / max_count) * plot_h)
        cv2.line(canvas, (plot_left, y), (plot_right, y), (225, 225, 225), 1)
        cv2.line(canvas, (plot_left - 4, y), (plot_left, y), (20, 20, 20), 1)
        cv2.putText(
            canvas,
            str(y_val),
            (12, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    for tick_i in range(5):
        frac = tick_i / 4.0
        x = plot_left + int(frac * plot_w)
        x_val = x_min + frac * (x_max - x_min)
        cv2.line(canvas, (x, plot_bottom), (x, plot_bottom + 4), (20, 20, 20), 1)
        label = f"{x_val:.1f}"
        if tick_i == 0:
            lx = x
        elif tick_i == 4:
            lx = x - 40
        else:
            lx = x - 18
        cv2.putText(
            canvas,
            label,
            (lx, plot_bottom + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), (20, 20, 20), 1)
    cv2.line(canvas, (plot_left, plot_bottom), (plot_left, plot_top), (20, 20, 20), 1)
    cv2.putText(
        canvas,
        "Sync delta histogram (ms)",
        (plot_left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "count",
        (10, plot_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "delta (ms)",
        (plot_left + (plot_w // 2) - 35, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"max={float(deltas.max()):.2f}  p95={float(np.percentile(deltas, 95)):.2f}",
        (plot_left, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "sync_histogram.png"
    if cv2.imwrite(str(out_path), canvas):
        return str(out_path)
    return None


def write_drop_timeline(
    rgb_times_ms: list[float],
    gap_intervals_ms: list[tuple[float, float]],
    output_dir: str | Path,
) -> str | None:
    if len(rgb_times_ms) < 2:
        return None
    try:
        import cv2
    except Exception:
        return None

    width, height = 920, 300
    left_margin = 72
    right_margin = 36
    top_margin = 46
    bottom_margin = 92
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    t0 = float(min(rgb_times_ms))
    t1 = float(max(rgb_times_ms))
    if t1 <= t0:
        return None
    duration_s = (t1 - t0) / 1000.0
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin

    def tx(t: float) -> int:
        return int(plot_left + (t - t0) * (plot_right - plot_left) / (t1 - t0))

    y = (plot_top + plot_bottom) // 2
    cv2.line(canvas, (plot_left, y), (plot_right, y), (20, 20, 20), 1)
    for left, right in gap_intervals_ms:
        x0 = tx(left)
        x1 = tx(right)
        cv2.rectangle(canvas, (x0, y - 30), (max(x0 + 1, x1), y + 30), (80, 80, 220), -1)

    for tick_i in range(5):
        frac = tick_i / 4.0
        x = plot_left + int(frac * (plot_right - plot_left))
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), (225, 225, 225), 1)
        cv2.line(canvas, (x, plot_bottom), (x, plot_bottom + 4), (20, 20, 20), 1)
        tick_s = duration_s * frac
        label = f"{tick_s:.1f}s"
        if tick_i == 0:
            lx = x
        elif tick_i == 4:
            lx = x - 44
        else:
            lx = x - 18
        cv2.putText(
            canvas,
            label,
            (lx, plot_bottom + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), (20, 20, 20), 1)

    cv2.putText(
        canvas,
        "Drop/Gap timeline",
        (plot_left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "time (s)",
        (plot_left + ((plot_right - plot_left) // 2) - 24, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "gap blocks",
        (10, plot_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"start_s=0.00 end_s={duration_s:.2f} duration_s={duration_s:.2f}",
        (plot_left, height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"gaps={len(gap_intervals_ms)}",
        (plot_left, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "drop_timeline.png"
    if cv2.imwrite(str(out_path), canvas):
        return str(out_path)
    return None


def write_exposure_debug_csv(
    exposure_rows: list[dict[str, float | int | str]],
    output_dir: str | Path,
) -> str | None:
    if not exposure_rows:
        return None
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "exposure_samples.csv"
    rows = sorted(
        exposure_rows,
        key=lambda r: (int(r.get("sample_i", 0)), float(r.get("t_ms", 0.0))),
    )
    fields = [
        "sample_i",
        "t_ms",
        "roi_margin_ratio",
        "low_clip",
        "high_clip",
        "high_clip_luma",
        "high_clip_any_channel",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "contrast",
        "dynamic_range",
        "exposure_bad",
        "reasons",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: _format_csv_value(row.get(key))
                for key in fields
            }
            writer.writerow(formatted)
    return str(out_path)


def write_blur_debug_csv(
    blur_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> str | None:
    if not blur_rows:
        return None
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "blur_samples.csv"
    rows = sorted(
        blur_rows,
        key=lambda r: int(r.get("sample_i", 0)),
    )
    fields = [
        "sample_i",
        "t_ms",
        "roi_margin_ratio",
        "decode_ok",
        "blur_value",
        "blur_threshold",
        "blur_ok",
        "decode_error_code",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {key: _format_csv_value(row.get(key)) for key in fields}
            writer.writerow(formatted)
    return str(out_path)


def write_depth_debug_csv(
    depth_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> str | None:
    if not depth_rows:
        return None
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "depth_samples.csv"
    rows = sorted(
        depth_rows,
        key=lambda r: int(r.get("sample_i", 0)),
    )
    fields = [
        "sample_i",
        "t_ms",
        "decode_ok",
        "invalid_ratio",
        "min_depth",
        "max_depth",
        "dtype",
        "error_code",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {key: _format_csv_value(row.get(key)) for key in fields}
            writer.writerow(formatted)
    return str(out_path)


def write_clean_segments_json(
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    file_name: str,
) -> str | None:
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / file_name
    payload = sanitize_json_value(list(segments))
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    return str(out_path)


def select_exposure_evidence_rows(
    exposure_rows: list[dict[str, Any]],
    k: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    reason_order = ["low_clip", "high_clip", "flat_and_dark", "flat_and_bright"]
    selected: dict[str, list[dict[str, Any]]] = {reason: [] for reason in reason_order}
    warnings: list[str] = []
    if k <= 0:
        return selected, warnings

    for reason in reason_order:
        candidates = [row for row in exposure_rows if reason in _parse_reasons(row.get("reasons"))]
        if not candidates:
            continue
        if reason == "low_clip":
            ordered, used_fallback = _sort_or_fallback(
                candidates,
                value_keys=("low_clip",),
                key_fn=lambda row: (-float(row["low_clip"]), int(row.get("sample_i", 0))),
            )
        elif reason == "high_clip":
            ordered, used_fallback = _sort_or_fallback(
                candidates,
                value_keys=("high_clip",),
                key_fn=lambda row: (-float(row["high_clip"]), int(row.get("sample_i", 0))),
            )
        elif reason == "flat_and_dark":
            ordered, used_fallback = _sort_or_fallback(
                candidates,
                value_keys=("dynamic_range", "p50"),
                key_fn=lambda row: (
                    float(row["dynamic_range"]),
                    float(row["p50"]),
                    int(row.get("sample_i", 0)),
                ),
            )
        else:
            ordered, used_fallback = _sort_or_fallback(
                candidates,
                value_keys=("dynamic_range", "p50"),
                key_fn=lambda row: (
                    float(row["dynamic_range"]),
                    -float(row["p50"]),
                    int(row.get("sample_i", 0)),
                ),
            )
        if used_fallback:
            warnings.append(
                f"{reason}: missing one or more scoring columns; used sample_i fallback ordering"
            )
        selected[reason] = ordered[:k]
    return selected, warnings


def write_exposure_evidence_frames(
    selected_rows: dict[str, list[dict[str, Any]]],
    rgb_frames_by_pos: dict[int, np.ndarray],
    output_dir: str | Path,
) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "low_clip": None,
        "high_clip": None,
        "flat_and_dark": None,
        "flat_and_bright": None,
    }
    try:
        import cv2
    except Exception:
        return out
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    dir_by_reason = {
        "low_clip": debug_dir / "exposure_low_clip_frames",
        "high_clip": debug_dir / "exposure_high_clip_frames",
        "flat_and_dark": debug_dir / "exposure_flat_and_dark_frames",
        "flat_and_bright": debug_dir / "exposure_flat_and_bright_frames",
    }
    for reason, rows in selected_rows.items():
        if reason not in dir_by_reason:
            continue
        target_dir = dir_by_reason[reason]
        target_dir.mkdir(parents=True, exist_ok=True)
        wrote = 0
        for rank, row in enumerate(rows, start=1):
            sample_i = int(row.get("sample_i", 0))
            frame = rgb_frames_by_pos.get(sample_i)
            if frame is None:
                continue
            t_ms = float(row.get("t_ms", 0.0))
            score_suffix = _exposure_score_suffix(reason, row)
            file_name = (
                f"{reason}_rank{rank:02d}_i{sample_i:04d}_t{t_ms:.0f}_{score_suffix}.jpg"
            )
            out_path = target_dir / file_name
            ok = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            if ok:
                wrote += 1
        out[reason] = str(target_dir) if wrote > 0 else None
    return out


def write_exposure_evidence_error(
    warnings: list[str],
    output_dir: str | Path,
) -> str | None:
    if not warnings:
        return None
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "exposure_evidence_error.txt"
    out_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    return str(out_path)


def write_blur_evidence_frames(
    blur_fail_rows: list[dict[str, Any]],
    blur_pass_rows: list[dict[str, Any]],
    output_dir: str | Path,
    k: int,
) -> tuple[str | None, str | None]:
    if k <= 0:
        return None, None
    try:
        import cv2
    except Exception:
        return None, None
    debug_dir = Path(output_dir) / "debug"
    fail_dir = debug_dir / "blur_fail_frames"
    pass_dir = debug_dir / "blur_pass_frames"
    fail_dir.mkdir(parents=True, exist_ok=True)
    pass_dir.mkdir(parents=True, exist_ok=True)

    fail_sel, pass_sel = select_blur_evidence_rows(blur_fail_rows, blur_pass_rows, k)

    wrote_fail = _write_evidence_set(fail_sel, fail_dir, "fail", cv2)
    wrote_pass = _write_evidence_set(pass_sel, pass_dir, "pass", cv2)

    fail_path = str(fail_dir) if wrote_fail > 0 else None
    pass_path = str(pass_dir) if wrote_pass > 0 else None
    return fail_path, pass_path


def _write_evidence_set(rows: list[dict[str, Any]], out_dir: Path, prefix: str, cv2_mod: Any) -> int:
    wrote = 0
    for rank, row in enumerate(rows, start=1):
        frame = row.get("frame")
        if frame is None:
            continue
        file_name = _evidence_file_name(prefix, rank, row)
        out_path = out_dir / file_name
        ok = cv2_mod.imwrite(
            str(out_path),
            frame,
            [int(cv2_mod.IMWRITE_JPEG_QUALITY), 90],
        )
        if ok:
            wrote += 1
    return wrote


def select_blur_evidence_rows(
    blur_fail_rows: list[dict[str, Any]],
    blur_pass_rows: list[dict[str, Any]],
    k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if k <= 0:
        return [], []
    fail_sel = sorted(
        blur_fail_rows,
        key=lambda row: (float(row.get("blur_value", 0.0)), int(row.get("sample_i", 0))),
    )[:k]
    pass_sel = sorted(
        blur_pass_rows,
        key=lambda row: (-float(row.get("blur_value", 0.0)), int(row.get("sample_i", 0))),
    )[:k]
    return fail_sel, pass_sel


def write_blur_annotated_evidence_frames(
    blur_fail_rows: list[dict[str, Any]],
    blur_pass_rows: list[dict[str, Any]],
    output_dir: str | Path,
    k: int,
    blur_threshold: float | None,
) -> tuple[str | None, str | None]:
    if k <= 0:
        return None, None
    try:
        import cv2
    except Exception:
        return None, None

    debug_dir = Path(output_dir) / "debug"
    src_fail_dir = debug_dir / "blur_fail_frames"
    src_pass_dir = debug_dir / "blur_pass_frames"
    ann_fail_dir = debug_dir / "blur_fail_frames_annotated"
    ann_pass_dir = debug_dir / "blur_pass_frames_annotated"
    ann_fail_dir.mkdir(parents=True, exist_ok=True)
    ann_pass_dir.mkdir(parents=True, exist_ok=True)

    fail_sel, pass_sel = select_blur_evidence_rows(blur_fail_rows, blur_pass_rows, k)
    fail_count = _write_annotated_set(
        rows=fail_sel,
        src_dir=src_fail_dir,
        out_dir=ann_fail_dir,
        prefix="fail",
        threshold=blur_threshold,
        cv2_mod=cv2,
    )
    pass_count = _write_annotated_set(
        rows=pass_sel,
        src_dir=src_pass_dir,
        out_dir=ann_pass_dir,
        prefix="pass",
        threshold=blur_threshold,
        cv2_mod=cv2,
    )
    fail_path = str(ann_fail_dir) if fail_count > 0 else None
    pass_path = str(ann_pass_dir) if pass_count > 0 else None
    return fail_path, pass_path


def write_evidence_manifest_json(
    output_dir: str | Path,
    blur_fail_rows: list[dict[str, Any]],
    blur_pass_rows: list[dict[str, Any]],
    k: int,
    blur_threshold: float | None,
    rgb_stride: int,
    max_rgb_frames: int,
    cv2_available: bool,
    annotated_available: bool,
) -> str | None:
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = debug_dir / "evidence_manifest.json"

    fail_sel, pass_sel = select_blur_evidence_rows(blur_fail_rows, blur_pass_rows, k)
    manifest = {
        "schema_version": 1,
        "selection_context": {
            "rgb_stride": int(rgb_stride),
            "max_rgb_frames": int(max_rgb_frames),
            "evidence_frames_k": int(k),
            "cv2_available": bool(cv2_available),
        },
        "disabled_reason": None if cv2_available else "cv2_unavailable",
        "evidence_sets": {
            "blur_fail": _manifest_entries(
                rows=fail_sel,
                prefix="fail",
                decision="fail",
                blur_threshold=blur_threshold,
                cv2_available=cv2_available,
                annotated_available=annotated_available,
            ),
            "blur_pass": _manifest_entries(
                rows=pass_sel,
                prefix="pass",
                decision="pass",
                blur_threshold=blur_threshold,
                cv2_available=cv2_available,
                annotated_available=annotated_available,
            ),
        },
    }
    sanitized = sanitize_json_value(manifest)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    return str(manifest_path)


def write_benchmarks_json(
    benchmarks: dict[str, Any],
    output_dir: str | Path,
) -> str | None:
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "benchmarks.json"
    sanitized = sanitize_json_value(benchmarks)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
    return str(path)


def _manifest_entries(
    rows: list[dict[str, Any]],
    prefix: str,
    decision: str,
    blur_threshold: float | None,
    cv2_available: bool,
    annotated_available: bool,
) -> list[dict[str, Any]]:
    if not cv2_available:
        return []
    entries: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        file_name = _evidence_file_name(prefix, rank, row)
        source_rel = f"debug/blur_{decision}_frames/{file_name}"
        annotated_rel = (
            f"debug/blur_{decision}_frames_annotated/{file_name}"
            if annotated_available
            else None
        )
        entries.append(
            {
                "rank": int(rank),
                "sample_i": int(row.get("sample_i", 0)),
                "timestamp_ms": float(row.get("t_ms", 0.0)),
                "blur_value": float(row.get("blur_value", 0.0)),
                "blur_threshold": float(blur_threshold) if blur_threshold is not None else None,
                "source_image_relpath": source_rel,
                "annotated_image_relpath": annotated_rel,
                "decision": decision,
            }
        )
    return entries


def _write_annotated_set(
    rows: list[dict[str, Any]],
    src_dir: Path,
    out_dir: Path,
    prefix: str,
    threshold: float | None,
    cv2_mod: Any,
) -> int:
    wrote = 0
    for rank, row in enumerate(rows, start=1):
        file_name = _evidence_file_name(prefix, rank, row)
        src_path = src_dir / file_name
        if not src_path.exists():
            continue
        img = cv2_mod.imread(str(src_path), cv2_mod.IMREAD_COLOR)
        if img is None:
            continue
        sample_i = int(row.get("sample_i", 0))
        t_ms = float(row.get("t_ms", 0.0))
        blur_value = float(row.get("blur_value", 0.0))
        _draw_annotation(
            img=img,
            score=blur_value,
            threshold=threshold,
            sample_i=sample_i,
            t_ms=t_ms,
            cv2_mod=cv2_mod,
        )
        ok = cv2_mod.imwrite(
            str(out_dir / file_name),
            img,
            [int(cv2_mod.IMWRITE_JPEG_QUALITY), 90],
        )
        if ok:
            wrote += 1
    return wrote


def _draw_annotation(
    img: np.ndarray,
    score: float,
    threshold: float | None,
    sample_i: int,
    t_ms: float,
    cv2_mod: Any,
) -> None:
    lines = [
        "BLUR",
        f"score={score:.4f}",
        f"threshold={threshold:.4f}" if threshold is not None else "threshold=none",
        f"sample_i={sample_i}",
        f"t_ms={t_ms:.4f}",
    ]
    x = 12
    y = 24
    cv2_mod.rectangle(img, (8, 6), (420, 120), (0, 0, 0), thickness=-1)
    for line in lines:
        cv2_mod.putText(
            img,
            line,
            (x, y),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2_mod.LINE_AA,
        )
        y += 20


def _evidence_file_name(prefix: str, rank: int, row: dict[str, Any]) -> str:
    sample_i = int(row.get("sample_i", 0))
    t_ms = float(row.get("t_ms", 0.0))
    blur_value = float(row.get("blur_value", 0.0))
    return f"{prefix}_rank{rank:02d}_i{sample_i:04d}_t{t_ms:.0f}_blur{blur_value:.2f}.jpg"


def _parse_reasons(raw: Any) -> set[str]:
    if not isinstance(raw, str):
        return set()
    return {part.strip() for part in raw.split(";") if part.strip()}


def _sort_or_fallback(
    rows: list[dict[str, Any]],
    value_keys: tuple[str, ...],
    key_fn: Any,
) -> tuple[list[dict[str, Any]], bool]:
    complete = []
    for row in rows:
        if all(_to_float(row.get(key)) is not None for key in value_keys):
            complete.append(row)
    if len(complete) == len(rows):
        return sorted(complete, key=key_fn), False
    # Fallback remains scoped to this reason's own candidate rows.
    return sorted(rows, key=lambda row: int(row.get("sample_i", 0))), True


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _exposure_score_suffix(reason: str, row: dict[str, Any]) -> str:
    if reason == "low_clip":
        low_clip = _to_float(row.get("low_clip")) or 0.0
        p50 = _to_float(row.get("p50")) or 0.0
        return f"lc{low_clip:.2f}_p50{p50:.0f}"
    if reason == "high_clip":
        high_clip = _to_float(row.get("high_clip")) or 0.0
        p50 = _to_float(row.get("p50")) or 0.0
        return f"hc{high_clip:.2f}_p50{p50:.0f}"
    dynamic_range = _to_float(row.get("dynamic_range")) or 0.0
    p50 = _to_float(row.get("p50")) or 0.0
    return f"dr{dynamic_range:.2f}_p50{p50:.0f}"


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
