from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


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
    for idx, pos in enumerate(sorted(frames_by_pos.keys())[:max_previews]):
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
    width, height = 900, 360
    margin = 40
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    max_count = max(1, int(hist.max()))
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bin_w = max(1, plot_w // bins)

    for i, count in enumerate(hist):
        x0 = margin + i * bin_w
        x1 = x0 + bin_w - 2
        h = int((count / max_count) * plot_h)
        y0 = height - margin
        y1 = y0 - h
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 120, 220), thickness=-1)

    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (20, 20, 20), 1)
    cv2.line(canvas, (margin, height - margin), (margin, margin), (20, 20, 20), 1)
    cv2.putText(
        canvas,
        "Sync delta histogram (ms)",
        (margin, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"max={float(deltas.max()):.2f}  p95={float(np.percentile(deltas, 95)):.2f}",
        (margin, height - 10),
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

    width, height = 900, 220
    margin = 40
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    t0 = float(min(rgb_times_ms))
    t1 = float(max(rgb_times_ms))
    if t1 <= t0:
        return None

    def tx(t: float) -> int:
        return int(margin + (t - t0) * (width - 2 * margin) / (t1 - t0))

    y = height // 2
    cv2.line(canvas, (margin, y), (width - margin, y), (20, 20, 20), 1)
    for left, right in gap_intervals_ms:
        x0 = tx(left)
        x1 = tx(right)
        cv2.rectangle(canvas, (x0, y - 20), (max(x0 + 1, x1), y + 20), (80, 80, 220), -1)

    cv2.putText(
        canvas,
        "Drop/Gap timeline",
        (margin, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"gaps={len(gap_intervals_ms)}",
        (margin, height - 10),
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

    fail_sel = sorted(
        blur_fail_rows,
        key=lambda row: (float(row.get("blur_value", 0.0)), int(row.get("sample_i", 0))),
    )[:k]
    pass_sel = sorted(
        blur_pass_rows,
        key=lambda row: (-float(row.get("blur_value", 0.0)), int(row.get("sample_i", 0))),
    )[:k]

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
        sample_i = int(row.get("sample_i", 0))
        t_ms = float(row.get("t_ms", 0.0))
        blur_value = float(row.get("blur_value", 0.0))
        file_name = (
            f"{prefix}_rank{rank:02d}_i{sample_i:04d}_t{t_ms:.0f}_blur{blur_value:.2f}.jpg"
        )
        out_path = out_dir / file_name
        ok = cv2_mod.imwrite(
            str(out_path),
            frame,
            [int(cv2_mod.IMWRITE_JPEG_QUALITY), 90],
        )
        if ok:
            wrote += 1
    return wrote


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
