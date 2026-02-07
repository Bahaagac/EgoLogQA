from __future__ import annotations


def extract_segments(
    sampled_times_ns: list[int],
    frame_ok: list[bool],
    max_gap_fill_ms: float,
    min_segment_seconds: float,
    forced_break_positions: set[int] | None = None,
) -> list[dict[str, float | int]]:
    forced = forced_break_positions or set()
    segments: list[dict[str, float | int]] = []
    current_start_ns: int | None = None
    last_ok_ns: int | None = None
    last_ok_pos: int | None = None

    for pos, ok in enumerate(frame_ok):
        if not ok:
            continue
        t_ns = sampled_times_ns[pos]
        if current_start_ns is None:
            current_start_ns = t_ns
            last_ok_ns = t_ns
            last_ok_pos = pos
            continue

        dt_ms = (t_ns - (last_ok_ns or t_ns)) / 1_000_000.0
        forced_break = pos in forced or (last_ok_pos is not None and last_ok_pos in forced)
        monotonic_break = last_ok_ns is not None and t_ns < last_ok_ns
        if forced_break or monotonic_break or dt_ms > max_gap_fill_ms:
            _append_segment(segments, current_start_ns, last_ok_ns or current_start_ns, min_segment_seconds)
            current_start_ns = t_ns
        last_ok_ns = t_ns
        last_ok_pos = pos

    if current_start_ns is not None and last_ok_ns is not None:
        _append_segment(segments, current_start_ns, last_ok_ns, min_segment_seconds)

    return segments


def _append_segment(
    segments: list[dict[str, float | int]],
    start_ns: int,
    end_ns: int,
    min_segment_seconds: float,
) -> None:
    duration_s = (end_ns - start_ns) / 1_000_000_000.0
    if duration_s < min_segment_seconds:
        return
    segments.append(
        {
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
            "duration_s": float(duration_s),
        }
    )
