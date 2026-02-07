from __future__ import annotations

from egologqa.segments import extract_segments


def test_segment_extraction_respects_forced_break() -> None:
    times = [
        0,
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
        4_000_000_000,
        5_000_000_000,
    ]
    frame_ok = [True, True, True, True, True, True]
    out = extract_segments(
        sampled_times_ns=times,
        frame_ok=frame_ok,
        max_gap_fill_ms=1500.0,
        min_segment_seconds=1.0,
        forced_break_positions={3},
    )
    assert len(out) == 2
    assert out[0]["start_ns"] == 0
    assert out[0]["end_ns"] == 2_000_000_000
    assert out[1]["start_ns"] == 4_000_000_000
