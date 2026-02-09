from __future__ import annotations

from egologqa.artifacts import select_exposure_evidence_rows


def test_exposure_evidence_selection_is_deterministic() -> None:
    rows = [
        {"sample_i": 5, "t_ms": 5000.0, "reasons": "low_clip", "low_clip": 0.2, "p50": 10.0},
        {"sample_i": 3, "t_ms": 3000.0, "reasons": "low_clip", "low_clip": 0.2, "p50": 9.0},
        {"sample_i": 1, "t_ms": 1000.0, "reasons": "low_clip", "low_clip": 0.4, "p50": 12.0},
    ]
    selected, warnings = select_exposure_evidence_rows(rows, k=2)
    assert warnings == []
    assert [row["sample_i"] for row in selected["low_clip"]] == [1, 3]


def test_exposure_evidence_fallback_stays_within_reason_group() -> None:
    rows = [
        {"sample_i": 2, "t_ms": 2000.0, "reasons": "flat_and_dark", "p50": 10.0},
        {"sample_i": 4, "t_ms": 4000.0, "reasons": "flat_and_dark", "p50": 12.0},
        {"sample_i": 1, "t_ms": 1000.0, "reasons": "low_clip", "low_clip": 0.8, "p50": 5.0},
    ]
    selected, warnings = select_exposure_evidence_rows(rows, k=4)
    assert any("flat_and_dark" in msg for msg in warnings)
    assert [row["sample_i"] for row in selected["flat_and_dark"]] == [2, 4]
