from __future__ import annotations

from pathlib import Path

import numpy as np

from egologqa.artifacts import write_blur_evidence_frames


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_blur_evidence_selection_is_deterministic_with_ties(tmp_path: Path) -> None:
    fail_rows = [
        {"sample_i": 3, "t_ms": 3000.0, "blur_value": 1.0, "frame": _frame(3)},
        {"sample_i": 1, "t_ms": 1000.0, "blur_value": 1.0, "frame": _frame(1)},
        {"sample_i": 2, "t_ms": 2000.0, "blur_value": 0.5, "frame": _frame(2)},
    ]
    pass_rows = [
        {"sample_i": 9, "t_ms": 9000.0, "blur_value": 200.0, "frame": _frame(9)},
        {"sample_i": 7, "t_ms": 7000.0, "blur_value": 200.0, "frame": _frame(7)},
        {"sample_i": 8, "t_ms": 8000.0, "blur_value": 150.0, "frame": _frame(8)},
    ]
    fail_dir, pass_dir = write_blur_evidence_frames(fail_rows, pass_rows, tmp_path, k=2)
    assert fail_dir is not None
    assert pass_dir is not None

    fail_names = sorted(p.name for p in Path(fail_dir).glob("*.jpg"))
    pass_names = sorted(p.name for p in Path(pass_dir).glob("*.jpg"))

    assert fail_names == [
        "fail_rank01_i0002_t2000_blur0.50.jpg",
        "fail_rank02_i0001_t1000_blur1.00.jpg",
    ]
    assert pass_names == [
        "pass_rank01_i0007_t7000_blur200.00.jpg",
        "pass_rank02_i0009_t9000_blur200.00.jpg",
    ]
