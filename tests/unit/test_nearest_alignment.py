from __future__ import annotations

import numpy as np

from egologqa.metrics.time_metrics import nearest_abs_delta, nearest_indices


def test_nearest_abs_delta_basic() -> None:
    q = np.array([10.0, 20.0, 40.0], dtype=np.float64)
    ref = np.array([0.0, 18.0, 44.0], dtype=np.float64)
    out = nearest_abs_delta(q, ref)
    assert out.tolist() == [8.0, 2.0, 4.0]


def test_nearest_indices_basic() -> None:
    q = np.array([1.0, 9.0, 31.0], dtype=np.float64)
    ref = np.array([0.0, 10.0, 30.0], dtype=np.float64)
    out = nearest_indices(q, ref)
    assert out.tolist() == [0, 1, 2]


def test_nearest_indices_tie_prefers_left() -> None:
    q = np.array([5.0], dtype=np.float64)
    ref = np.array([0.0, 10.0], dtype=np.float64)
    out = nearest_indices(q, ref)
    assert out.tolist() == [0]
