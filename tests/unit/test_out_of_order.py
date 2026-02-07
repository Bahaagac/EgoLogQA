from __future__ import annotations

from egologqa.metrics.time_metrics import compute_out_of_order_ratio


def test_out_of_order_ratio() -> None:
    count, ratio = compute_out_of_order_ratio([1.0, 2.0, 1.5, 3.0, 2.5])
    assert count == 2
    assert ratio == 2 / 4
