from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egologqa.artifacts import write_drop_timeline, write_sync_histogram


def _dark_pixel_count(img: np.ndarray) -> int:
    return int(np.sum(np.all(img < 90, axis=2)))


def test_sync_histogram_contains_readable_axis_regions(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    out = write_sync_histogram(
        sync_deltas_ms=[0.2, 0.5, 0.7, 1.2, 1.5, 1.8, 2.0, 2.4, 2.8, 3.1],
        output_dir=tmp_path,
    )
    assert out is not None
    path = Path(out)
    assert path.exists()
    img = cv2.imread(str(path))
    assert img is not None
    assert img.shape[0] >= 400
    assert _dark_pixel_count(img[-120:, :, :]) > 400
    assert _dark_pixel_count(img[:, :72, :]) > 100


def test_drop_timeline_contains_time_axis_labels(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    out = write_drop_timeline(
        rgb_times_ms=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        gap_intervals_ms=[(80.0, 120.0), (340.0, 380.0)],
        output_dir=tmp_path,
    )
    assert out is not None
    path = Path(out)
    assert path.exists()
    img = cv2.imread(str(path))
    assert img is not None
    assert img.shape[0] >= 280
    assert _dark_pixel_count(img[-120:, :, :]) > 350
