from __future__ import annotations

import numpy as np
import pytest

from egologqa.metrics.pixel_metrics import compute_rgb_pixel_metrics
from egologqa.models import ThresholdsConfig


def _as_bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def test_blur_uses_fixed_threshold_and_defined_formula() -> None:
    cv2 = pytest.importorskip("cv2")
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255
    frame = _as_bgr(gray)

    thresholds = ThresholdsConfig()
    thresholds.blur_roi_margin_ratio = 0.10
    thresholds.blur_threshold_min = 80.0

    metrics, blur_ok, _exposure_ok, _rows, _errors = compute_rgb_pixel_metrics(
        [frame], thresholds, [0], [0.0]
    )

    margin = int(min(gray.shape[:2]) * thresholds.blur_roi_margin_ratio)
    roi = gray[margin : gray.shape[0] - margin, margin : gray.shape[1] - margin]
    expected_blur = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    assert metrics["blur_threshold"] == thresholds.blur_threshold_min
    assert metrics["blur_median"] == pytest.approx(expected_blur)
    assert blur_ok == [expected_blur >= thresholds.blur_threshold_min]
