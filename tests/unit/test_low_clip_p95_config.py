from __future__ import annotations

from pathlib import Path

import pytest

from egologqa.config import load_config


def _write_config(tmp_path: Path, low_clip_p95_max_literal: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"thresholds:\n  low_clip_p95_max: {low_clip_p95_max_literal}\n",
        encoding="utf-8",
    )
    return path


def test_default_yaml_includes_low_clip_p95_max() -> None:
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "configs" / "microagi00_ros2.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    assert raw["thresholds"]["low_clip_p95_max"] == 180.0


@pytest.mark.parametrize(
    "literal",
    [
        ".nan",
        ".inf",
        "-.inf",
        "!!float .nan",
        "!!float .inf",
        "!!float -.inf",
    ],
)
def test_low_clip_p95_max_rejects_non_finite_values(
    tmp_path: Path, literal: str
) -> None:
    path = _write_config(tmp_path, literal)
    with pytest.raises(ValueError, match="thresholds.low_clip_p95_max"):
        load_config(path)


@pytest.mark.parametrize("value", ["-1", "256"])
def test_low_clip_p95_max_rejects_out_of_range_values(
    tmp_path: Path, value: str
) -> None:
    path = _write_config(tmp_path, value)
    with pytest.raises(ValueError, match="thresholds.low_clip_p95_max"):
        load_config(path)


def test_low_clip_p95_max_accepts_valid_value(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "180.0")
    cfg = load_config(path)
    assert cfg.thresholds.low_clip_p95_max == pytest.approx(180.0)
