from __future__ import annotations

import copy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from egologqa.models import QAConfig, TopicOverrides


DEFAULT_CONFIG_PATH = Path("configs/microagi00_ros2.yaml")


def load_config(config_path: str | Path | None) -> QAConfig:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load config files.") from exc
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = QAConfig()
    cfg = _merge_dataclass(cfg, raw)
    _validate_config(cfg)
    return cfg


def apply_topic_overrides(config: QAConfig, overrides: TopicOverrides | None) -> QAConfig:
    if overrides is None:
        return config
    merged = copy.deepcopy(config)
    if overrides.rgb_topic:
        merged.topics.rgb_topic = overrides.rgb_topic
    if overrides.depth_topic:
        merged.topics.depth_topic = overrides.depth_topic
    if overrides.imu_accel_topic:
        merged.topics.imu_accel_topic = overrides.imu_accel_topic
    if overrides.imu_gyro_topic:
        merged.topics.imu_gyro_topic = overrides.imu_gyro_topic
    if any(
        [
            overrides.rgb_topic,
            overrides.depth_topic,
            overrides.imu_accel_topic,
            overrides.imu_gyro_topic,
        ]
    ):
        merged.topics.mode = "explicit"
    _validate_config(merged)
    return merged


def config_to_dict(config: QAConfig) -> dict[str, Any]:
    return asdict(config)


def _merge_dataclass(obj: Any, patch: dict[str, Any]) -> Any:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass object")
    for field_info in fields(obj):
        key = field_info.name
        if key not in patch:
            continue
        current = getattr(obj, key)
        incoming = patch[key]
        if is_dataclass(current) and isinstance(incoming, dict):
            _merge_dataclass(current, incoming)
        else:
            setattr(obj, key, incoming)
    return obj


def _validate_config(config: QAConfig) -> None:
    if config.topics.mode not in {"explicit", "auto"}:
        raise ValueError("topics.mode must be 'explicit' or 'auto'")
    if config.sampling.rgb_stride <= 0:
        raise ValueError("sampling.rgb_stride must be > 0")
    if config.sampling.max_rgb_frames <= 0:
        raise ValueError("sampling.max_rgb_frames must be > 0")
    if config.integrity.out_of_order_warn_ratio < 0:
        raise ValueError("integrity.out_of_order_warn_ratio must be >= 0")
    if config.segments.min_segment_seconds <= 0:
        raise ValueError("segments.min_segment_seconds must be > 0")
    if config.thresholds.exposure_roi_margin_ratio < 0:
        raise ValueError("thresholds.exposure_roi_margin_ratio must be >= 0")
    if config.thresholds.exposure_roi_margin_ratio >= 0.5:
        raise ValueError("thresholds.exposure_roi_margin_ratio must be < 0.5")
    if not (0 <= config.thresholds.low_clip_pixel_value <= 255):
        raise ValueError("thresholds.low_clip_pixel_value must be in [0, 255]")
    if not (0 <= config.thresholds.high_clip_pixel_value <= 255):
        raise ValueError("thresholds.high_clip_pixel_value must be in [0, 255]")
    if not (0.0 <= config.thresholds.low_clip_warn <= 1.0):
        raise ValueError("thresholds.low_clip_warn must be in [0, 1]")
    if not (0.0 <= config.thresholds.high_clip_warn <= 1.0):
        raise ValueError("thresholds.high_clip_warn must be in [0, 1]")
    if not (0.0 <= config.thresholds.median_dark <= 255.0):
        raise ValueError("thresholds.median_dark must be in [0, 255]")
    if not (0.0 <= config.thresholds.median_bright <= 255.0):
        raise ValueError("thresholds.median_bright must be in [0, 255]")
    if not (0.0 <= config.thresholds.dynamic_range_min <= 255.0):
        raise ValueError("thresholds.dynamic_range_min must be in [0, 255]")
