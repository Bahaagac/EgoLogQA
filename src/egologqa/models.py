from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class HeaderStamp:
    sec: int
    nanosec: int


@dataclass
class TopicAutoConfig:
    rgb_regex: str = r"color.*compressed"
    depth_regex: str = r"depth.*compressedDepth"
    imu_regex: str = r"imu.*sample"


@dataclass
class TopicsConfig:
    mode: str = "explicit"
    rgb_topic: Optional[str] = None
    depth_topic: Optional[str] = None
    imu_accel_topic: Optional[str] = None
    imu_gyro_topic: Optional[str] = None
    auto: TopicAutoConfig = field(default_factory=TopicAutoConfig)


@dataclass
class ExpectedRatesConfig:
    image_hz: float = 30.0
    imu_hz: float = 200.0


@dataclass
class SamplingConfig:
    rgb_stride: int = 5
    max_rgb_frames: int = 12000


@dataclass
class ThresholdsConfig:
    image_gap_factor: float = 2.5
    imu_gap_factor: float = 5.0
    sync_warn_ms: float = 16.0
    sync_fail_ms: float = 33.0
    imu_window_ms: float = 20.0
    blur_threshold_min: float = 80.0
    low_clip_threshold: float = 0.05
    high_clip_threshold: float = 0.05
    contrast_min: float = 25.0
    depth_invalid_threshold: float = 0.35
    drop_warn_ratio: float = 0.05
    drop_fail_ratio: float = 0.10
    imu_missing_warn_ratio: float = 0.10
    blur_fail_warn_ratio: float = 0.20
    exposure_bad_warn_ratio: float = 0.20
    depth_invalid_mean_warn: float = 0.35
    # Legacy exposure keys retained for backward compatibility.
    # contrast_min/low_clip_threshold/high_clip_threshold are not used by the v1.3
    # exposure classifier (which uses *_warn and dynamic-range based logic).
    low_clip_pixel_value: int = 5
    high_clip_pixel_value: int = 250
    exposure_roi_margin_ratio: float = 0.05
    low_clip_warn: float = 0.20
    high_clip_warn: float = 0.20
    dynamic_range_min: float = 10.0
    median_dark: float = 40.0
    median_bright: float = 215.0


@dataclass
class SegmentConfig:
    max_gap_fill_ms: float = 200.0
    min_segment_seconds: float = 5.0


@dataclass
class GateConfig:
    fail_if_no_segments_min_duration_s: float = 30.0
    gate_warn_floor_error_codes: list[str] = field(
        default_factory=lambda: ["TIMESTAMP_OUT_OF_ORDER_HIGH"]
    )


@dataclass
class IntegrityConfig:
    out_of_order_warn_ratio: float = 0.001


@dataclass
class DecodeConfig:
    warn_on_depth_pixel_decode_failure: bool = False


@dataclass
class DebugConfig:
    export_exposure_csv: bool = True


@dataclass
class QAConfig:
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    expected_rates: ExpectedRatesConfig = field(default_factory=ExpectedRatesConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class TopicOverrides:
    rgb_topic: Optional[str] = None
    depth_topic: Optional[str] = None
    imu_accel_topic: Optional[str] = None
    imu_gyro_topic: Optional[str] = None


@dataclass(frozen=True)
class MessageRecord:
    topic: str
    log_time_ns: int
    publish_time_ns: Optional[int]
    msg: Any
    type_name: Optional[str] = None


@dataclass
class TopicScanInfo:
    message_count: int = 0
    first_log_time_ns: Optional[int] = None
    last_log_time_ns: Optional[int] = None

    @property
    def duration_s(self) -> float:
        if (
            self.first_log_time_ns is None
            or self.last_log_time_ns is None
            or self.last_log_time_ns <= self.first_log_time_ns
        ):
            return 0.0
        return (self.last_log_time_ns - self.first_log_time_ns) / 1_000_000_000.0

    @property
    def approx_rate_hz(self) -> Optional[float]:
        duration = self.duration_s
        if duration <= 0.0 or self.message_count < 2:
            return None
        return (self.message_count - 1) / duration


@dataclass
class SelectedTopics:
    rgb_topic: Optional[str]
    depth_topic: Optional[str]
    imu_accel_topic: Optional[str]
    imu_gyro_topic: Optional[str]
    imu_mode: str


@dataclass
class AnalysisResult:
    report: dict[str, Any]
    gate: str
    output_path: Path
    # Additive convenience fields for CLI and future integrations.
    recommended_action: str = ""
    fail_reasons: list[str] = field(default_factory=list)
    warn_reasons: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    report_path: Path | None = None
