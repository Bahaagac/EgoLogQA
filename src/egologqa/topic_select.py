from __future__ import annotations

import re
from typing import Optional

from egologqa.models import QAConfig, SelectedTopics, TopicScanInfo


def select_topics(
    config: QAConfig,
    stats: dict[str, TopicScanInfo],
) -> SelectedTopics:
    if config.topics.mode == "explicit":
        return _select_explicit(config)
    return _select_auto(config, stats)


def _select_explicit(config: QAConfig) -> SelectedTopics:
    accel = config.topics.imu_accel_topic
    gyro = config.topics.imu_gyro_topic
    if accel and gyro and accel == gyro:
        imu_mode = "single_topic_assumed_both"
    elif accel and gyro:
        imu_mode = "dual_topics"
    elif accel or gyro:
        topic = accel or gyro
        accel = topic
        gyro = topic
        imu_mode = "single_topic_assumed_both"
    else:
        imu_mode = "none"
    return SelectedTopics(
        rgb_topic=config.topics.rgb_topic,
        depth_topic=config.topics.depth_topic,
        imu_accel_topic=accel,
        imu_gyro_topic=gyro,
        imu_mode=imu_mode,
    )


def _select_auto(
    config: QAConfig,
    stats: dict[str, TopicScanInfo],
) -> SelectedTopics:
    rgb = _pick_topic(
        stats=stats,
        regex=config.topics.auto.rgb_regex,
        expected_rate_hz=config.expected_rates.image_hz,
    )
    depth = _pick_topic(
        stats=stats,
        regex=config.topics.auto.depth_regex,
        expected_rate_hz=config.expected_rates.image_hz,
    )
    imu_candidates = _pick_topics(
        stats=stats,
        regex=config.topics.auto.imu_regex,
        expected_rate_hz=config.expected_rates.imu_hz,
    )
    accel: Optional[str] = None
    gyro: Optional[str] = None
    imu_mode = "none"
    if len(imu_candidates) >= 2:
        accel = imu_candidates[0]
        gyro = imu_candidates[1]
        imu_mode = "dual_topics"
    elif len(imu_candidates) == 1:
        accel = imu_candidates[0]
        gyro = imu_candidates[0]
        imu_mode = "single_topic_assumed_both"
    return SelectedTopics(
        rgb_topic=rgb,
        depth_topic=depth,
        imu_accel_topic=accel,
        imu_gyro_topic=gyro,
        imu_mode=imu_mode,
    )


def _pick_topic(
    stats: dict[str, TopicScanInfo],
    regex: str,
    expected_rate_hz: float,
) -> Optional[str]:
    candidates = _pick_topics(stats, regex, expected_rate_hz)
    return candidates[0] if candidates else None


def _pick_topics(
    stats: dict[str, TopicScanInfo],
    regex: str,
    expected_rate_hz: float,
) -> list[str]:
    pattern = re.compile(regex, re.IGNORECASE)
    matched = [topic for topic in stats if pattern.search(topic)]
    scored: list[tuple[int, float, str]] = []
    for topic in matched:
        info = stats[topic]
        rate = info.approx_rate_hz
        rate_distance = abs((rate or 0.0) - expected_rate_hz)
        scored.append((-info.message_count, rate_distance, topic))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[2] for item in scored]
