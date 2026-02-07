from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

from egologqa.models import MessageRecord, TopicScanInfo


class MessageSource(Protocol):
    def scan_topics(self) -> dict[str, TopicScanInfo]:
        ...

    def iter_messages(self, topics: Optional[set[str]] = None) -> Iterable[MessageRecord]:
        ...


@dataclass
class InMemoryMessageSource:
    records: list[MessageRecord]

    def scan_topics(self) -> dict[str, TopicScanInfo]:
        stats: dict[str, TopicScanInfo] = {}
        for rec in self.records:
            info = stats.setdefault(rec.topic, TopicScanInfo())
            info.message_count += 1
            if info.first_log_time_ns is None or rec.log_time_ns < info.first_log_time_ns:
                info.first_log_time_ns = rec.log_time_ns
            if info.last_log_time_ns is None or rec.log_time_ns > info.last_log_time_ns:
                info.last_log_time_ns = rec.log_time_ns
        return stats

    def iter_messages(self, topics: Optional[set[str]] = None) -> Iterable[MessageRecord]:
        for rec in self.records:
            if topics is not None and rec.topic not in topics:
                continue
            yield rec


class MCapMessageSource:
    def __init__(self, file_path: str | Path):
        self.file_path = str(file_path)

    def scan_topics(self) -> dict[str, TopicScanInfo]:
        stats: dict[str, TopicScanInfo] = {}
        for rec in self.iter_messages():
            info = stats.setdefault(rec.topic, TopicScanInfo())
            info.message_count += 1
            if info.first_log_time_ns is None or rec.log_time_ns < info.first_log_time_ns:
                info.first_log_time_ns = rec.log_time_ns
            if info.last_log_time_ns is None or rec.log_time_ns > info.last_log_time_ns:
                info.last_log_time_ns = rec.log_time_ns
        return stats

    def iter_messages(self, topics: Optional[set[str]] = None) -> Iterable[MessageRecord]:
        try:
            from mcap_ros2.reader import read_ros2_messages
        except Exception as exc:  # pragma: no cover - import path depends on env
            raise RuntimeError(
                "mcap-ros2-support is required to read MCAP files"
            ) from exc

        for item in read_ros2_messages(self.file_path):
            schema, channel, log_time_ns, publish_time_ns, ros_msg = _extract_record_fields(item)
            topic = getattr(channel, "topic", None)
            if topic is None:
                continue
            if topics is not None and topic not in topics:
                continue
            type_name = getattr(schema, "name", None) if schema is not None else None
            yield MessageRecord(
                topic=topic,
                log_time_ns=int(log_time_ns or 0),
                publish_time_ns=publish_time_ns,
                msg=ros_msg,
                type_name=type_name,
            )


def _extract_record_fields(item: object) -> tuple[object | None, object | None, int, int | None, object]:
    """Normalize outputs from different mcap-ros2-support versions."""
    # Older API: (schema, channel, message, ros_msg)
    if isinstance(item, tuple) and len(item) == 4:
        schema, channel, message, ros_msg = item
        log_time_ns = _to_int(getattr(message, "log_time", 0), default=0)
        publish_time_ns = _to_optional_int(getattr(message, "publish_time", None))
        return schema, channel, log_time_ns, publish_time_ns, ros_msg

    # Newer API: McapROS2Message object.
    schema = getattr(item, "schema", None)
    channel = getattr(item, "channel", None)
    ros_msg = getattr(item, "ros_msg", None)
    log_time_ns = _to_optional_int(getattr(item, "log_time_ns", None))
    publish_time_ns = _to_optional_int(getattr(item, "publish_time_ns", None))

    # Some versions expose timing under nested "message" metadata.
    message_meta = getattr(item, "message", None)
    if log_time_ns is None and message_meta is not None:
        log_time_ns = _to_optional_int(getattr(message_meta, "log_time", None))
    if publish_time_ns is None and message_meta is not None:
        publish_time_ns = _to_optional_int(getattr(message_meta, "publish_time", None))

    return schema, channel, int(log_time_ns or 0), publish_time_ns, ros_msg


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None
