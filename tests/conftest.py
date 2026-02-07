from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeStamp:
    sec: int
    nanosec: int


@dataclass
class FakeHeader:
    stamp: FakeStamp


@dataclass
class FakeMessage:
    header: FakeHeader | None = None
    data: bytes = b""
    format: str = ""


def make_message_from_ns(t_ns: int, data: bytes = b"", fmt: str = "") -> FakeMessage:
    sec = t_ns // 1_000_000_000
    nsec = t_ns % 1_000_000_000
    return FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=sec, nanosec=nsec)), data=data, format=fmt)
