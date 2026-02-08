from __future__ import annotations

from egologqa.time import extract_header_stamp_ns, extract_stamp_ns
from tests.conftest import FakeMessage, FakeHeader, FakeStamp


def test_extract_uses_header_stamp_when_valid() -> None:
    msg = FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=2, nanosec=5)))
    t_ns, source, used_fallback = extract_stamp_ns(msg, 100)
    assert t_ns == 2_000_000_005
    assert source == "header"
    assert not used_fallback


def test_extract_falls_back_when_header_zero() -> None:
    msg = FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=0, nanosec=0)))
    t_ns, source, used_fallback = extract_stamp_ns(msg, 333)
    assert t_ns == 333
    assert source == "log_time"
    assert used_fallback


def test_extract_returns_invalid_when_both_invalid() -> None:
    msg = FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=0, nanosec=0)))
    t_ns, source, used_fallback = extract_stamp_ns(msg, 0)
    assert t_ns == 0
    assert source == "invalid"
    assert used_fallback


def test_extract_header_stamp_ns_valid_only() -> None:
    msg = FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=3, nanosec=7)))
    assert extract_header_stamp_ns(msg) == 3_000_000_007


def test_extract_header_stamp_ns_invalid_values_return_zero() -> None:
    class BadStamp:
        sec = "bad"
        nanosec = 1

    class BadHeader:
        stamp = BadStamp()

    class BadMsg:
        header = BadHeader()

    assert extract_header_stamp_ns(FakeMessage(header=FakeHeader(stamp=FakeStamp(sec=0, nanosec=0)))) == 0
    assert extract_header_stamp_ns(BadMsg()) == 0
