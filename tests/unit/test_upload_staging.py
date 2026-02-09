from __future__ import annotations

from pathlib import Path

import pytest

from egologqa.kiosk_helpers import stage_uploaded_mcap


class _BufferUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def getbuffer(self):
        return memoryview(self._payload)


class _ValueUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def getvalue(self):
        return self._payload


class _ReadUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def read(self):
        return self._payload


class _UnsupportedUpload:
    def __init__(self):
        self.name = "broken.mcap"
        self.size = 0


def test_stage_uploaded_mcap_writes_bytes_and_sanitizes_name(tmp_path: Path) -> None:
    uploaded = _BufferUpload("../../My Clip.mcap", b"abc123")
    staged = stage_uploaded_mcap(uploaded, tmp_path)

    assert staged == tmp_path / "input" / "uploaded_My_Clip.mcap"
    assert staged.read_bytes() == b"abc123"


def test_stage_uploaded_mcap_normalizes_non_mcap_extension(tmp_path: Path) -> None:
    uploaded = _ValueUpload("session.bin", b"payload")
    staged = stage_uploaded_mcap(uploaded, tmp_path)

    assert staged.name == "uploaded_session.mcap"
    assert staged.read_bytes() == b"payload"


def test_stage_uploaded_mcap_supports_read_fallback(tmp_path: Path) -> None:
    uploaded = _ReadUpload("input.mcap", b"xyz")
    staged = stage_uploaded_mcap(uploaded, tmp_path)

    assert staged.exists()
    assert staged.read_bytes() == b"xyz"


def test_stage_uploaded_mcap_rejects_missing_upload(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No uploaded file provided"):
        stage_uploaded_mcap(None, tmp_path)


def test_stage_uploaded_mcap_rejects_unsupported_upload_object(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported"):
        stage_uploaded_mcap(_UnsupportedUpload(), tmp_path)
