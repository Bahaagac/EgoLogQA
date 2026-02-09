from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from egologqa.io.hf_fetch import download_to_temp, list_mcap_files, resolve_cached_file


def test_list_mcap_files_filters_and_sorts(monkeypatch) -> None:
    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, **kwargs):
            return [
                SimpleNamespace(path="z/clip_b.mcap", size=20, oid="b"),
                SimpleNamespace(path="a/readme.txt", size=5, oid="x"),
                {"path": "a/clip_a.mcap", "size": 10, "oid": "a"},
            ]

    def fake_load():
        return FakeApi, (lambda **_: "unused"), object

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)

    files = list_mcap_files(
        "MicroAGI-Labs/MicroAGI00",
        revision="main",
        token="t",
        prefix="a/",
    )
    assert [f["path"] for f in files] == ["a/clip_a.mcap"]
    assert files[0]["size_bytes"] == 10


def test_list_mcap_files_prefix_empty_returns_all_matching(monkeypatch) -> None:
    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, **kwargs):
            return [
                SimpleNamespace(path="raw_mcaps/z/clip_b.mcap", size=20, oid="b"),
                SimpleNamespace(path="raw_mcaps/a/readme.txt", size=5, oid="x"),
                {"path": "raw_mcaps/a/clip_a.mcap", "size": 10, "oid": "a"},
            ]

    def fake_load():
        return FakeApi, (lambda **_: "unused"), object

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)

    files = list_mcap_files("MicroAGI-Labs/MicroAGI00", revision="main", token="t", prefix="")
    assert [f["path"] for f in files] == ["raw_mcaps/a/clip_a.mcap", "raw_mcaps/z/clip_b.mcap"]
    assert files[1]["oid"] == "b"


def test_list_mcap_files_missing_size_maps_to_none(monkeypatch) -> None:
    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, **kwargs):
            del kwargs
            return [SimpleNamespace(path="raw_mcaps/clip_no_size.mcap", oid="x")]

    def fake_load():
        return FakeApi, (lambda **_: "unused"), object

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    files = list_mcap_files("MicroAGI-Labs/MicroAGI00", prefix="raw_mcaps/")
    assert files[0]["size_bytes"] is None


def test_download_to_temp_streams_and_reports_progress(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "6"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def iter_content(self, chunk_size=0):
            del chunk_size
            yield b"abc"
            yield b"def"

    def fake_load():
        return object, (lambda **_: "https://example.test/file.mcap"), object

    def fake_get(url, headers=None, stream=True, timeout=None):
        del headers, stream, timeout
        assert url == "https://example.test/file.mcap"
        return FakeResponse()

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    progress = []
    path = download_to_temp(
        repo_id="r",
        revision="main",
        file_path="x.mcap",
        token=None,
        progress_cb=lambda d, t: progress.append((d, t)),
    )
    try:
        assert path.exists()
        assert path.read_bytes() == b"abcdef"
        assert progress[0] == (0, 6)
        assert progress[-1] == (6, 6)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "Authentication failed"),
        (404, "File not found"),
    ],
)
def test_download_to_temp_http_failures(monkeypatch, status_code: int, expected: str) -> None:
    class FakeResponse:
        headers = {}
        text = "failure"

        def __init__(self, code: int):
            self.status_code = code

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def iter_content(self, chunk_size=0):
            del chunk_size
            return iter(())

    def fake_load():
        return object, (lambda **_: "https://example.test/file.mcap"), object

    def fake_get(*args, **kwargs):
        del args, kwargs
        return FakeResponse(status_code)

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    with pytest.raises(RuntimeError, match=expected):
        download_to_temp("repo", "main", "f.mcap")


def test_download_to_temp_network_failure(monkeypatch) -> None:
    def fake_load():
        return object, (lambda **_: "https://example.test/file.mcap"), object

    def fake_get(*args, **kwargs):
        del args, kwargs
        raise requests.RequestException("boom")

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Network error"):
        download_to_temp("repo", "main", "f.mcap")


def test_resolve_cached_file_cache_miss_downloads_once(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "6"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def iter_content(self, chunk_size=0):
            del chunk_size
            yield b"abc"
            yield b"def"

    def fake_load():
        return object, (lambda **_: "https://example.test/raw_mcaps/a.mcap"), object

    calls: list[tuple[Any, Any, Any, Any]] = []

    def fake_get(url, headers=None, stream=True, timeout=None):
        calls.append((url, headers, stream, timeout))
        return FakeResponse()

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    progress: list[tuple[int, int | None]] = []
    resolved = resolve_cached_file(
        repo_id="r",
        revision="main",
        file_path="raw_mcaps/a.mcap",
        token=None,
        cache_dir=tmp_path,
        progress_cb=lambda d, t: progress.append((d, t)),
    )

    assert resolved.exists()
    assert resolved.read_bytes() == b"abcdef"
    assert len(calls) == 1
    assert calls[0][0] == "https://example.test/raw_mcaps/a.mcap"
    assert calls[0][2] is True
    assert calls[0][3] == (10.0, 30.0)
    assert progress[0] == (0, None)
    assert progress[1] == (0, 6)
    assert progress[-1] == (6, 6)


def test_resolve_cached_file_cache_hit_skips_download(monkeypatch, tmp_path: Path) -> None:
    existing = tmp_path / "raw_mcaps" / "a.mcap"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"cached")

    def fake_load():
        return object, (lambda **_: "https://example.test/raw_mcaps/a.mcap"), object

    def fake_get(**kwargs):
        del kwargs
        raise AssertionError("requests.get should not be called on cache hit")

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    resolved = resolve_cached_file(
        repo_id="r",
        revision="main",
        file_path="raw_mcaps/a.mcap",
        token=None,
        cache_dir=tmp_path,
    )
    assert resolved == existing


def test_resolve_cached_file_auth_error(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        status_code = 401
        headers = {}
        text = "unauthorized"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def iter_content(self, chunk_size=0):
            del chunk_size
            return iter(())

    def fake_load():
        return object, (lambda **_: "https://example.test/raw_mcaps/a.mcap"), object

    def fake_get(*args, **kwargs):
        del args, kwargs
        return FakeResponse()

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Authentication failed"):
        resolve_cached_file(
            repo_id="r",
            revision="main",
            file_path="raw_mcaps/a.mcap",
            token="bad",
            cache_dir=tmp_path,
        )


def test_resolve_cached_file_timeout_retries_then_fails(monkeypatch, tmp_path: Path) -> None:
    def fake_load():
        return object, (lambda **_: "https://example.test/raw_mcaps/a.mcap"), object

    attempts = {"count": 0}

    def fake_get(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        raise requests.Timeout("read timed out")

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)
    monkeypatch.setenv("EGOLOGQA_HF_DOWNLOAD_RETRIES", "2")
    monkeypatch.setenv("EGOLOGQA_HF_RETRY_BACKOFF_S", "0")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        resolve_cached_file(
            repo_id="r",
            revision="main",
            file_path="raw_mcaps/a.mcap",
            token=None,
            cache_dir=tmp_path,
        )

    assert attempts["count"] == 3


def test_resolve_cached_file_uses_timeout_env(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "4"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def iter_content(self, chunk_size=0):
            del chunk_size
            yield b"data"

    def fake_load():
        return object, (lambda **_: "https://example.test/raw_mcaps/a.mcap"), object

    seen_timeouts: list[tuple[float, float]] = []

    def fake_get(url, headers=None, stream=True, timeout=None):
        del url, headers, stream
        if isinstance(timeout, tuple):
            seen_timeouts.append((float(timeout[0]), float(timeout[1])))
        return FakeResponse()

    monkeypatch.setattr("egologqa.io.hf_fetch._load_hf_clients", fake_load)
    monkeypatch.setattr("egologqa.io.hf_fetch.requests.get", fake_get)
    monkeypatch.setenv("EGOLOGQA_HF_CONNECT_TIMEOUT_S", "2.5")
    monkeypatch.setenv("EGOLOGQA_HF_READ_TIMEOUT_S", "7")
    monkeypatch.setenv("EGOLOGQA_HF_DOWNLOAD_RETRIES", "0")

    resolved = resolve_cached_file(
        repo_id="r",
        revision="main",
        file_path="raw_mcaps/a.mcap",
        token=None,
        cache_dir=tmp_path,
    )
    assert resolved.exists()
    assert seen_timeouts == [(2.5, 7.0)]
