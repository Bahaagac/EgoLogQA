from __future__ import annotations

from pathlib import Path

import pytest

from egologqa.io.local_fs import (
    LocalDirNotFound,
    LocalDirNotReadable,
    TooManyFiles,
    is_readable_file,
    list_mcap_files_in_dir,
)


def test_list_mcap_files_sorted_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "b.mcap").write_bytes(b"bbbb")
    (tmp_path / "a.mcap").write_bytes(b"aa")
    (tmp_path / "c.mcap").write_bytes(b"c")

    rows = list_mcap_files_in_dir(str(tmp_path))
    assert [row["name"] for row in rows] == ["a.mcap", "b.mcap", "c.mcap"]
    assert [row["size_bytes"] for row in rows] == [2, 4, 1]
    assert all(Path(str(row["path"])).is_absolute() for row in rows)


def test_list_mcap_filters_extension(tmp_path: Path) -> None:
    (tmp_path / "x.mcap").write_bytes(b"x")
    (tmp_path / "y.txt").write_text("y", encoding="utf-8")
    (tmp_path / "z.mcap.tmp").write_text("z", encoding="utf-8")
    rows = list_mcap_files_in_dir(str(tmp_path))
    assert [row["name"] for row in rows] == ["x.mcap"]


def test_too_many_files_limit(tmp_path: Path) -> None:
    (tmp_path / "a.mcap").write_bytes(b"a")
    (tmp_path / "b.mcap").write_bytes(b"b")
    with pytest.raises(TooManyFiles) as exc_info:
        list_mcap_files_in_dir(str(tmp_path), max_files=1)
    assert exc_info.value.total_count == 2
    assert exc_info.value.max_files == 1


def test_missing_directory_raises() -> None:
    with pytest.raises(LocalDirNotFound):
        list_mcap_files_in_dir("~/this/path/does/not/exist/for/EgoLogQA")


def test_unreadable_directory_raises(monkeypatch, tmp_path: Path) -> None:
    def fake_listdir(_):
        raise OSError("permission denied")

    monkeypatch.setattr("egologqa.io.local_fs.os.listdir", fake_listdir)
    with pytest.raises(LocalDirNotReadable):
        list_mcap_files_in_dir(str(tmp_path))


def test_is_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "ok.mcap"
    path.write_bytes(b"abc")
    ok, reason = is_readable_file(path)
    assert ok is True
    assert reason is None

    missing_ok, missing_reason = is_readable_file(tmp_path / "missing.mcap")
    assert missing_ok is False
    assert missing_reason and "File not found" in missing_reason
