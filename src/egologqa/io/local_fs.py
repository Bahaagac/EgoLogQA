from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class LocalDirNotFound(RuntimeError):
    """Raised when the requested local folder does not exist."""


class LocalDirNotReadable(RuntimeError):
    """Raised when the requested local folder cannot be read."""


class TooManyFiles(RuntimeError):
    """Raised when folder listing exceeds configured file cap."""

    def __init__(self, total_count: int, max_files: int):
        self.total_count = int(total_count)
        self.max_files = int(max_files)
        super().__init__(
            f"Found {self.total_count} .mcap files, which exceeds max_files={self.max_files}."
        )


def _resolve_dir_path(dir_path: str) -> Path:
    expanded = os.path.expandvars(dir_path)
    resolved = Path(expanded).expanduser().resolve()
    return resolved


def list_mcap_files_in_dir(dir_path: str, max_files: int = 500) -> list[dict[str, Any]]:
    """List immediate `.mcap` children from a local directory.

    Returns records with:
      - path: absolute path string
      - name: basename
      - size_bytes: integer file size
      - mtime_ns: integer mtime in nanoseconds
    """
    if max_files <= 0:
        raise ValueError("max_files must be > 0")

    folder = _resolve_dir_path(dir_path)
    if not folder.exists():
        raise LocalDirNotFound(f"Directory not found: {folder}")
    if not folder.is_dir():
        raise LocalDirNotReadable(f"Path is not a directory: {folder}")

    try:
        names = os.listdir(folder)
    except OSError as exc:
        raise LocalDirNotReadable(f"Directory is not readable: {folder}") from exc

    rows: list[dict[str, Any]] = []
    for name in names:
        if not name.lower().endswith(".mcap"):
            continue
        candidate = folder / name
        try:
            if not candidate.is_file():
                continue
            stats = candidate.stat()
        except OSError:
            # Skip unreadable entries; selected file readability is rechecked before run.
            continue

        rows.append(
            {
                "path": str(candidate.resolve()),
                "name": candidate.name,
                "size_bytes": int(stats.st_size),
                "mtime_ns": int(stats.st_mtime_ns),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["name"]).lower(),
            int(row["size_bytes"]),
            int(row["mtime_ns"]),
            str(row["path"]),
        )
    )

    if len(rows) > max_files:
        raise TooManyFiles(total_count=len(rows), max_files=max_files)

    return rows


def is_readable_file(path: str | Path) -> tuple[bool, str | None]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    try:
        with file_path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return False, f"File is not readable: {file_path} ({exc})"
    return True, None
