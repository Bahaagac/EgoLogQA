from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests


DownloadProgressCallback = Callable[[int, int | None], None]


def _load_hf_clients() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    except Exception as exc:  # pragma: no cover - import error surface
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face source mode. "
            "Install project dependencies and retry."
        ) from exc
    return HfApi, hf_hub_url, hf_hub_download


def _as_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return entry
    return {
        "path": getattr(entry, "path", None),
        "size": getattr(entry, "size", None),
        "oid": getattr(entry, "oid", None),
        "lastCommit": getattr(entry, "lastCommit", None),
        "last_commit": getattr(entry, "last_commit", None),
    }


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _download_settings() -> tuple[float, float, int, float]:
    connect_timeout_s = _env_float("EGOLOGQA_HF_CONNECT_TIMEOUT_S", default=10.0, minimum=0.1)
    read_timeout_s = _env_float("EGOLOGQA_HF_READ_TIMEOUT_S", default=30.0, minimum=0.1)
    retry_count = _env_int("EGOLOGQA_HF_DOWNLOAD_RETRIES", default=1, minimum=0)
    retry_backoff_s = _env_float("EGOLOGQA_HF_RETRY_BACKOFF_S", default=1.0, minimum=0.0)
    return connect_timeout_s, read_timeout_s, retry_count, retry_backoff_s


def list_mcap_files(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
    prefix: str = "raw_mcaps/",
) -> list[dict[str, Any]]:
    HfApi, _, _ = _load_hf_clients()
    api = HfApi(token=token or None)
    list_prefix = prefix.rstrip("/") if prefix else None
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=list_prefix,
            recursive=True,
            expand=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to list files for dataset '{repo_id}' at revision '{revision}': {exc}"
        ) from exc

    files: list[dict[str, Any]] = []
    for entry in entries:
        data = _as_dict(entry)
        path = data.get("path")
        if not isinstance(path, str):
            continue
        if prefix and not path.startswith(prefix):
            continue
        if not path.lower().endswith(".mcap"):
            continue
        files.append(
            {
                "path": path,
                "size_bytes": data.get("size"),
                "oid": data.get("oid"),
                "lastCommit": data.get("lastCommit") or data.get("last_commit"),
            }
        )
    return sorted(files, key=lambda row: row["path"])


def _raise_for_http_error(status_code: int, file_path: str, response_text: str) -> None:
    if status_code in (401, 403):
        raise RuntimeError(
            f"Authentication failed (HTTP {status_code}) while downloading '{file_path}'. "
            "Provide a valid Hugging Face token for gated/private datasets."
        )
    if status_code == 404:
        raise RuntimeError(
            f"File not found (HTTP 404) while downloading '{file_path}'. "
            "Check dataset, revision, and file path."
        )
    raise RuntimeError(
        f"Download failed (HTTP {status_code}) for '{file_path}': {response_text[:240]}"
    )


def download_to_temp(
    repo_id: str,
    revision: str,
    file_path: str,
    token: str | None = None,
    progress_cb: DownloadProgressCallback | None = None,
) -> Path:
    _, hf_hub_url, _ = _load_hf_clients()
    url = hf_hub_url(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        revision=revision,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mcap")
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    downloaded = 0
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(10, 300)) as resp:
            if resp.status_code != 200:
                _raise_for_http_error(resp.status_code, file_path, resp.text or "")
            total_raw = resp.headers.get("Content-Length")
            total = int(total_raw) if total_raw and total_raw.isdigit() else None
            if progress_cb is not None:
                progress_cb(downloaded, total)
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total)
        return tmp_path
    except requests.RequestException as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Network error while downloading '{file_path}' from Hugging Face: {exc}"
        ) from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def resolve_cached_file(
    repo_id: str,
    revision: str,
    file_path: str,
    token: str | None,
    cache_dir: str | Path,
    progress_cb: DownloadProgressCallback | None = None,
) -> Path:
    _, hf_hub_url, _ = _load_hf_clients()
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    candidate = cache_root / file_path

    if candidate.exists():
        try:
            size = candidate.stat().st_size
        except OSError:
            size = 0
        if size > 0:
            if progress_cb is not None:
                progress_cb(size, size)
            return candidate
        candidate.unlink(missing_ok=True)

    if progress_cb is not None:
        progress_cb(0, None)

    url = hf_hub_url(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        revision=revision,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    connect_timeout_s, read_timeout_s, retry_count, retry_backoff_s = _download_settings()
    attempts = max(1, retry_count + 1)
    candidate.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        downloaded = 0
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mcap.part",
            dir=str(candidate.parent),
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(connect_timeout_s, read_timeout_s),
            ) as resp:
                if resp.status_code != 200:
                    _raise_for_http_error(resp.status_code, file_path, resp.text or "")
                total_raw = resp.headers.get("Content-Length")
                total = int(total_raw) if total_raw and total_raw.isdigit() else None
                if progress_cb is not None:
                    progress_cb(downloaded, total)
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb is not None:
                            progress_cb(downloaded, total)

            tmp_path.replace(candidate)
            if progress_cb is not None:
                try:
                    size = candidate.stat().st_size
                except OSError:
                    size = 0
                progress_cb(size, size if size > 0 else None)
            return candidate

        except requests.Timeout as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt < attempts:
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            raise RuntimeError(
                f"Network timeout while fetching '{file_path}' from dataset '{repo_id}' "
                f"after {attempts} attempts. Please retry or use Local disk mode."
            ) from exc
        except requests.RequestException as exc:
            tmp_path.unlink(missing_ok=True)
            msg = str(exc).lower()
            if "timed out" in msg and attempt < attempts:
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s * attempt)
                continue
            if "timed out" in msg:
                raise RuntimeError(
                    f"Network timeout while fetching '{file_path}' from dataset '{repo_id}' "
                    f"after {attempts} attempts. Please retry or use Local disk mode."
                ) from exc
            raise RuntimeError(
                f"Network error while fetching '{file_path}' from dataset '{repo_id}': {exc}"
            ) from exc
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    raise RuntimeError(
        f"Network timeout while fetching '{file_path}' from dataset '{repo_id}'."
    )
