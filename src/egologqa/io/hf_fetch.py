from __future__ import annotations

import tempfile
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
    _, _, hf_hub_download = _load_hf_clients()
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    candidate = cache_root / file_path
    if candidate.exists():
        if progress_cb is not None:
            try:
                size = candidate.stat().st_size
            except OSError:
                size = 0
            progress_cb(size, size if size > 0 else None)
        return candidate
    if progress_cb is not None:
        progress_cb(0, None)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            revision=revision,
            token=token or None,
            local_dir=str(cache_root),
        )
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message:
            raise RuntimeError(
                f"Authentication failed while fetching '{file_path}'. "
                "Provide a valid Hugging Face token for private/gated datasets."
            ) from exc
        if "404" in message:
            raise RuntimeError(
                f"File '{file_path}' was not found in dataset '{repo_id}' revision '{revision}'."
            ) from exc
        if "timed out" in message.lower():
            raise RuntimeError(
                f"Network timeout while fetching '{file_path}'. Please retry."
            ) from exc
        raise RuntimeError(
            f"Failed to resolve cached file '{file_path}' from dataset '{repo_id}': {exc}"
        ) from exc

    resolved = Path(local_path)
    if progress_cb is not None:
        try:
            size = resolved.stat().st_size
        except OSError:
            size = 0
        progress_cb(size, size if size > 0 else None)
    return resolved
