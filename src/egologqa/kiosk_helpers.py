from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_runs_base_dir(env_value: str | None) -> Path:
    if env_value and env_value.strip():
        return Path(env_value).expanduser()
    return Path.home() / ".cache" / "egologqa" / "runs"


def sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def build_run_basename(repo_id: str, revision: str, hf_path: str) -> str:
    stem = sanitize_component(Path(hf_path).stem)
    rev = sanitize_component(revision)
    digest = hashlib.sha1(f"{repo_id}:{revision}:{hf_path}".encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{rev}_{digest}"


def allocate_run_dir(base_dir: Path, basename: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / basename
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    idx = 2
    while True:
        trial = base_dir / f"{basename}_{idx}"
        if not trial.exists():
            trial.mkdir(parents=True, exist_ok=False)
            return trial
        idx += 1


def write_latest_run_pointer(base_dir: Path, run_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    marker = base_dir / "latest_run.txt"
    marker.write_text(str(run_dir.resolve()), encoding="utf-8")
    return marker


def ensure_writable_dir(path: Path, label: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".egologqa_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"{label} is not writable: {path}") from exc
    finally:
        probe.unlink(missing_ok=True)
    return path


def human_bytes(size_bytes: int | None) -> str:
    if size_bytes is None or size_bytes < 0:
        return "unknown"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.0f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    return f"{size_bytes / (1024**3):.1f} GB"


def build_hf_display_label(path: str, prefix: str, size_bytes: int | None) -> str:
    shown = path
    if prefix and shown.startswith(prefix):
        shown = shown[len(prefix) :]
    shown = Path(shown).name if shown else path
    return f"{shown} ({human_bytes(size_bytes)})"


def make_local_option_label(name: str, size_bytes: int | None) -> str:
    return f"{name} ({human_bytes(size_bytes)})"


def stage_uploaded_mcap(uploaded_file: Any, output_dir: Path) -> Path:
    if uploaded_file is None:
        raise RuntimeError("No uploaded file provided.")

    raw_name = str(getattr(uploaded_file, "name", "") or "uploaded")
    sanitized_name = sanitize_component(Path(raw_name).name)
    staged_stem = Path(sanitized_name).stem or "uploaded"
    staged_name = f"uploaded_{staged_stem}.mcap"
    staged_dir = output_dir / "input"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / staged_name

    data: bytes
    if hasattr(uploaded_file, "getbuffer"):
        data = bytes(uploaded_file.getbuffer())
    elif hasattr(uploaded_file, "getvalue"):
        data = bytes(uploaded_file.getvalue())
    elif hasattr(uploaded_file, "read"):
        data = bytes(uploaded_file.read())
    else:
        raise RuntimeError("Uploaded file object is unsupported.")

    staged_path.write_bytes(data)
    return staged_path


def build_timestamped_run_basename(file_name: str, suffix: str | None = None) -> str:
    stem = sanitize_component(Path(file_name).stem)
    utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = suffix if suffix is not None else secrets.token_hex(2)
    return f"{stem}_{utc_stamp}_{rand}"


def resolve_source_kind(local_uploaded: bool, hf_selected: bool) -> str | None:
    if local_uploaded:
        return "local"
    if hf_selected:
        return "hf"
    return None


def map_error_bucket(exc: Exception) -> str:
    message = f"{exc.__class__.__name__}: {exc}".lower()
    if "huggingface_hub is required" in message or "no module named 'huggingface_hub'" in message:
        return "`huggingface_hub` is required. Install project deps in venv."
    if "401" in message or "403" in message or "authentication failed" in message:
        return "Dataset requires auth. Set `HF_TOKEN`."
    if "404" in message or "not found" in message:
        return "Dataset/revision/path not found."
    if (
        "failed to resolve 'huggingface.co'" in message
        or "name resolution" in message
        or "connectionerror" in message
        or "maxretryerror" in message
        or "network error" in message
        or "timed out" in message
    ):
        return "Cannot reach huggingface.co. Check network/DNS/proxy."
    return "Could not load file list from Hugging Face."
