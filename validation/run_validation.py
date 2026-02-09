from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egologqa.config import load_config
from egologqa.io.hf_fetch import list_mcap_files, resolve_cached_file
from egologqa.pipeline import analyze_file
from egologqa.report import sanitize_json_value


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "run"


def _run_dir_name(hf_path: str) -> str:
    stem = _slug(Path(hf_path).stem)
    digest = hashlib.sha1(hf_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json_value(payload),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic EgoLogQA validation batch")
    parser.add_argument("--repo-id", default="MicroAGI-Labs/MicroAGI00")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="raw_mcaps/")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--config", default="configs/microagi00_ros2.yaml")
    parser.add_argument("--output-root", default="validation")
    parser.add_argument(
        "--cache-dir",
        default=str(Path("~/.cache/EgoLogQA/hf_mcaps").expanduser()),
    )
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)

    token = args.token if args.token is not None else os.getenv("HF_TOKEN")
    cfg = load_config(args.config)

    output_root = (REPO_ROOT / args.output_root).resolve()
    runs_dir = output_root / "runs"
    results_dir = output_root / "results"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    listed = list_mcap_files(
        repo_id=args.repo_id,
        revision=args.revision,
        token=token,
        prefix=args.prefix,
    )
    selected_paths = [row["path"] for row in listed[: max(0, args.limit)]]

    run_rows: list[dict[str, object]] = []
    for hf_path in selected_paths:
        local_path = resolve_cached_file(
            repo_id=args.repo_id,
            revision=args.revision,
            file_path=hf_path,
            token=token,
            cache_dir=args.cache_dir,
        )
        run_dir = runs_dir / _run_dir_name(hf_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        result = analyze_file(
            input_path=local_path,
            output_dir=run_dir,
            config=cfg,
        )
        run_rows.append(
            {
                "hf_path": hf_path,
                "gate": result.gate,
                "recommended_action": result.recommended_action,
                "run_dir": run_dir.relative_to(REPO_ROOT).as_posix(),
                "report": (run_dir / "report.json").relative_to(REPO_ROOT).as_posix(),
            }
        )

    run_rows.sort(key=lambda row: str(row["hf_path"]))
    run_index = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "prefix": args.prefix,
        "count": len(run_rows),
        "runs": run_rows,
    }
    _write_json(results_dir / "run_index.json", run_index)
    print(f"Wrote {len(run_rows)} runs to {(results_dir / 'run_index.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
