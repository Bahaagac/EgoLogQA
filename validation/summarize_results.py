from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egologqa.report import sanitize_json_value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hf_path = (row.get("hf_path") or "").strip()
            if not hf_path:
                continue
            labels[hf_path] = {
                "human_label": (row.get("human_label") or "").strip(),
                "human_recommended_action": (row.get("human_recommended_action") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
                "reviewer": (row.get("reviewer") or "").strip(),
                "reviewed_at_utc": (row.get("reviewed_at_utc") or "").strip(),
            }
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize EgoLogQA validation runs")
    parser.add_argument("--output-root", default="validation")
    parser.add_argument("--labels", default="validation/labels.csv")
    args = parser.parse_args(argv)

    output_root = (REPO_ROOT / args.output_root).resolve()
    results_dir = output_root / "results"
    run_index_path = results_dir / "run_index.json"
    if not run_index_path.exists():
        raise SystemExit(f"Missing run index: {run_index_path}")

    run_index = _read_json(run_index_path)
    runs = sorted(run_index.get("runs", []), key=lambda row: str(row.get("hf_path", "")))
    labels = _load_labels((REPO_ROOT / args.labels).resolve())

    summary_rows: list[dict[str, str]] = []
    gate_counts: Counter[str] = Counter()
    warn_reason_counts: Counter[str] = Counter()
    confusion: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in runs:
        hf_path = str(row.get("hf_path", ""))
        report_rel = str(row.get("report", ""))
        report_path = (REPO_ROOT / report_rel).resolve()
        report = _read_json(report_path) if report_path.exists() else {}
        gate = str(report.get("gate", {}).get("gate", row.get("gate", "")))
        tool_action = str(
            report.get("gate", {}).get("recommended_action", row.get("recommended_action", ""))
        )
        warn_reasons = list(report.get("gate", {}).get("warn_reasons", []))
        fail_reasons = list(report.get("gate", {}).get("fail_reasons", []))

        gate_counts[gate] += 1
        warn_reason_counts.update(str(x) for x in warn_reasons)

        label = labels.get(hf_path, {})
        human_action = label.get("human_recommended_action", "")
        if human_action:
            confusion[(human_action, tool_action)] += 1

        summary_rows.append(
            {
                "hf_path": hf_path,
                "gate": gate,
                "tool_recommended_action": tool_action,
                "human_recommended_action": human_action,
                "match": "1" if human_action and human_action == tool_action else "0",
                "warn_reasons": ";".join(str(x) for x in warn_reasons),
                "fail_reasons": ";".join(str(x) for x in fail_reasons),
            }
        )

    summary_rows.sort(key=lambda r: r["hf_path"])

    summary_csv = results_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "hf_path",
            "gate",
            "tool_recommended_action",
            "human_recommended_action",
            "match",
            "warn_reasons",
            "fail_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    confusion_rows = sorted(
        (
            {
                "human_recommended_action": human,
                "tool_recommended_action": tool,
                "count": count,
            }
            for (human, tool), count in confusion.items()
        ),
        key=lambda r: (r["human_recommended_action"], r["tool_recommended_action"]),
    )
    _write_json(results_dir / "confusion_matrix.json", {"rows": confusion_rows})

    top_warn = warn_reason_counts.most_common(10)
    summary_md = results_dir / "summary.md"
    lines = [
        "# Validation Summary",
        "",
        f"- Dataset: `{run_index.get('repo_id', '')}`",
        f"- Revision: `{run_index.get('revision', '')}`",
        f"- Prefix: `{run_index.get('prefix', '')}`",
        f"- Runs analyzed: `{len(summary_rows)}`",
        "",
        "## Gate Counts",
        "",
    ]
    if gate_counts:
        for gate, count in sorted(gate_counts.items()):
            lines.append(f"- {gate}: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Top WARN Reasons", ""])
    if top_warn:
        for code, count in top_warn:
            lines.append(f"- {code}: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Confusion Matrix (Human vs Tool Action)", ""])
    if confusion_rows:
        for row in confusion_rows:
            lines.append(
                f"- {row['human_recommended_action']} -> {row['tool_recommended_action']}: {row['count']}"
            )
    else:
        lines.append("- No labeled rows found. Create `validation/labels.csv` from `validation/labels_template.csv`.")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {(summary_md).relative_to(REPO_ROOT)} and {(summary_csv).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
