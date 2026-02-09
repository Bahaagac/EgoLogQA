#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
PY="${PY:-$REPO/.venv/bin/python}"
CLI="${CLI:-$REPO/.venv/bin/EgoLogQA}"
if [ ! -x "$CLI" ] && [ -x "$REPO/.venv/bin/egologqa" ]; then
    CLI="$REPO/.venv/bin/egologqa"
fi
CFG="${CFG:-$REPO/configs/microagi00_ros2.yaml}"
LIST="${LIST:-$REPO/qa_mcap_list.txt}"
BASELINE_ROOT="${BASELINE_ROOT:-$REPO/baselines}"
BEFORE_ROOT="$BASELINE_ROOT/exposure_lowclip_before"
AFTER_ROOT="$BASELINE_ROOT/exposure_lowclip_after"
BEFORE_SUMMARY="$BASELINE_ROOT/exposure_lowclip_before_summary.jsonl"
AFTER_SUMMARY="$BASELINE_ROOT/exposure_lowclip_after_summary.jsonl"
KNOWN_DARK_FILE_ID="${KNOWN_DARK_FILE_ID:-}"
MODE="${1:-all}"

log() {
    printf '[lowclip-regression] %s\n' "$*" >&2
}

fail() {
    printf '[lowclip-regression][FAIL] %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [ -f "$path" ] || fail "required file missing: $path"
}

stem_from_path() {
    local path="$1"
    local name
    name="$(basename "$path")"
    printf '%s\n' "${name%.mcap}"
}

run_phase() {
    local phase="$1"
    local root="$2"
    local mcap stem out report

    require_file "$LIST"
    require_file "$CFG"
    rm -rf "$root"
    mkdir -p "$root"

    while IFS= read -r mcap || [ -n "$mcap" ]; do
        case "$mcap" in
            ""|\#*) continue ;;
        esac
        [ -f "$mcap" ] || fail "mcap path not found: $mcap"

        stem="$(stem_from_path "$mcap")"
        out="$root/$stem"
        mkdir -p "$out"
        log "[$phase] analyze: $mcap"
        "$CLI" analyze --input "$mcap" --config "$CFG" --output "$out" >"$out/analyze.log" 2>&1
        report="$out/report.json"
        [ -f "$report" ] || fail "[$phase] missing report.json: $report"
    done < "$LIST"
}

summarize_phase() {
    local phase="$1"
    local root="$2"
    local summary="$3"
    local mcap stem report low_clip_dir img_count

    require_file "$LIST"
    : > "$summary"
    while IFS= read -r mcap || [ -n "$mcap" ]; do
        case "$mcap" in
            ""|\#*) continue ;;
        esac
        stem="$(stem_from_path "$mcap")"
        report="$root/$stem/report.json"
        [ -f "$report" ] || fail "[$phase] missing report.json during summary: $report"

        low_clip_dir="$root/$stem/debug/exposure_low_clip_frames"
        img_count=0
        if [ -d "$low_clip_dir" ]; then
            img_count="$(find "$low_clip_dir" -type f -name '*.jpg' | wc -l | tr -d ' ')"
        fi

        jq -nc \
            --arg file_id "$stem" \
            --arg gate "$(jq -r '.gate.gate' "$report")" \
            --argjson warn_reasons "$(jq -c '.gate.warn_reasons // []' "$report")" \
            --argjson exposure_bad_ratio "$(jq -c '.metrics.exposure_bad_ratio' "$report")" \
            --argjson low_clip_count "$(jq -c '.metrics.exposure_bad_reason_counts.low_clip // 0' "$report")" \
            --argjson evidence_low_clip_images_count "$img_count" \
            '{
                file_id: $file_id,
                gate: $gate,
                warn_reasons: $warn_reasons,
                exposure_bad_ratio: $exposure_bad_ratio,
                low_clip_count: $low_clip_count,
                evidence_low_clip_images_count: $evidence_low_clip_images_count
            }' >> "$summary"
    done < "$LIST"
    log "[$phase] wrote summary: $summary"
}

compare_summaries() {
    require_file "$BEFORE_SUMMARY"
    require_file "$AFTER_SUMMARY"
    KNOWN_DARK_FILE_ID="$KNOWN_DARK_FILE_ID" "$PY" - "$BEFORE_SUMMARY" "$AFTER_SUMMARY" <<'PY'
import json
import os
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["file_id"]): row for row in rows}


before_path = Path(sys.argv[1])
after_path = Path(sys.argv[2])
known_dark = os.environ.get("KNOWN_DARK_FILE_ID", "").strip()

before = load(before_path)
after = load(after_path)
if set(before.keys()) != set(after.keys()):
    missing_before = sorted(set(after.keys()) - set(before.keys()))
    missing_after = sorted(set(before.keys()) - set(after.keys()))
    print("FAIL: summary file_id mismatch")
    print("missing_in_before:", missing_before)
    print("missing_in_after:", missing_after)
    sys.exit(1)

failures: list[str] = []
manual_flags: list[str] = []

for file_id in sorted(before.keys()):
    br = before[file_id]
    ar = after[file_id]

    if br["gate"] == "PASS" and ar["gate"] != "PASS":
        failures.append(f"{file_id}: gate worsened PASS -> {ar['gate']}")

    if (
        br["gate"] == "PASS"
        and ar["gate"] == "WARN"
        and "WARN_EXPOSURE_BAD_RATIO_GT_WARN" in ar.get("warn_reasons", [])
    ):
        failures.append(f"{file_id}: unexpected exposure PASS -> WARN")

    if int(ar["evidence_low_clip_images_count"]) > int(br["evidence_low_clip_images_count"]):
        failures.append(
            f"{file_id}: low-clip evidence images increased "
            f"({br['evidence_low_clip_images_count']} -> {ar['evidence_low_clip_images_count']})"
        )

    if (
        int(ar["low_clip_count"]) > int(br["low_clip_count"])
        and int(ar["evidence_low_clip_images_count"]) > int(br["evidence_low_clip_images_count"])
    ):
        failures.append(
            f"{file_id}: low_clip_count increased with evidence increase "
            f"({br['low_clip_count']} -> {ar['low_clip_count']})"
        )

    if (
        br["gate"] == "WARN"
        and ar["gate"] == "PASS"
        and "WARN_EXPOSURE_BAD_RATIO_GT_WARN" in br.get("warn_reasons", [])
    ):
        manual_flags.append(file_id)

if known_dark:
    row = after.get(known_dark)
    if row is None:
        failures.append(f"KNOWN_DARK_FILE_ID not found in summaries: {known_dark}")
    elif int(row["low_clip_count"]) < 1:
        failures.append(f"{known_dark}: known-dark case no longer triggers low_clip")
elif manual_flags:
    failures.append(
        "WARN->PASS transitions involving exposure detected with no KNOWN_DARK_FILE_ID; "
        "manual evidence confirmation required."
    )
    for file_id in manual_flags:
        failures.append(f"{file_id}: WARN(exposure) -> PASS requires manual confirmation")

if failures:
    print("FAIL: regression checks failed")
    for item in failures:
        print(f"- {item}")
    sys.exit(1)

print("PASS: regression checks passed")
PY
}

case "$MODE" in
    before)
        run_phase "before" "$BEFORE_ROOT"
        summarize_phase "before" "$BEFORE_ROOT" "$BEFORE_SUMMARY"
        ;;
    after)
        run_phase "after" "$AFTER_ROOT"
        summarize_phase "after" "$AFTER_ROOT" "$AFTER_SUMMARY"
        ;;
    compare)
        compare_summaries
        ;;
    all)
        run_phase "before" "$BEFORE_ROOT"
        summarize_phase "before" "$BEFORE_ROOT" "$BEFORE_SUMMARY"
        run_phase "after" "$AFTER_ROOT"
        summarize_phase "after" "$AFTER_ROOT" "$AFTER_SUMMARY"
        compare_summaries
        ;;
    *)
        fail "unknown mode '$MODE' (expected: before|after|compare|all)"
        ;;
esac
