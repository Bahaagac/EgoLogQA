#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
PY="$REPO/.venv/bin/python"
CLI="$REPO/.venv/bin/egologqa"
BASECFG="$REPO/configs/microagi00_ros2.yaml"
OUTROOT="$REPO/out/verify4"

M1="${M1:-$HOME/.cache/egologqa/hf_mcaps/raw_mcaps/Bakery_Food_Preparation_15f719ff.mcap}"
M2="${M2:-$HOME/.cache/egologqa/hf_mcaps/raw_mcaps/Bedroom_Bed_Making_55e608cd.mcap}"
M3="${M3:-$HOME/Downloads/Desktop_Hardware_Assembly_9c098411.mcap}"
M4="${M4:-$HOME/Downloads/Folding_Clothing_Items_30a47f4f.mcap}"
MCAPS=("$M1" "$M2" "$M3" "$M4")

RUN_HARNESS_DETERMINISM="${RUN_HARNESS_DETERMINISM:-1}"
RUN_HARNESS_UI="${RUN_HARNESS_UI:-1}"
RUN_HARNESS_VALIDATION="${RUN_HARNESS_VALIDATION:-0}"
ALLOW_PYTHONPATH_HACK="${ALLOW_PYTHONPATH_HACK:-0}"
PRECHECK_FAILS=()

log() {
    printf '[verify4] %s\n' "$*" >&2
}

warn() {
    printf '[verify4][WARN] %s\n' "$*" >&2
}

fail() {
    printf '[verify4][FAIL] %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || fail "missing required command: $cmd"
}

run_preflight_probes() {
    PRECHECK_FAILS=()

    if ! "$CLI" --help >/dev/null 2>&1; then
        PRECHECK_FAILS+=("\"$CLI\" --help")
    fi
    if ! "$CLI" analyze -h >/dev/null 2>&1; then
        PRECHECK_FAILS+=("\"$CLI\" analyze -h")
    fi
    if ! "$CLI" analyze --help >/dev/null 2>&1; then
        PRECHECK_FAILS+=("\"$CLI\" analyze --help")
    fi
    if ! "$PY" -c 'import egologqa' >/dev/null 2>&1; then
        PRECHECK_FAILS+=("\"$PY\" -c 'import egologqa'")
    fi
    if ! "$PY" -c 'import egologqa.cli' >/dev/null 2>&1; then
        PRECHECK_FAILS+=("\"$PY\" -c 'import egologqa.cli'")
    fi

    [ "${#PRECHECK_FAILS[@]}" -eq 0 ]
}

print_preflight_failures() {
    local probe
    for probe in "${PRECHECK_FAILS[@]}"; do
        warn "preflight probe failed: $probe"
    done
}

restore_pythonpath() {
    local original="$1"
    if [ -n "$original" ]; then
        export PYTHONPATH="$original"
    else
        unset PYTHONPATH
    fi
}

enforce_cli_python_preflight() {
    local orig_pythonpath="${PYTHONPATH-}"

    if run_preflight_probes; then
        return 0
    fi

    print_preflight_failures
    if [ "$ALLOW_PYTHONPATH_HACK" != "1" ]; then
        fail "egologqa CLI/import preflight failed. Fix install (for example: pip install -e .) instead of relying on PYTHONPATH hacks."
    fi

    [ -d "$REPO/src" ] || fail "ALLOW_PYTHONPATH_HACK=1 but $REPO/src not found"

    export PYTHONPATH="$REPO/src${orig_pythonpath:+:$orig_pythonpath}"
    warn "ALLOW_PYTHONPATH_HACK=1 enabled for this run only; retrying preflight with PYTHONPATH=$REPO/src"

    if run_preflight_probes; then
        warn "Hack mode enabled for this run only; fix packaging."
        return 0
    fi

    print_preflight_failures
    restore_pythonpath "$orig_pythonpath"
    fail "egologqa CLI/import preflight still failing after hack retry."
}

is_gt() {
    local a="$1"
    local b="$2"
    awk -v a="$a" -v b="$b" 'BEGIN { exit !(a > b) }'
}

is_eq() {
    local a="$1"
    local b="$2"
    awk -v a="$a" -v b="$b" 'BEGIN { exit !(a == b) }'
}

assert_rel_posix_path() {
    local p="$1"
    local label="$2"
    [ -n "$p" ] && [ "$p" != "NONE" ] && [ "$p" != "null" ] || fail "$label missing"
    case "$p" in
        /*) fail "$label absolute unix path: $p" ;;
    esac
    echo "$p" | grep -Eq '^[A-Za-z]:[\\/]' && fail "$label absolute windows path: $p"
    echo "$p" | grep -q '\\' && fail "$label has backslash: $p"
    case "$p" in
        ../*|*/../*|*/..|..) fail "$label traversal segment: $p" ;;
    esac
}

resolve_sync_thresholds() {
    local report="$1"
    local warn_ms fail_ms path
    local candidates=(
        "config_used.thresholds"
        "config_used.gate.thresholds"
        "config_used.gates.thresholds"
        "config_used.sync.thresholds"
    )
    for path in "${candidates[@]}"; do
        warn_ms="$(jq -r ".${path}.sync_warn_ms // empty" "$report" 2>/dev/null || true)"
        fail_ms="$(jq -r ".${path}.sync_fail_ms // empty" "$report" 2>/dev/null || true)"
        if [ -n "$warn_ms" ] && [ -n "$fail_ms" ] && [ "$warn_ms" != "null" ] && [ "$fail_ms" != "null" ]; then
            log "sync thresholds resolved from .$path (warn=$warn_ms fail=$fail_ms)"
            printf '%s|%s|%s\n' "$path" "$warn_ms" "$fail_ms"
            return 0
        fi
    done

    log "sync thresholds not found in known config_used paths for report: $report"
    jq -c '.config_used | keys' "$report" || true
    jq -c '.config_used.thresholds | keys' "$report" || true
    fail "unable to resolve sync_warn_ms/sync_fail_ms from config_used"
}

validate_manifest_entry_paths() {
    local manifest_file="$1"
    local run_dir="$2"
    local label="$3"
    local p

    while IFS= read -r p; do
        [ -n "$p" ] || continue
        assert_rel_posix_path "$p" "$label manifest path"
        [ -f "$run_dir/$p" ] || fail "$label manifest path does not resolve: $p"
    done < <(
        jq -r '
          [
            (.evidence_sets.blur_fail[]?.source_image_relpath // empty),
            (.evidence_sets.blur_pass[]?.source_image_relpath // empty)
          ] | .[]
        ' "$manifest_file"
    )
}

validate_manifest_annotated_paths() {
    local manifest_file="$1"
    local run_dir="$2"
    local label="$3"
    local p

    while IFS= read -r p; do
        [ -n "$p" ] || continue
        assert_rel_posix_path "$p" "$label manifest annotated path"
        [ -f "$run_dir/$p" ] || fail "$label manifest annotated path does not resolve: $p"
    done < <(
        jq -r '
          [
            (.evidence_sets.blur_fail[]?.annotated_image_relpath),
            (.evidence_sets.blur_pass[]?.annotated_image_relpath)
          ]
          | flatten
          | map(select(. != null))
          | .[]
        ' "$manifest_file"
    )
}

check_sync_gate_alignment() {
    local report="$1"
    local label="$2"
    local resolved path warn_ms fail_ms rest
    local sync_p95 gate fail_count warn_has fail_has fail_reasons

    resolved="$(resolve_sync_thresholds "$report")"
    path="${resolved%%|*}"
    rest="${resolved#*|}"
    warn_ms="${rest%%|*}"
    fail_ms="${rest##*|}"

    sync_p95="$(jq -r '.metrics.sync_p95_ms // "null"' "$report")"
    if [ "$sync_p95" = "null" ]; then
        log "SKIP: sync gate alignment for $label (sync_p95_ms missing)"
        return 0
    fi

    gate="$(jq -r '.gate.gate // .gate.status // "UNKNOWN"' "$report")"
    fail_count="$(jq -r '(.gate.fail_reasons // []) | length' "$report")"
    fail_reasons="$(jq -c '.gate.fail_reasons // []' "$report")"
    jq -e '(.gate.warn_reasons // []) | index("WARN_SYNC_P95_GT_WARN") != null' "$report" >/dev/null 2>&1 && warn_has=1 || warn_has=0
    jq -e '(.gate.fail_reasons // []) | index("FAIL_SYNC_P95_GT_FAIL") != null' "$report" >/dev/null 2>&1 && fail_has=1 || fail_has=0

    if is_gt "$sync_p95" "$fail_ms"; then
        [ "$gate" = "FAIL" ] || fail "sync alignment $label: sync_p95_ms=$sync_p95 exceeds fail=$fail_ms (from .$path) but gate=$gate"
        [ "$fail_count" -gt 0 ] || fail "sync alignment $label: sync_p95_ms=$sync_p95 exceeds fail=$fail_ms but fail_reasons is empty"
        if [ "$fail_has" -eq 1 ]; then
            log "PASS: sync alignment $label fail reason includes FAIL_SYNC_P95_GT_FAIL"
        else
            warn "sync alignment $label: fail threshold exceeded but FAIL_SYNC_P95_GT_FAIL missing; fail_reasons=$fail_reasons"
        fi
    elif is_gt "$sync_p95" "$warn_ms"; then
        [ "$warn_has" -eq 1 ] || fail "sync alignment $label: sync_p95_ms=$sync_p95 exceeds warn=$warn_ms (from .$path) but WARN_SYNC_P95_GT_WARN missing"
        case "$gate" in
            WARN|FAIL) ;;
            *) fail "sync alignment $label: warn threshold exceeded but gate=$gate (expected WARN or FAIL)" ;;
        esac
    else
        [ "$warn_has" -eq 0 ] || fail "sync alignment $label: sync_p95_ms=$sync_p95 <= warn=$warn_ms but WARN_SYNC_P95_GT_WARN present"
        [ "$fail_has" -eq 0 ] || fail "sync alignment $label: sync_p95_ms=$sync_p95 <= fail=$fail_ms but FAIL_SYNC_P95_GT_FAIL present"
    fi

    if is_eq "$sync_p95" "$warn_ms"; then
        log "NOTE: sync alignment $label boundary equality at warn threshold (sync_p95_ms == sync_warn_ms == $warn_ms)"
    fi
    if is_eq "$sync_p95" "$fail_ms"; then
        log "NOTE: sync alignment $label boundary equality at fail threshold (sync_p95_ms == sync_fail_ms == $fail_ms)"
    fi
}

make_cfg() {
    local out="$1"
    local manifest="$2"
    local annot="$3"
    "$PY" - <<PY
import yaml
with open("$BASECFG", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("debug", {})
cfg["debug"]["write_evidence_manifest"] = bool(int("$manifest"))
cfg["debug"]["write_annotated_evidence"] = bool(int("$annot"))
cfg["debug"]["benchmarks_enabled"] = False
with open("$out", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_mode() {
    local mode="$1"
    local f="$2"
    local cfg="$3"
    shift 3
    local b out ec rpt
    b="$(basename "$f" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
    out="$OUTROOT/${mode}_$b"
    rm -rf "$out"
    mkdir -p "$out"
    set +e
    "$CLI" analyze --input "$f" --output "$out" --config "$cfg" "$@" > "$out/analyze.log" 2>&1
    ec=$?
    set -e
    printf '%s\n' "$ec" > "$out/exit_code.txt"
    rpt="$(find "$out" -maxdepth 10 -name report.json | head -n 1)"
    [ -f "$rpt" ] || fail "missing report for mode=$mode file=$f"
}

run_direct_determinism() {
    local f="$1"
    local label="$2"
    local b out1 out2 ec1 ec2 rpt1 rpt2 man1 man2
    b="$(basename "$f" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
    out1="$OUTROOT/det_${label}_${b}_1"
    out2="$OUTROOT/det_${label}_${b}_2"

    rm -rf "$out1" "$out2"
    mkdir -p "$out1" "$out2"

    set +e
    "$CLI" analyze --input "$f" --output "$out1" --config "$CFG_MANIFEST" > "$out1/analyze.log" 2>&1
    ec1=$?
    "$CLI" analyze --input "$f" --output "$out2" --config "$CFG_MANIFEST" > "$out2/analyze.log" 2>&1
    ec2=$?
    set -e
    log "determinism run $label exits: first=$ec1 second=$ec2"
    [ "$ec1" = "$ec2" ] || fail "determinism $label exit codes differ: $ec1 vs $ec2"

    rpt1="$(find "$out1" -maxdepth 10 -name report.json | head -n 1)"
    rpt2="$(find "$out2" -maxdepth 10 -name report.json | head -n 1)"
    man1="$(find "$out1" -maxdepth 12 -name evidence_manifest.json | head -n 1)"
    man2="$(find "$out2" -maxdepth 12 -name evidence_manifest.json | head -n 1)"
    [ -f "$rpt1" ] || fail "determinism $label missing report 1"
    [ -f "$rpt2" ] || fail "determinism $label missing report 2"
    [ -f "$man1" ] || fail "determinism $label missing manifest 1"
    [ -f "$man2" ] || fail "determinism $label missing manifest 2"
    g1="$(jq -cS '.gate' "$rpt1")"
    g2="$(jq -cS '.gate' "$rpt2")"
    [ "$g1" = "$g2" ] || fail "determinism $label gate differs"

    cmp -s "$man1" "$man2" || fail "determinism $label manifest bytes differ"

    jq -S 'del(.input.analyzed_at_utc)' "$rpt1" > "$OUTROOT/det_${label}_r1.json"
    jq -S 'del(.input.analyzed_at_utc)' "$rpt2" > "$OUTROOT/det_${label}_r2.json"
    diff -u "$OUTROOT/det_${label}_r1.json" "$OUTROOT/det_${label}_r2.json" >/dev/null || fail "determinism $label report semantics differ"
}

require_cmd jq
require_cmd find
require_cmd diff
require_cmd grep
require_cmd tr
require_cmd basename

[ -x "$PY" ] || fail "missing python: $PY"
[ -x "$CLI" ] || fail "missing CLI: $CLI"
[ -f "$BASECFG" ] || fail "missing base config: $BASECFG"
enforce_cli_python_preflight
for f in "${MCAPS[@]}"; do
    [ -f "$f" ] || fail "missing mcap: $f"
done

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT/cfg"

CFG_DEFAULT="$OUTROOT/cfg/default.yaml"
CFG_MANIFEST="$OUTROOT/cfg/manifest.yaml"
CFG_ANNOT="$OUTROOT/cfg/annot.yaml"
CFG_MANIFEST_ANNOT="$OUTROOT/cfg/manifest_annot.yaml"
make_cfg "$CFG_DEFAULT" 0 0
make_cfg "$CFG_MANIFEST" 1 0
make_cfg "$CFG_ANNOT" 0 1
make_cfg "$CFG_MANIFEST_ANNOT" 1 1

log "running 4 files x 3 modes"
for f in "${MCAPS[@]}"; do
    run_mode default "$f" "$CFG_DEFAULT"
    run_mode manifest "$f" "$CFG_MANIFEST"
    run_mode bench "$f" "$CFG_DEFAULT" --bench
done

log "running single-file annotation mode"
run_mode annot "$M1" "$CFG_ANNOT"
log "running combined manifest+annot mode for M2"
run_mode manifest_annot "$M2" "$CFG_MANIFEST_ANNOT"

EXPECTED='config_used,errors,gate,input,metrics,sampling,segments,streams,time,tool'
: > "$OUTROOT/default_summary.jsonl"

for f in "${MCAPS[@]}"; do
    b="$(basename "$f" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
    DR="$OUTROOT/default_$b"
    MR="$OUTROOT/manifest_$b"
    BR="$OUTROOT/bench_$b"

    D_RPT="$(find "$DR" -maxdepth 10 -name report.json | head -n 1)"
    M_RPT="$(find "$MR" -maxdepth 10 -name report.json | head -n 1)"
    B_RPT="$(find "$BR" -maxdepth 10 -name report.json | head -n 1)"

    [ "$(jq -r 'keys|sort|join(",")' "$D_RPT")" = "$EXPECTED" ] || fail "top-level keys drift for $b"
    jq -e '.metrics.evidence_manifest_path? == null' "$D_RPT" >/dev/null || fail "default report unexpectedly contains evidence_manifest_path for $b"
    jq -e '.metrics.benchmarks_path? == null' "$D_RPT" >/dev/null || fail "default report unexpectedly contains benchmarks_path for $b"
    check_sync_gate_alignment "$D_RPT" "$b"

    test -z "$(find "$DR" -maxdepth 12 -name evidence_manifest.json | head -n 1)" || fail "manifest unexpectedly present in default for $b"
    MANIFEST_FILE="$(find "$MR" -maxdepth 12 -name evidence_manifest.json | head -n 1)"
    BENCH_FILE="$(find "$BR" -maxdepth 12 -name benchmarks.json | head -n 1)"
    [ -f "$MANIFEST_FILE" ] || fail "manifest missing for $b"
    [ -f "$BENCH_FILE" ] || fail "benchmarks missing for $b"
    test -z "$(find "$MR" -maxdepth 12 -name benchmarks.json | head -n 1)" || fail "benchmarks unexpectedly present in manifest mode for $b"
    test -z "$(find "$BR" -maxdepth 12 -name evidence_manifest.json | head -n 1)" || fail "manifest unexpectedly present in bench mode for $b"

    mp="$(jq -r '.metrics.evidence_manifest_path // "NONE"' "$M_RPT")"
    bp="$(jq -r '.metrics.benchmarks_path // "NONE"' "$B_RPT")"
    assert_rel_posix_path "$mp" "evidence_manifest_path"
    assert_rel_posix_path "$bp" "benchmarks_path"
    [ -f "$MR/$mp" ] || fail "metrics.evidence_manifest_path does not resolve for $b: $mp"
    [ -f "$BR/$bp" ] || fail "metrics.benchmarks_path does not resolve for $b: $bp"

    jq -e '.schema_version==1 and (.selection_context|type=="object") and (.evidence_sets|type=="object") and (.evidence_sets.blur_fail|type=="array") and (.evidence_sets.blur_pass|type=="array")' "$MANIFEST_FILE" >/dev/null \
        || fail "evidence_manifest schema invalid for $b"
    validate_manifest_entry_paths "$MANIFEST_FILE" "$MR" "$b"
    jq -e '
      (.schema_version != null) and
      (.phase_durations_s|type=="object") and
      (.phase_durations_s.scan? != null) and
      (.phase_durations_s.pass1? != null) and
      (.phase_durations_s.pass2? != null) and
      ((.phase_durations_s.total? != null) or (.total_s? != null))
    ' "$BENCH_FILE" >/dev/null \
        || fail "benchmarks schema invalid for $b"

    jq -S '{gate,segments,streams,sampling,time,errors,metrics:(.metrics|del(.benchmarks_path))}' "$D_RPT" > "$OUTROOT/d_$b.json"
    jq -S '{gate,segments,streams,sampling,time,errors,metrics:(.metrics|del(.benchmarks_path))}' "$B_RPT" > "$OUTROOT/e_$b.json"
    diff -u "$OUTROOT/d_$b.json" "$OUTROOT/e_$b.json" >/dev/null || fail "bench non-interference failed for $b"

    EC="$(cat "$DR/exit_code.txt")"
    jq -c --arg ec "$EC" '
    {
      exit_code: ($ec|tonumber),
      file: .input.file_path,
      gate_fail: (.gate.fail_reasons // []),
      gate_warn: (.gate.warn_reasons // []),
      depth_topic: (.streams.depth_topic // "NONE"),
      sync_p95_ms: (.metrics.sync_p95_ms // "NONE"),
      sync_drift_ms_per_min: (.metrics.sync_drift_ms_per_min // "NONE"),
      sync_jitter_p95_ms: (.metrics.sync_jitter_p95_ms // "NONE"),
      tb_n: (.metrics.rgb_timebase_diff_sample_count // "NONE"),
      tb_abs_p95: (.metrics.rgb_timebase_diff_abs_p95_ms // "NONE"),
      tb_signed_p50: (.metrics.rgb_timebase_diff_signed_p50_ms // "NONE")
    }' "$D_RPT" >> "$OUTROOT/default_summary.jsonl"
done

M1B="$(basename "$M1" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
A1="$OUTROOT/annot_$M1B"
test -z "$(find "$A1" -maxdepth 12 -name evidence_manifest.json | head -n 1)" || fail "manifest present in annot-only mode"
if "$PY" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("cv2") else 1)'; then
    raw_count="$(find "$A1" -type d \( -name 'blur_fail_frames' -o -name 'blur_pass_frames' \) -print0 \
      | xargs -0 -I{} find "{}" -type f -name '*.jpg' 2>/dev/null \
      | wc -l | tr -d ' ')"
    if [ "${raw_count:-0}" -gt 0 ]; then
        test -d "$A1/debug/blur_fail_frames_annotated" || fail "cv2 present and fail evidence exists but blur_fail_frames_annotated missing"
        test -d "$A1/debug/blur_pass_frames_annotated" || fail "cv2 present and pass evidence exists but blur_pass_frames_annotated missing"
    else
        log "WARN: no raw evidence frames in annot run; skipping annotated-dir requirement"
    fi
fi

M2B="$(basename "$M2" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
MA_DIR="$OUTROOT/manifest_annot_$M2B"
MA_MANIFEST="$(find "$MA_DIR" -maxdepth 12 -name evidence_manifest.json | head -n 1)"
[ -f "$MA_MANIFEST" ] || fail "combined manifest+annot run missing evidence_manifest.json for $M2B"
jq -e '.selection_context.cv2_available != null' "$MA_MANIFEST" >/dev/null || fail "combined manifest+annot missing selection_context.cv2_available for $M2B"
validate_manifest_entry_paths "$MA_MANIFEST" "$MA_DIR" "$M2B-manifest_annot"

ma_cv2_available="$(jq -r '.selection_context.cv2_available' "$MA_MANIFEST")"
ma_evidence_count="$(jq -r '((.evidence_sets.blur_fail // []) | length) + ((.evidence_sets.blur_pass // []) | length)' "$MA_MANIFEST")"
ma_annot_nonnull_count="$(jq -r '
  [
    (.evidence_sets.blur_fail[]?.annotated_image_relpath),
    (.evidence_sets.blur_pass[]?.annotated_image_relpath)
  ]
  | flatten
  | map(select(. != null))
  | length
' "$MA_MANIFEST")"
if [ "$ma_cv2_available" = "true" ] && [ "${ma_evidence_count:-0}" -gt 0 ]; then
    if [ "${ma_annot_nonnull_count:-0}" -gt 0 ]; then
        validate_manifest_annotated_paths "$MA_MANIFEST" "$MA_DIR" "$M2B-manifest_annot"
    else
        ma_ann_dir_any="$(find "$MA_DIR" -type d \( -name 'blur_fail_frames_annotated' -o -name 'blur_pass_frames_annotated' \) | head -n 1)"
        [ -n "$ma_ann_dir_any" ] || fail "combined manifest+annot has evidence but no annotated directories for $M2B"
        ma_ann_jpg_count="$(find "$MA_DIR" -type f \( -path '*/blur_fail_frames_annotated/*.jpg' -o -path '*/blur_pass_frames_annotated/*.jpg' \) | wc -l | tr -d ' ')"
        ma_src_jpg_count="$(find "$MA_DIR" -type f \( -path '*/blur_fail_frames/*.jpg' -o -path '*/blur_pass_frames/*.jpg' \) | wc -l | tr -d ' ')"
        [ "${ma_ann_jpg_count:-0}" -gt 0 ] || fail "combined manifest+annot fallback found no annotated jpg files for $M2B"
        [ "${ma_src_jpg_count:-0}" -gt 0 ] || fail "combined manifest+annot fallback found no source jpg files for $M2B"
        [ "${ma_ann_jpg_count:-0}" -le "${ma_src_jpg_count:-0}" ] || fail "combined manifest+annot fallback annotated count exceeds source count for $M2B"
    fi
else
    log "SKIP: combined manifest+annot strict annotation linkage for $M2B (cv2_available=$ma_cv2_available evidence_count=$ma_evidence_count)"
fi

jq -s -e 'map(select((.depth_topic=="NONE") and ((.sync_drift_ms_per_min!="NONE") or (.sync_jitter_p95_ms!="NONE")))) | length == 0' "$OUTROOT/default_summary.jsonl" >/dev/null \
    || fail "sync diagnostics present without depth"
jq -s -e 'map(select((.tb_n=="NONE") and ((.tb_abs_p95!="NONE") or (.tb_signed_p50!="NONE")))) | length == 0' "$OUTROOT/default_summary.jsonl" >/dev/null \
    || fail "timebase diagnostics present without samples"
jq -s -e 'map(select((.tb_n!="NONE") and ((.tb_abs_p95=="NONE") or (.tb_signed_p50=="NONE")))) | length == 0' "$OUTROOT/default_summary.jsonl" >/dev/null \
    || fail "timebase metrics missing despite samples"
jq -s -e 'map(select((.sync_drift_ms_per_min!="NONE") and (.sync_jitter_p95_ms=="NONE"))) | length == 0' "$OUTROOT/default_summary.jsonl" >/dev/null \
    || fail "drift present without jitter"

if jq -s -e 'length > 0 and all(.[]; (.tb_abs_p95 | type == "number" and . == 0) and (.tb_signed_p50 | type == "number" and . == 0))' "$OUTROOT/default_summary.jsonl" >/dev/null 2>&1; then
    log "NOTE: timebase diffs are zero on this 4-file real-data set; non-zero/invalid-header behavior is covered by unit tests."
fi

log "default summaries:"
cat "$OUTROOT/default_summary.jsonl"

HARNESS="$REPO/scripts/post_change_verify.sh"
[ -f "$HARNESS" ] || HARNESS="$REPO/post_change_verify.sh"
[ -f "$HARNESS" ] || fail "post_change_verify harness not found"

if [ "$RUN_HARNESS_DETERMINISM" = "1" ]; then
    log "running direct determinism checks for M1 and M3"
    run_direct_determinism "$M1" "M1"
    run_direct_determinism "$M3" "M3"
fi

if [ "$RUN_HARNESS_UI" = "1" ]; then
    log "running UI startup harness check for M1"
    MCAP="$M1" RUN_STREAMLIT=1 RUN_VALIDATION=0 sh "$HARNESS"
fi

if [ "$RUN_HARNESS_VALIDATION" = "1" ]; then
    log "running optional validation harness check for M1"
    MCAP="$M1" RUN_STREAMLIT=0 RUN_VALIDATION=1 sh "$HARNESS"
fi

log "PASS: verify4 checks completed"
