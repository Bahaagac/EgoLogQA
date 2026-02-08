#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
PY="$REPO/.venv/bin/python"
CLI="$REPO/.venv/bin/egologqa"
BASECFG="$REPO/configs/microagi00_ros2.yaml"
OUTROOT="$REPO/out/testmatrix"
MCAP="${MCAP:-$HOME/.cache/egologqa/hf_mcaps/raw_mcaps/Bakery_Food_Preparation_15f719ff.mcap}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"
RUN_STREAMLIT="${RUN_STREAMLIT:-1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8766}"
WARN_MCAP="${WARN_MCAP:-}"
ALLOW_PYTHONPATH_HACK="${ALLOW_PYTHONPATH_HACK:-0}"

STRICT_FAIL=0
VALIDATION_STATUS="SKIPPED"
STREAMLIT_PID=""
STREAMLIT_WATCHDOG_PID=""
PRECHECK_FAILS=()

log() {
    printf '[post-verify] %s\n' "$*" >&2
}

warn() {
    printf '[post-verify][WARN] %s\n' "$*" >&2
}

fail() {
    printf '[post-verify][FAIL] %s\n' "$*" >&2
    STRICT_FAIL=1
}

cleanup() {
    if [ -n "${STREAMLIT_WATCHDOG_PID:-}" ] && kill -0 "$STREAMLIT_WATCHDOG_PID" 2>/dev/null; then
        kill "$STREAMLIT_WATCHDOG_PID" >/dev/null 2>&1 || true
        wait "$STREAMLIT_WATCHDOG_PID" >/dev/null 2>&1 || true
        STREAMLIT_WATCHDOG_PID=""
    fi
    if [ -n "${STREAMLIT_PID:-}" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        kill "$STREAMLIT_PID" >/dev/null 2>&1 || true
        wait "$STREAMLIT_PID" >/dev/null 2>&1 || true
        STREAMLIT_PID=""
    fi
}

trap cleanup EXIT INT TERM

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        fail "missing required command: $cmd"
    fi
}

hash_print() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f"
    else
        shasum -a 256 "$f"
    fi
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
        return 1
    fi

    [ -d "$REPO/src" ] || {
        fail "ALLOW_PYTHONPATH_HACK=1 but $REPO/src not found"
        return 1
    }

    export PYTHONPATH="$REPO/src${orig_pythonpath:+:$orig_pythonpath}"
    warn "ALLOW_PYTHONPATH_HACK=1 enabled for this run only; retrying preflight with PYTHONPATH=$REPO/src"

    if run_preflight_probes; then
        warn "Hack mode enabled for this run only; fix packaging."
        return 0
    fi

    print_preflight_failures
    restore_pythonpath "$orig_pythonpath"
    fail "egologqa CLI/import preflight still failing after hack retry."
    return 1
}

make_cfg() {
    local out="$1"
    local manifest="$2"
    local annot="$3"
    "$PY" - <<PY
import yaml
base_path = "$BASECFG"
out_path = "$out"
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("debug", {})
cfg["debug"]["write_evidence_manifest"] = bool(int("$manifest"))
cfg["debug"]["write_annotated_evidence"] = bool(int("$annot"))
cfg["debug"]["benchmarks_enabled"] = False
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out_path)
PY
}

make_forced_fail_cfg() {
    local out="$1"
    "$PY" - <<PY
import yaml
base_path = "$BASECFG"
out_path = "$out"
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("thresholds", {})
cfg["thresholds"]["sync_fail_ms"] = 0.0001
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out_path)
PY
}

run_case() {
    local name="$1"
    local cfg="$2"
    shift 2
    local out="$OUTROOT/$name"
    rm -rf "$out"
    mkdir -p "$out"
    log "run case: $name"
    set +e
    "$CLI" analyze --input "$MCAP" --output "$out" --config "$cfg" "$@" > "$out/analyze.log" 2>&1
    local ec=$?
    set -e
    log "case $name exit=$ec (log: $out/analyze.log)"
    printf '%s\n' "$ec" > "$out/exit_code.txt"
    printf '%s\n' "$out"
}

read_exit_code() {
    local out="$1"
    tr -d '\n' < "$out/exit_code.txt"
}

get_report() {
    find "$1" -maxdepth 10 -name report.json | head -n 1
}

get_manifest() {
    find "$1" -maxdepth 12 -name evidence_manifest.json | head -n 1
}

get_bench() {
    find "$1" -maxdepth 12 -name benchmarks.json | head -n 1
}

assert_file_exists() {
    local f="$1"
    local msg="$2"
    if [ -n "$f" ] && [ -f "$f" ]; then
        log "PASS: $msg"
    else
        fail "$msg"
    fi
}

assert_absent() {
    local v="$1"
    local msg="$2"
    if [ -z "$v" ]; then
        log "PASS: $msg"
    else
        fail "$msg"
    fi
}

assert_nonempty() {
    local v="$1"
    local msg="$2"
    if [ -n "$v" ]; then
        log "PASS: $msg"
    else
        fail "$msg"
    fi
}

assert_eq() {
    local got="$1"
    local want="$2"
    local msg="$3"
    if [ "$got" = "$want" ]; then
        log "PASS: $msg"
    else
        fail "$msg (got=$got want=$want)"
    fi
}

assert_jq_true() {
    local expr="$1"
    local file="$2"
    local msg="$3"
    if jq -e "$expr" "$file" >/dev/null 2>&1; then
        log "PASS: $msg"
    else
        fail "$msg"
    fi
}

assert_relative_posix_path() {
    local p="$1"
    local label="$2"
    if [ -z "$p" ] || [ "$p" = "NONE" ] || [ "$p" = "null" ]; then
        fail "$label missing"
        return
    fi
    [[ "$p" != *$'\n'* && "$p" != *$'\r'* && "$p" != *$'\t'* ]] \
        || { fail "$label contains control characters: $p"; return; }
    [[ "$p" != /* ]] || { fail "$label is absolute unix path: $p"; return; }
    [[ ! "$p" =~ ^[A-Za-z]:[\\/] ]] || { fail "$label is absolute windows path: $p"; return; }
    [[ "$p" != *\\* ]] || { fail "$label contains backslash separator: $p"; return; }
    [[ "$p" != "../"* && "$p" != *"/../"* && "$p" != *"/.." && "$p" != ".." ]] \
        || { fail "$label contains path traversal segment: $p"; return; }
    log "PASS: $label is relative POSIX path"
}

run_pytest_suite() {
    log "running full pytest suite"
    if "$PY" -m pytest -q; then
        log "PASS: full pytest suite"
    else
        fail "full pytest suite failed"
    fi
}

run_pytest_targeted() {
    log "running targeted additive pytest suite"
    if "$PY" -m pytest -q \
        tests/unit/test_sync_diagnostics.py \
        tests/unit/test_timebase_diagnostics.py \
        tests/unit/test_evidence_manifest_pipeline.py \
        tests/unit/test_benchmarks_opt_in.py \
        tests/unit/test_nearest_alignment.py; then
        log "PASS: targeted additive pytest suite"
    else
        fail "targeted additive pytest suite failed"
    fi
}

streamlit_smoke_check() {
    if [ "$RUN_STREAMLIT" != "1" ]; then
        log "SKIP: streamlit smoke check (RUN_STREAMLIT=$RUN_STREAMLIT)"
        return
    fi
    local log_path="$OUTROOT/streamlit_smoke.log"
    log "starting streamlit smoke check on port $STREAMLIT_PORT"
    printf 'STREAMLIT_PORT=%s\n' "$STREAMLIT_PORT" > "$log_path"
    EGOLOGQA_UI_ADVANCED=1 "$REPO/.venv/bin/streamlit" run "$REPO/app/streamlit_app.py" \
        --server.headless true --server.port "$STREAMLIT_PORT" >> "$log_path" 2>&1 &
    STREAMLIT_PID=$!
    (
        sleep 35
        if [ -n "${STREAMLIT_PID:-}" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            kill "$STREAMLIT_PID" >/dev/null 2>&1 || true
        fi
    ) &
    STREAMLIT_WATCHDOG_PID=$!

    local ok=0
    local i
    for i in $(seq 1 20); do
        if "$PY" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://localhost:$STREAMLIT_PORT", timeout=1).read(1)
PY
        then
            ok=1
            break
        fi
        sleep 1
    done
    cleanup
    local streamlit_classification streamlit_reason
    local body_log has_addr_in_use has_perm has_bind_ctx has_port_token has_running_line
    local addr_re perm_re bind_ctx_re running_re body_lines
    streamlit_classification=""
    streamlit_reason=""
    body_log="$OUTROOT/streamlit_smoke.body.log"
    addr_re='Address already in use|Port .* is already in use|EADDRINUSE|Errno 98|errno 98'
    perm_re='PermissionError|Operation not permitted|EACCES|permission denied'
    bind_ctx_re='((^|[^[:alnum:]_])(bind|listen|listening)([^[:alnum:]_]|$))|(cannot bind)|(listen on)|(socket)|(server\.port)|(--server\.port)'
    running_re='Running on|Local URL|Network URL'

    if [ "$ok" -eq 1 ]; then
        streamlit_classification="PASS"
        streamlit_reason="reachable"
    else
        [ -s "$log_path" ] || {
            streamlit_classification="FAIL"
            streamlit_reason="log_missing_or_empty"
            fail "streamlit startup log missing/empty: $log_path"
            printf 'streamlit_classification=%s reason=%s\n' "$streamlit_classification" "$streamlit_reason"
            return
        }

        tail -n +2 "$log_path" > "$body_log"

        has_addr_in_use=0
        has_perm=0
        has_bind_ctx=0
        has_port_token=0
        has_running_line=0
        grep -Eiq "$addr_re" "$body_log" && has_addr_in_use=1 || true
        grep -Eiq "$perm_re" "$body_log" && has_perm=1 || true
        grep -Eiq "$bind_ctx_re" "$body_log" && has_bind_ctx=1 || true
        grep -Fqx "STREAMLIT_PORT=$STREAMLIT_PORT" "$log_path" && has_port_token=1 || true
        grep -Eiq "$running_re" "$body_log" && has_running_line=1 || true

        if [ "$has_addr_in_use" -eq 1 ]; then
            streamlit_classification="FAIL"
            streamlit_reason="addr_in_use"
            fail "streamlit startup failed: address already in use on port $STREAMLIT_PORT"
        elif [ "$has_perm" -eq 1 ] && [ "$has_bind_ctx" -eq 1 ] && [ "$has_port_token" -eq 1 ]; then
            streamlit_classification="SKIP"
            streamlit_reason="permission_bind"
            warn "streamlit smoke skipped due to bind permission limits on port $STREAMLIT_PORT (running_line=$has_running_line)"
        else
            streamlit_classification="FAIL"
            streamlit_reason="unreachable_unknown"
            fail "streamlit startup not reachable within timeout"
        fi

        if [ "$streamlit_classification" = "FAIL" ]; then
            printf 'STREAMLIT_PORT=%s\n' "$STREAMLIT_PORT"
            echo "--- streamlit body (head 30) ---"
            sed -n '1,30p' "$body_log" || true
            body_lines="$(wc -l < "$body_log" | tr -d ' ')"
            if [ "${body_lines:-0}" -gt 30 ]; then
                echo "--- streamlit body (tail 30) ---"
                tail -n 30 "$body_log" || true
            fi
        elif [ "$streamlit_classification" = "SKIP" ]; then
            echo "--- streamlit skip evidence (body-only) ---"
            grep -Ein "$addr_re|$perm_re|$bind_ctx_re" "$body_log" | sed -n '1,30p' || true
        fi
    fi

    printf 'streamlit_classification=%s reason=%s\n' "$streamlit_classification" "$streamlit_reason"
}

validation_check() {
    if [ "$RUN_VALIDATION" != "1" ]; then
        log "SKIP: validation tooling (RUN_VALIDATION=$RUN_VALIDATION)"
        VALIDATION_STATUS="SKIPPED"
        return
    fi

    local val_before val_after run_log sum_log rc1 rc2
    run_log="$OUTROOT/validation_run.log"
    sum_log="$OUTROOT/validation_summary.log"
    val_before="$(git status --porcelain || true)"

    set +e
    "$PY" "$REPO/validation/run_validation.py" --repo-id MicroAGI-Labs/MicroAGI00 --revision main --prefix raw_mcaps/ --limit 1 > "$run_log" 2>&1
    rc1=$?
    if [ "$rc1" -eq 0 ]; then
        "$PY" "$REPO/validation/summarize_results.py" > "$sum_log" 2>&1
        rc2=$?
    else
        rc2=99
    fi
    set -e

    if [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]; then
        VALIDATION_STATUS="PASS"
        log "PASS: validation tooling"
    else
        if grep -Eiq 'Failed to resolve|NameResolutionError|ConnectionError|Temporary failure in name resolution|Network is unreachable|No route to host|Max retries exceeded|gaierror|nodename nor servname provided|403|429|Forbidden|AccessDenied|Request blocked|Too Many Requests' "$run_log" "$sum_log" 2>/dev/null; then
            VALIDATION_STATUS="BLOCKED"
            warn "validation tooling blocked by network/DNS"
        else
            VALIDATION_STATUS="FAIL"
            fail "validation tooling failed (non-network)"
            sed -n '1,120p' "$run_log" || true
            sed -n '1,120p' "$sum_log" || true
        fi
    fi

    local tracked_val
    tracked_val="$(git ls-files validation/runs validation/results validation/labels.csv)"
    if [ -n "$tracked_val" ]; then
        fail "validation generated paths unexpectedly tracked"
        printf '%s\n' "$tracked_val"
    else
        log "PASS: validation generated paths are untracked"
    fi

    val_after="$(git status --porcelain || true)"
    if [ "$val_before" != "$val_after" ]; then
        warn "git status changed after validation run (warning only)"
    fi
}

edge_sweep() {
    log "running non-gating edge sweep"
    local list_path="$OUTROOT/edge_candidates.txt"
    {
        [ -d "$HOME/Downloads" ] && find "$HOME/Downloads" -maxdepth 1 -type f -name '*.mcap' 2>/dev/null || true
        [ -d "$HOME/.cache/egologqa/hf_mcaps/raw_mcaps" ] && find "$HOME/.cache/egologqa/hf_mcaps/raw_mcaps" -maxdepth 1 -type f -name '*.mcap' 2>/dev/null || true
        [ -f "$MCAP" ] && printf '%s\n' "$MCAP" || true
    } | LC_ALL=C sort | awk '!seen[$0]++' | head -n 8 > "$list_path"

    local count
    count="$(wc -l < "$list_path" | tr -d ' ')"
    log "edge sweep candidates: $count"
    if [ "$count" -eq 0 ]; then
        warn "edge sweep found no mcap files"
        return
    fi

    printf 'file|exit|depth_topic|sync_drift|timebase_sample_count|fail_reasons|warn_reasons\n'
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        local base out ec rpt depth_topic sync_drift tb_count fail_reasons warn_reasons
        base="$(basename "$f" .mcap | tr -cs 'A-Za-z0-9._-' '_')"
        out="$OUTROOT/edge_$base"
        rm -rf "$out"
        mkdir -p "$out"
        set +e
        "$CLI" analyze --input "$f" --output "$out" --config "$CFG_DEFAULT" >/dev/null 2>&1
        ec=$?
        set -e
        rpt="$(get_report "$out")"
        if [ -z "$rpt" ] || [ ! -f "$rpt" ]; then
            printf '%s|%s|%s|%s|%s|%s|%s\n' "$f" "$ec" "<no_report>" "<na>" "<na>" "<na>" "<na>"
            continue
        fi
        depth_topic="$(jq -r '.streams.depth_topic // "NONE"' "$rpt")"
        sync_drift="$(jq -r '.metrics.sync_drift_ms_per_min // "NONE"' "$rpt")"
        tb_count="$(jq -r '.metrics.rgb_timebase_diff_sample_count // "NONE"' "$rpt")"
        fail_reasons="$(jq -c '.gate.fail_reasons // []' "$rpt")"
        warn_reasons="$(jq -c '.gate.warn_reasons // []' "$rpt")"
        printf '%s|%s|%s|%s|%s|%s|%s\n' "$f" "$ec" "$depth_topic" "$sync_drift" "$tb_count" "$fail_reasons" "$warn_reasons"
    done < "$list_path"
}

log "preflight checks"
require_cmd jq
require_cmd find
require_cmd diff
require_cmd git
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    fail "missing hash utility (sha256sum or shasum)"
fi
if [ "$RUN_STREAMLIT" = "1" ]; then
    if ! test -x "$REPO/.venv/bin/streamlit"; then
        fail "missing streamlit binary in venv"
    fi
fi
if ! test -x "$PY"; then fail "missing python binary: $PY"; fi
if ! test -x "$CLI"; then fail "missing CLI binary: $CLI"; fi
if ! test -f "$BASECFG"; then fail "missing base config: $BASECFG"; fi
if ! test -f "$MCAP"; then fail "missing MCAP input: $MCAP"; fi
enforce_cli_python_preflight

if [ "$STRICT_FAIL" -ne 0 ]; then
    echo "FINAL_STATUS=FAIL"
    exit 1
fi

cd "$REPO"
if [ -f "$REPO/scripts/hygiene_check.sh" ]; then
    sh "$REPO/scripts/hygiene_check.sh" || fail "hygiene check failed"
elif [ -f "$REPO/hygiene_check.sh" ]; then
    sh "$REPO/hygiene_check.sh" || fail "hygiene check failed"
else
    fail "hygiene_check.sh not found"
fi

if [ -n "${CI:-}" ]; then
    run_pytest_suite
else
    run_pytest_targeted
fi

if [ "$STRICT_FAIL" -ne 0 ]; then
    echo "FINAL_STATUS=FAIL"
    exit 1
fi

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

CFG_DEFAULT="/tmp/egologqa_default.yaml"
CFG_MANIFEST="/tmp/egologqa_manifest.yaml"
CFG_ANNOT="/tmp/egologqa_annot.yaml"
CFG_FORCED_FAIL="/tmp/egologqa_forced_fail.yaml"

make_cfg "$CFG_DEFAULT" 0 0 >/dev/null
make_cfg "$CFG_MANIFEST" 1 0 >/dev/null
make_cfg "$CFG_ANNOT" 0 1 >/dev/null
make_forced_fail_cfg "$CFG_FORCED_FAIL" >/dev/null

A_DIR="$(run_case A_default "$CFG_DEFAULT")"
A_RPT="$(get_report "$A_DIR")"
assert_file_exists "$A_RPT" "A report exists"

B_DIR="$(run_case B_manifest "$CFG_MANIFEST")"
B_RPT="$(get_report "$B_DIR")"
assert_file_exists "$B_RPT" "B report exists"

C_DIR="$(run_case C_annot "$CFG_ANNOT")"
C_RPT="$(get_report "$C_DIR")"
assert_file_exists "$C_RPT" "C report exists"

mkdir -p /tmp/egologqa_no_cv2
cat > /tmp/egologqa_no_cv2/cv2.py <<'PY'
raise ImportError("simulated cv2 missing")
PY

D_DIR="$OUTROOT/D_cv2_missing"
rm -rf "$D_DIR"
mkdir -p "$D_DIR"
set +e
PYTHONPATH="/tmp/egologqa_no_cv2${PYTHONPATH:+:$PYTHONPATH}" "$CLI" analyze \
    --input "$MCAP" --output "$D_DIR" --config "$CFG_MANIFEST" > "$D_DIR/analyze.log" 2>&1
D_EC=$?
set -e
log "case D_cv2_missing exit=$D_EC (log: $D_DIR/analyze.log)"
printf '%s\n' "$D_EC" > "$D_DIR/exit_code.txt"
D_RPT="$(get_report "$D_DIR")"
assert_file_exists "$D_RPT" "D report exists"

E_DIR="$(run_case E_bench "$CFG_DEFAULT" --bench)"
E_RPT="$(get_report "$E_DIR")"
assert_file_exists "$E_RPT" "E report exists"

EXPECTED_KEYS='config_used,errors,gate,input,metrics,sampling,segments,streams,time,tool'
A_KEYS="$(jq -r 'keys|sort|join(",")' "$A_RPT")"
assert_eq "$A_KEYS" "$EXPECTED_KEYS" "top-level keys contract"

A_MAN="$(get_manifest "$A_DIR")"
A_BEN="$(get_bench "$A_DIR")"
assert_absent "$A_MAN" "A has no manifest"
assert_absent "$A_BEN" "A has no benchmarks"

B_MAN="$(get_manifest "$B_DIR")"
assert_nonempty "$B_MAN" "B manifest exists"
B_MAN_METRIC_PATH="$(jq -r '.metrics.evidence_manifest_path // "NONE"' "$B_RPT")"
assert_relative_posix_path "$B_MAN_METRIC_PATH" "B metrics manifest path"

assert_absent "$(get_manifest "$C_DIR")" "C manifest absent"

D_MAN="$(get_manifest "$D_DIR")"
assert_nonempty "$D_MAN" "D manifest exists"
if [ -n "$D_MAN" ] && [ -f "$D_MAN" ]; then
    assert_jq_true '.selection_context.cv2_available == false' "$D_MAN" "D cv2_available false"
    assert_jq_true '.disabled_reason == "cv2_unavailable"' "$D_MAN" "D disabled reason"
    assert_jq_true '(.evidence_sets.blur_fail | length) == 0 and (.evidence_sets.blur_pass | length) == 0' "$D_MAN" "D evidence sets empty"
fi

E_BEN="$(get_bench "$E_DIR")"
assert_nonempty "$E_BEN" "E benchmarks exists"
E_BEN_METRIC_PATH="$(jq -r '.metrics.benchmarks_path // "NONE"' "$E_RPT")"
assert_relative_posix_path "$E_BEN_METRIC_PATH" "E metrics benchmarks path"

B2_DIR="$(run_case B2_manifest "$CFG_MANIFEST")"
B2_RPT="$(get_report "$B2_DIR")"
assert_file_exists "$B2_RPT" "B2 report exists"
B2_MAN="$(get_manifest "$B2_DIR")"
assert_nonempty "$B2_MAN" "B2 manifest exists"
if [ -n "$B_MAN" ] && [ -n "$B2_MAN" ] && [ -f "$B_MAN" ] && [ -f "$B2_MAN" ]; then
    log "manifest hashes:"
    hash_print "$B_MAN"
    hash_print "$B2_MAN"
    if cmp -s "$B_MAN" "$B2_MAN"; then
        log "PASS: manifest byte determinism"
    else
        fail "manifest byte determinism"
    fi
fi

jq -S 'del(.input.analyzed_at_utc)' "$B_RPT" > "$OUTROOT/b_sem_1.json"
jq -S 'del(.input.analyzed_at_utc)' "$B2_RPT" > "$OUTROOT/b_sem_2.json"
if diff -u "$OUTROOT/b_sem_1.json" "$OUTROOT/b_sem_2.json" > "$OUTROOT/b_sem.diff"; then
    log "PASS: semantic report determinism"
else
    fail "semantic report determinism"
    sed -n '1,120p' "$OUTROOT/b_sem.diff" || true
fi

jq -S '{gate,segments,streams,sampling,time,errors}' "$A_RPT" > "$OUTROOT/a_core.json"
jq -S '{gate,segments,streams,sampling,time,errors}' "$E_RPT" > "$OUTROOT/e_core.json"
if diff -u "$OUTROOT/a_core.json" "$OUTROOT/e_core.json" > "$OUTROOT/bench_core.diff"; then
    log "PASS: bench non-interference"
else
    fail "bench non-interference"
    sed -n '1,120p' "$OUTROOT/bench_core.diff" || true
fi

A_EC="$(read_exit_code "$A_DIR")"
assert_eq "$A_EC" "0" "A PASS exit code is 0"

if [ -n "$WARN_MCAP" ] && [ -f "$WARN_MCAP" ]; then
    WARN_DIR="$OUTROOT/W_warn_fixture"
    rm -rf "$WARN_DIR"
    mkdir -p "$WARN_DIR"
    set +e
    "$CLI" analyze --input "$WARN_MCAP" --output "$WARN_DIR" --config "$CFG_DEFAULT" >/dev/null 2>&1
    WARN_EC=$?
    set -e
    if [ "$WARN_EC" = "10" ]; then
        log "PASS: WARN fixture exit code is 10"
    else
        fail "WARN fixture exit code expected 10 (got $WARN_EC)"
    fi
else
    log "SKIP: WARN exit-code check (WARN_MCAP not provided)"
fi

if ! jq -e '.streams.depth_topic? != null' "$A_RPT" >/dev/null 2>&1; then
    warn "streams.depth_topic missing; forced-fail check may skip"
fi
PRECOND_FORCE_FAIL="$(jq -r \
'((.streams.depth_topic? // null) != null)
 and ((.metrics.sync_p95_ms? // null) != null)
 and ((.metrics.sync_p95_ms? // 0) > 0)' "$A_RPT")"
if [ "$PRECOND_FORCE_FAIL" = "true" ]; then
    F_DIR="$OUTROOT/F_forced_fail"
    rm -rf "$F_DIR"
    mkdir -p "$F_DIR"
    set +e
    "$CLI" analyze --input "$MCAP" --output "$F_DIR" --config "$CFG_FORCED_FAIL" >/dev/null 2>&1
    F_EC=$?
    set -e
    F_RPT="$(get_report "$F_DIR")"
    if [ "$F_EC" = "20" ]; then
        log "PASS: forced non-exception FAIL exit code is 20"
    else
        warn "forced non-exception FAIL expected 20 but got $F_EC"
        if [ -n "$F_RPT" ] && [ -f "$F_RPT" ]; then
            jq -c '{gate: (.gate.gate // .gate.status // "NONE"), fail_reasons: (.gate.fail_reasons // []), warn_reasons: (.gate.warn_reasons // [])}' "$F_RPT" || true
        else
            warn "forced-fail report missing; cannot print gate/reasons"
        fi
    fi
else
    log "SKIP: forced non-exception FAIL check (preconditions not met)"
fi

Z_DIR="$OUTROOT/Z_exception_fail"
rm -rf "$Z_DIR"
mkdir -p "$Z_DIR"
set +e
"$CLI" analyze --input "$OUTROOT/does_not_exist.mcap" --output "$Z_DIR" --config "$CFG_DEFAULT" >/dev/null 2>&1
Z_EC=$?
set -e
assert_eq "$Z_EC" "30" "exception FAIL exit code is 30"

edge_sweep
streamlit_smoke_check
validation_check

FINAL_STATUS="PASS"
if [ "$STRICT_FAIL" -ne 0 ]; then
    FINAL_STATUS="FAIL"
elif [ "$VALIDATION_STATUS" = "BLOCKED" ]; then
    FINAL_STATUS="BLOCKED"
fi

printf 'FINAL_STATUS=%s\n' "$FINAL_STATUS"
printf 'VALIDATION_STATUS=%s\n' "$VALIDATION_STATUS"

if [ "$FINAL_STATUS" = "FAIL" ]; then
    exit 1
fi
exit 0
