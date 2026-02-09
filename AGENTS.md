# AGENTS.md

Operational guide for contributors/agents working on EgoLogQA.

This file is intentionally concise. It covers execution-critical rules and pointers.
For full commit-locked technical detail, use `docs/EGOLOGQA_MASTER_DOCUMENTATION.md`.

Last updated: 2026-02-09  
Repository root: `.`  
Branch at audit: `main`  
HEAD at audit: `279b744c8dcce6eb93405a38bd317cea96e0b2a2`

## 1) Project Identity

EgoLogQA is a deterministic QC analyzer for MicroAGI00-style ROS2 MCAP logs.

Per run it produces:
- canonical `report.json` (always)
- gate token (`PASS` / `WARN` / `FAIL`)
- recommendation token (`USE_FULL_SEQUENCE`, `USE_SEGMENTS_ONLY`, `FIX_TIME_ALIGNMENT`, `RECAPTURE_OR_SKIP`)
- integrity segments (`segments[]`)
- optional diagnostics artifacts (plots, CSVs, evidence, manifest, benchmarks)

## 2) What Must Not Drift

### Top-level report keys (fixed)
- `tool`, `input`, `streams`, `time`, `sampling`, `metrics`, `gate`, `segments`, `config_used`, `errors`

### `errors[]` shape (fixed)
- `severity` (`WARN` or `ERROR`)
- `code` (string)
- `message` (string)
- `context` (object)

### Gate reason ordering (fixed)
From `src/egologqa/constants.py`:
- FAIL order:
  1. `FAIL_ANALYSIS_ERROR`
  2. `FAIL_NO_RGB_STREAM`
  3. `FAIL_SYNC_P95_GT_FAIL`
  4. `FAIL_DROP_RATIO_GT_FAIL`
  5. `FAIL_DEPTH_FAIL_RATIO_GT_FAIL`
  6. `FAIL_DEPTH_INVALID_MEAN_GT_FAIL`
  7. `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`
- WARN order:
  1. `WARN_DEPTH_TIMESTAMP_MISSING`
  2. `WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED`
  3. `WARN_RGB_PIXEL_DECODE_UNSUPPORTED`
  4. `WARN_SYNC_P95_GT_WARN`
  5. `WARN_SYNC_JITTER_P95_GT_WARN`
  6. `WARN_SYNC_DRIFT_ABS_GT_WARN`
  7. `WARN_DROP_RATIO_GT_WARN`
  8. `WARN_IMU_MISSING_RATIO_GT_WARN`
  9. `WARN_BLUR_FAIL_RATIO_GT_WARN`
  10. `WARN_EXPOSURE_BAD_RATIO_GT_WARN`
  11. `WARN_DEPTH_INVALID_MEAN_GT_WARN`

### Recommendation mapping
- PASS -> `USE_FULL_SEQUENCE`
- WARN -> `USE_SEGMENTS_ONLY`
- FAIL -> `RECAPTURE_OR_SKIP`
- Additional gate pathway may emit `FIX_TIME_ALIGNMENT`

## 3) Runtime Reality (This Machine)

Observed during audit:
- `python`: not found on PATH
- `pip`: not found on PATH
- `pytest`: not found on PATH
- `python3`: `/opt/homebrew/bin/python3` (`Python 3.14.3`)
- project interpreter: `.venv/bin/python` (`Python 3.11.14`)

Rule:
- Always use `.venv/bin/python` and `.venv/bin/EgoLogQA` for deterministic behavior.

## 4) Fast Commands

Install:
```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Analyze:
```bash
.venv/bin/EgoLogQA analyze \
  --input /absolute/path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml
```

Analyze with benchmarks:
```bash
.venv/bin/EgoLogQA analyze \
  --input /absolute/path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml \
  --bench
```

Tests:
```bash
.venv/bin/python -m pytest -q
```

UI:
```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## 5) Key Source-of-Truth Files

Core contracts:
- `src/egologqa/pipeline.py`
- `src/egologqa/gate.py`
- `src/egologqa/constants.py`
- `src/egologqa/report.py`
- `src/egologqa/models.py`
- `src/egologqa/config.py`
- `configs/microagi00_ros2.yaml`

Pixel/timing logic:
- `src/egologqa/metrics/pixel_metrics.py`
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/frame_flags.py`
- `src/egologqa/segments.py`
- `src/egologqa/drop_regions.py`

Interfaces/UI:
- `src/egologqa/cli.py`
- `app/streamlit_app.py`
- `src/egologqa/ui_text.py`
- `src/egologqa/ai_summary.py`

IO/decoding:
- `src/egologqa/io/reader.py`
- `src/egologqa/decoders/rgb.py`
- `src/egologqa/decoders/depth.py`
- `src/egologqa/io/hf_fetch.py`
- `src/egologqa/io/local_fs.py`

## 6) Operational Tooling

Portability hygiene:
- workflow: `.github/workflows/hygiene.yml`
- check script: `scripts/hygiene_check.sh`

Verification harnesses:
- `scripts/post_change_verify.sh`
  - runs preflight checks, targeted/full pytest, multi-case artifact checks, determinism checks, optional Streamlit smoke, optional validation tooling
  - emits classified status (`PASS`, `FAIL`, `BLOCKED`)
- `scripts/verify_4_mcap.sh`
  - 4-MCAP matrix across default/manifest/bench modes
  - checks path contracts, schema checks, determinism, bench non-interference
  - optionally calls `post_change_verify.sh`

Validation pack:
- `validation/run_validation.py`
- `validation/summarize_results.py`
- `validation/README.md`

## 7) Non-Negotiable Engineering Rules

1. Do not add new top-level keys to `report.json`.
2. Keep reason enums and their ordering stable unless explicitly migrating.
3. Keep `errors[]` schema stable.
4. Preserve deterministic write/canonicalization behavior.
5. Treat diagnostics as additive; do not silently erase core integrity behavior.
6. Keep backward compatibility for MCAP reader API shapes (tuple/object).

## 8) Private Master Doc Policy

`docs/EGOLOGQA_MASTER_DOCUMENTATION.md` is private and intentionally not public.

Required state:
- `.gitignore` includes `docs/EGOLOGQA_MASTER_DOCUMENTATION.md`
- file remains untracked (`git ls-files docs/EGOLOGQA_MASTER_DOCUMENTATION.md` should be empty)

## 9) Current Test Snapshot (Point-in-Time)

At audited HEAD:
- command: `.venv/bin/python -m pytest -q`
- result: `1 failed, 140 passed` (141 total)
- failing test:
  - `tests/unit/test_ai_summary.py::test_streamlit_quick_summary_renders_three_ai_lines_and_debug_caption`

Use this as a factual baseline, not a permanent expectation.

## 10) Recommended Change Workflow

1. Read affected contracts in source files first.
2. Implement minimal code changes.
3. Run targeted tests for touched subsystems.
4. Run full `pytest -q`.
5. Re-check schema/order/path contracts.
6. Update docs if behavior changed.

## 11) Escalation Guide for Documentation

If behavior changes in any of these areas, update both:
- `README.md` (clear first impression + user-facing behavior)
- `docs/EGOLOGQA_MASTER_DOCUMENTATION.md` (exhaustive commit-locked spec)

`AGENTS.md` should remain concise and operational, not exhaustive.
