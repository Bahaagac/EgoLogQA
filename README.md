# EgoLogQA

EgoLogQA is a deterministic quality-control analyzer for MicroAGI00-style ROS2 MCAP logs.
It is designed to answer one operational question quickly: can this capture be used as-is, used partially, or should it be recaptured?

For every run, EgoLogQA always writes a canonical `report.json` and returns a gate decision (`PASS`, `WARN`, or `FAIL`) with ordered reasons.
It also extracts integrity-based usable segments so good temporal data is not discarded just because RGB/depth pixel decoding is partially unavailable.
Optional diagnostics (plots, previews, CSVs, evidence frames, manifest, benchmarks) support manual QA triage without changing the top-level report schema.

The project is CLI-first and also includes a Streamlit kiosk UI for operators.

## What You Get From One Run

- Gate decision: `PASS`, `WARN`, or `FAIL`
- Recommended action token:
  - `USE_FULL_SEQUENCE`
  - `USE_SEGMENTS_ONLY`
  - `FIX_TIME_ALIGNMENT`
  - `RECAPTURE_OR_SKIP`
- Deterministic `report.json` (always written)
- Ordered fail/warn reason enums
- Integrity segments (`segments[]`)
- Optional diagnostics artifacts (`report.md`, plots, previews, CSV debug files, evidence images, manifest, benchmarks)

## Quick Start

Install in the project virtualenv:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Analyze one MCAP:

```bash
.venv/bin/EgoLogQA analyze \
  --input /absolute/path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run the UI:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## How It Works

EgoLogQA uses a two-pass pipeline:

1. Pass 1 (`scan`/`pass1`): topic scan, timestamp extraction, drop/gap and sync diagnostics, integrity-time metrics, sampling plan.
2. Pass 2 (`pass2`): sampled RGB/depth decode, blur/exposure/depth metrics, frame flags, segment extraction, gate decision.

Then it writes artifacts deterministically:

1. Integrity segments (`report["segments"]`)
2. Clean segment artifacts (`clean_segments.json`, `clean_segments_nosync.json`)
3. Optional evidence/previews/plots
4. Canonical `report.json` (always) and best-effort `report.md`

## CLI Contract

Entrypoints:

- `EgoLogQA` (script)
- `egologqa` (script alias)
- `.venv/bin/python -m egologqa` (module)

Command shape:

```bash
EgoLogQA analyze --input <mcap> --output <output_dir> [--config <yaml>] [--rgb-topic ...] [--depth-topic ...] [--imu-accel-topic ...] [--imu-gyro-topic ...] [--bench]
```

Rules:

- `--input` and `--output` are required.
- `--output` is always treated as a directory path.
- `--bench` enables `debug/benchmarks.json` and sets `metrics.benchmarks_path` when successful.

Exit codes:

- `0` -> PASS
- `10` -> WARN
- `20` -> FAIL (non-exception path)
- `30` -> ERROR path (config/exception)

## Understanding Outputs

### Fixed `report.json` top-level keys

The top-level keys are contract-locked:

- `tool`
- `input`
- `streams`
- `time`
- `sampling`
- `metrics`
- `gate`
- `segments`
- `config_used`
- `errors`

### Key artifact outputs

Typical output directory contents include:

- `report.json` (always)
- `report.md` (best effort)
- `previews/*.png` (when preview export is enabled and decode succeeds)
- `plots/sync_histogram.png` and `plots/drop_timeline.png` (when data is available)
- `debug/exposure_samples.csv`, `debug/blur_samples.csv`, `debug/depth_samples.csv` (when enabled/data available)
- `debug/clean_segments.json`
- `debug/clean_segments_nosync.json`
- `debug/evidence_manifest.json` (optional)
- `debug/benchmarks.json` (optional)

Path contract:

- Artifact paths stored in `report.json` metrics are output-directory-relative POSIX paths.

### Recommended action tokens

- `USE_FULL_SEQUENCE`: use the full recording.
- `USE_SEGMENTS_ONLY`: use only clean/integrity segments.
- `FIX_TIME_ALIGNMENT`: sync pattern appears stably offset and likely fixable.
- `RECAPTURE_OR_SKIP`: data quality is not sufficient for safe downstream use.

## Determinism and Stability Contract

EgoLogQA guarantees semantic stability for equivalent input/config/runtime conditions:

- `report.json` is canonicalized (sorted keys, fixed separators).
- Floats are rounded to 4 decimals.
- NaN/Inf values are sanitized to `null`.
- Deterministic ordering is used in gate reasons and artifact selection.

Byte-identical output across different machines/OS builds is not guaranteed.

## Streamlit UI (Kiosk)

The UI has two source tabs:

- `Hugging Face` dataset file selection
- `Local disk` single-file upload (staged into the run directory)

Behavior and toggles:

- Analyze is disabled until a valid file selection exists.
- Advanced controls are hidden unless `EGOLOGQA_UI_ADVANCED=1`.
- AI summary is enabled by default (`EGOLOGQA_AI_SUMMARY_ENABLED=1`) and can use `EGOLOGQA_GEMINI_MODEL`.
- If AI summary fails (missing key, SDK/network/model issues), UI falls back to deterministic summary lines.

## Current Known Baseline (Point-in-Time)

At commit `ed48f2256d08326ed223bc0b7822721c9b93932d` (audited on 2026-02-09):

- Test command: `.venv/bin/python -m pytest -q`
- Result: `1 failed, 138 passed`
- Failing test: `tests/unit/test_ai_summary.py::test_streamlit_quick_summary_renders_three_ai_lines_and_debug_caption`

Treat this as a snapshot, not a permanent target.

## Where Technical Detail Lives

- Contributor/agent operational reference: `/Users/bahaagac/Documents/EgoLogQA/AGENTS.md`
- Commit-locked master technical documentation: `/Users/bahaagac/Documents/EgoLogQA/docs/EGOLOGQA_MASTER_DOCUMENTATION.md`
