# EgoLogQA

EgoLogQA is a deterministic quality-control analyzer for MicroAGI00-style ROS2 MCAP logs.

It answers one practical question for each recording:
- Is this recording safe to use as-is?
- Should we use only selected clean segments?
- Should we fix time alignment first?
- Should we recapture/skip?

EgoLogQA is CLI-first and includes a Streamlit operator UI.

## What This Project Does

For every run, EgoLogQA:
1. Reads the MCAP and inspects timestamp integrity.
2. Computes timing quality (sync, drop/gap, IMU coverage).
3. Samples RGB/depth frames and computes pixel diagnostics (blur, exposure, depth validity).
4. Builds integrity and clean segment masks.
5. Produces a deterministic gate and action recommendation.
6. Writes canonical machine output (`report.json`) plus optional diagnostics artifacts.

This project is designed so non-critical decoder issues degrade gracefully instead of crashing the whole analysis.

## What You Get From One Run

Technical gate token in `report.json`:
- `PASS`
- `WARN`
- `FAIL`

Human-facing wording in UI/docs:
- Pass
- Warning
- Fail

Recommended action token:
- `USE_FULL_SEQUENCE`
- `USE_SEGMENTS_ONLY`
- `FIX_TIME_ALIGNMENT`
- `RECAPTURE_OR_SKIP`

Primary outputs:
- `report.json` (always written)
- `segments[]` (integrity segments)
- Optional diagnostics: plots, previews, CSVs, evidence frames, manifest, benchmarks, markdown summary

## Why EgoLogQA Exists

MCAP quality issues are often mixed: timing may be good while decode is partial, or sync may be recoverably offset.

EgoLogQA separates:
- hard gating logic
- integrity-first segmentation
- additive diagnostics

This lets teams automate decisions without losing useful data.

## Quick Start

### 1) Install (project venv)

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

### 2) Analyze one file

```bash
.venv/bin/EgoLogQA analyze \
  --input /absolute/path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml
```

### 3) Run tests

```bash
.venv/bin/python -m pytest -q
```

### 4) Run the UI

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## How It Works (Short Technical Flow)

### Pass 1 (timing/integrity)
- Topic scan and stream selection.
- Timestamp extraction (`header.stamp` first, log-time fallback).
- Out-of-order diagnostics.
- RGB gap/drop estimation.
- RGB-depth sync metrics + drift/jitter diagnostics.
- Sampling plan construction.

### Pass 2 (sampled pixel checks)
- Decode sampled RGB/depth frames.
- Compute blur/exposure/depth metrics.
- Build per-sample flags.
- Extract segments.
- Evaluate gate and recommendation.

### Artifact/write phase
- Writes `report.json` canonically.
- Writes `report.md` best effort.
- Writes optional debug/evidence/manifest/bench artifacts based on config and data availability.

## CLI Contract

Entrypoints:
- `.venv/bin/EgoLogQA`
- `.venv/bin/egologqa`
- `.venv/bin/python -m egologqa`

Command:

```bash
EgoLogQA analyze \
  --input <mcap> \
  --output <output_dir> \
  [--config <yaml>] \
  [--rgb-topic <topic>] \
  [--depth-topic <topic>] \
  [--imu-accel-topic <topic>] \
  [--imu-gyro-topic <topic>] \
  [--bench]
```

Important rules:
- `--output` is always a directory path.
- `--bench` enables benchmarks export (`debug/benchmarks.json`) and `metrics.benchmarks_path` when successful.

Exit codes:
- `0` for PASS
- `10` for WARN
- `20` for FAIL (non-exception)
- `30` for configuration/exception failure paths

CLI stdout includes:
- gate status
- recommended action
- ordered fail/warn reasons
- grouped WARN/ERROR diagnostics
- artifact path summary
- final machine JSON line

## Streamlit UI (Operator Kiosk)

Main behavior:
- Two tabs: `Hugging Face` and `Local disk`.
- Local mode stages uploaded MCAP into run dir.
- HF mode resolves/downloads with cache reuse.
- Same-page progress and result rendering (synchronous run).
- Analyze buttons disabled until selection is valid.

Notable env vars:
- `EGOLOGQA_HF_REPO_ID` (default `MicroAGI-Labs/MicroAGI00`)
- `EGOLOGQA_HF_REVISION` (default `main`)
- `EGOLOGQA_HF_PREFIX` (default `raw_mcaps/`)
- `EGOLOGQA_HF_CACHE_DIR` (default `~/.cache/EgoLogQA/hf_mcaps`)
- `EGOLOGQA_RUNS_DIR` (default `~/.cache/EgoLogQA/runs`)
- `EGOLOGQA_UI_ADVANCED=1` enables advanced diagnostics panel
- `EGOLOGQA_AI_SUMMARY_ENABLED` (default `1`)
- `EGOLOGQA_GEMINI_MODEL` (optional model override)
- `EGOLOGQA_MAX_ANALYSIS_INPUT_GIB` (default `2.0`)

## Output Contract

Top-level `report.json` keys are fixed:
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

Canonicalization:
- sorted keys
- compact separators
- float rounding (4 decimals)
- NaN/Inf sanitized to `null`

## Determinism Scope

Guaranteed:
- semantic stability for equivalent inputs/config/runtime
- deterministic ordering for reason enums and evidence selection logic

Not guaranteed:
- universal byte identity across every machine/OS/library build

## Current Snapshot (Point-in-Time)

This repo moves. Treat this as an audited snapshot, not a permanent guarantee.

At commit `be1c8460a28129f54ea4fcc35fefa3fd5e204fe3`:
- `pytest -q`: `142 passed`

## Where To Read More

- Full commit-locked master spec (private/local): `docs/EGOLOGQA_MASTER_DOCUMENTATION.md`
- Operational contributor guide: `AGENTS.md`
- Validation tooling notes: `validation/README.md`
