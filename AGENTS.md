# AGENTS.md

Operational knowledge base for agents and developers working on EgoLogQA.

Last updated: 2026-02-09  
Repository root: repo root (`.`)  
Snapshot source: `docs/EGOLOGQA_MASTER_DOCUMENTATION.md`

This document is intentionally practical. It is a concise, code-grounded reference for daily implementation and review work.

## 1) What this project is

EgoLogQA is a CLI-first quality-control analyzer for MicroAGI00-style ROS2 MCAP logs.

Primary outputs per run:
- Deterministic `report.json` (always written)
- Gate decision: `PASS` / `WARN` / `FAIL`
- Ordered fail/warn reason enums
- Integrity segments (`segments[]`)
- Optional diagnostics artifacts (`report.md`, plots, previews, CSVs, evidence frames)

Design intent in current code:
- Integrity-first segmentation
- Graceful degradation when RGB/depth pixel decode is unavailable
- Deterministic artifact selection and canonical report writing
- Additive diagnostics in `metrics` and `errors[]` without top-level schema expansion

## 2) Repository layout (current)

Top-level:
- `README.md`
- `AGENTS.md`
- `pyproject.toml`
- `uv.lock`
- `configs/microagi00_ros2.yaml`
- `app/streamlit_app.py`
- `docs/EGOLOGQA_MASTER_DOCUMENTATION.md` (local source-of-truth reference)
- `scripts/hygiene_check.sh`
- `scripts/post_change_verify.sh`
- `scripts/verify_4_mcap.sh`
- `src/egologqa/...`
- `tests/...`
- runtime artifact areas (`report_out/`, `out/`, `~/.cache/egologqa/runs`)

Source package highlights:
- `src/egologqa/pipeline.py`
- `src/egologqa/gate.py`
- `src/egologqa/constants.py`
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/metrics/pixel_metrics.py`
- `src/egologqa/artifacts.py`
- `src/egologqa/report.py`
- `src/egologqa/frame_flags.py`
- `src/egologqa/segments.py`
- `src/egologqa/io/reader.py`
- `src/egologqa/io/hf_fetch.py`
- `src/egologqa/io/local_fs.py`
- `src/egologqa/kiosk_helpers.py`
- `src/egologqa/ui_text.py`

Tests:
- `tests/integration/test_pipeline_inmemory.py`
- `tests/unit/*.py`

## 3) Runtime environments and command gotchas

Observed environment snapshots can vary by machine. Practical rule is stable:
- Use `.venv/bin/python` and `.venv/bin/egologqa` for deterministic behavior.
- Do not rely on `python`/`pip` aliases being present.

Canonical commands:
- Analyze: `.venv/bin/python -m egologqa analyze ...`
- Tests: `.venv/bin/python -m pytest -q`
- UI: `.venv/bin/streamlit run app/streamlit_app.py`

## 4) Dependencies and pinning

From `pyproject.toml`:
- Python `>=3.11,<3.12`
- `mcap-ros2-support==0.5.5`
- `huggingface_hub==0.27.1`
- `numpy==1.26.4`
- `opencv-python==4.10.0.84`
- `PyYAML==6.0.2`
- `streamlit==1.41.1`
- Dev: `pytest==8.3.4`

Lock file:
- `uv.lock` is present and should be kept aligned with dependency policy.

## 5) CLI contract

Entrypoints:
- Script: `egologqa` -> `egologqa.cli:main`
- Module: `python -m egologqa`

Command shape:
- `egologqa analyze --input <mcap> --output <output_dir> --config <yaml>`
- Optional topic overrides:
  - `--rgb-topic`
  - `--depth-topic`
  - `--imu-accel-topic`
  - `--imu-gyro-topic`

Output path contract:
- `--output` is a directory path.
- If file-like value is passed, it is still treated as directory.

Exit codes:
- `0` PASS
- `10` WARN
- `20` FAIL
- `30` ERROR path

Stdout contract:
- Human-readable summary lines (`GATE STATUS`, `RECOMMENDED ACTION`, reasons, error groups)
- Final machine line: `{"gate": "...", "recommended_action": "...", "report": "..."}`

## 6) UI contract (Streamlit)

File:
- `app/streamlit_app.py`

Current behavior:
- Kiosk-style two source tabs: Hugging Face + Local disk
- Analyze disabled until a real selection is made
- Synchronous progress updates (no worker queue)
- Hidden advanced controls only when `EGOLOGQA_UI_ADVANCED=1`

Results rendering:
- `Overall Result` (PASS/WARNING/FAIL display text)
- Recommended action block with:
  - token
  - plain-language "What to do now"
  - plain-language "Why"
- Structured reason tables with code + meaning + observed context
- Integrity and clean segment sections
- Artifact tables and images (plots, previews, evidence)
- Raw report expander

Run storage defaults:
- `~/.cache/egologqa/runs`
- `latest_run.txt` pointer maintained

## 7) Report schema contract (top-level keys fixed)

`report.json` top-level keys must remain exactly:
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

No additional top-level keys should be introduced without migration.

### 7.1 `errors[]` schema

Each error entry:
- `severity`: `"WARN"` or `"ERROR"`
- `code`: string
- `message`: string
- `context`: object

### 7.2 Canonicalization and stability

Implemented in `src/egologqa/report.py`:
- sorted JSON keys
- compact separators
- float rounding to 4 decimals
- NaN/Inf -> `null`

Guarantee level:
- semantic stability for equivalent inputs/config/runtime

## 8) Gate logic and enum ordering (authoritative)

Defined in `src/egologqa/constants.py` and applied in `src/egologqa/gate.py`.

FAIL reason order (exact):
1. `FAIL_ANALYSIS_ERROR`
2. `FAIL_NO_RGB_STREAM`
3. `FAIL_SYNC_P95_GT_FAIL`
4. `FAIL_DROP_RATIO_GT_FAIL`
5. `FAIL_DEPTH_FAIL_RATIO_GT_FAIL`
6. `FAIL_DEPTH_INVALID_MEAN_GT_FAIL`
7. `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`

WARN reason order (exact):
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

Default recommended action map:
- PASS -> `USE_FULL_SEQUENCE`
- WARN -> `USE_SEGMENTS_ONLY`
- FAIL -> `RECAPTURE_OR_SKIP`

Additional action token used by gate logic:
- `FIX_TIME_ALIGNMENT` (stable-offset sync path)

WARN floor behavior:
- `gate.gate_warn_floor_error_codes` (default: `["TIMESTAMP_OUT_OF_ORDER_HIGH"]`)
- Raises PASS to WARN when matching WARN diagnostics exist.

## 9) Timestamp and integrity semantics

Timestamp extraction (`src/egologqa/time.py`):
- prefer ROS header stamp when valid (`> 0`)
- fallback to `log_time_ns` if header invalid
- otherwise `0` and invalid source

Stream timestamp presence rule in pipeline:
- present only when stream has at least 2 valid timestamps

Out-of-order diagnostics:
- each collector tracks count/ratio and inversion indices
- threshold: `integrity.out_of_order_warn_ratio` (default `0.001`)
- emits `TIMESTAMP_OUT_OF_ORDER_HIGH` in `errors[]`

Sorting scope:
- allowed for alignment operations
- drop interval detection and segment extraction preserve original sequence semantics

Sync availability diagnostics:
- if insufficient sync samples (`sync_sample_count < sync_min_samples`) emit `SYNC_INSUFFICIENT_SAMPLES`
- if depth timestamps unavailable for sync emit `SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING`

## 10) Segment semantics

Integrity segments (`report["segments"]`):
- basis: `frame_ok_integrity`
- `frame_ok_integrity = sync_ok_fail AND rgb_drop_ok AND imu_ok`

Clean segments (stored as artifacts):
- strict basis: `sync_ok_warn AND rgb_drop_ok AND imu_ok AND blur_ok AND exposure_ok AND depth_ok`
- no-sync counterfactual basis: strict basis with sync forced non-blocking
- paths in metrics:
  - `clean_segments_path`
  - `clean_segments_nosync_path`
  - `clean_segments_basis = "warn_strict_quality_mask"`

Segment extraction (`src/egologqa/segments.py`):
- starts on first `ok`
- continues while `ok` gaps are `<= max_gap_fill_ms`
- breaks on forced positions and monotonic inversions
- drops durations `< min_segment_seconds`

## 11) Exposure + blur semantics

Exposure classifier (`src/egologqa/metrics/pixel_metrics.py`):
- ROI on grayscale for luminance statistics (`p01/p05/p50/p95/p99`)
- `low_clip` from grayscale ROI
- `high_clip` is channel-aware effective value:
  - `high_clip_luma = mean(gray >= high_clip_pixel_value)`
  - `high_clip_any_channel = mean(max(B,G,R) >= high_clip_pixel_value)`
  - `high_clip = max(high_clip_luma, high_clip_any_channel)`
- exposure bad conditions remain brightness-context dependent

Reason keys:
- `low_clip`, `high_clip`, `flat_and_dark`, `flat_and_bright`

Exposure CSV/debug fields include:
- `high_clip_luma`
- `high_clip_any_channel`

Blur semantics (locked for comparability):
- grayscale Laplacian variance on ROI
- threshold compare only: `blur_ok = blur_value >= blur_threshold_min`
- percentile blur metrics are diagnostic, not gating thresholds

## 12) Depth decoding semantics

Depth decoder (`src/egologqa/decoders/depth.py`):
- PNG signature search in payload
- decode with OpenCV unchanged mode
- require 2D `uint16`

Decode error codes:
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`
- `DEPTH_UNEXPECTED_SHAPE`

Gate depth catastrophic checks are only evaluated when depth is eligible:
- depth topic present
- decode success count > 0
- depth valid frame count > 0

## 13) Reader compatibility

Reader abstraction:
- protocol in `src/egologqa/io/reader.py`
- production: `MCapMessageSource`
- tests: `InMemoryMessageSource`

Compatibility handling:
- tuple-style and object-style mcap ROS2 reader outputs are both accepted
- compatibility shim remains in `src/egologqa/reader.py`

## 14) Pipeline flow (authoritative)

Ordered runtime flow in `src/egologqa/pipeline.py`:
1. load/apply config and topic overrides
2. scan/select streams
3. pass1 timestamp collection and time/integrity metrics
4. sampling plan
5. pass2 decode sampled RGB/depth and compute pixel metrics
6. frame flags
7. integrity segments
8. strict and no-sync clean masks + clean segment artifacts
9. sync pattern + offset estimate metrics
10. gate evaluation
11. evidence decision and artifact writing
12. preview writing (excluding evidence positions when possible)
13. plot writing
14. report writing (`report.md` best effort, `report.json` always)

Progress phases:
- `scan`, `pass1`, `pass2`, `done`, `error`

Evidence decision highlights:
- blur evidence can auto-trigger on blur WARN
- exposure evidence auto-triggers when exposure reason counts > 0 and RGB decode successes > 0

## 15) Configuration schema (current)

Config file:
- `configs/microagi00_ros2.yaml`

Main sections:
- `topics`, `expected_rates`, `sampling`, `thresholds`, `segments`, `gate`, `integrity`, `decode`, `debug`

Notable defaults:
- sampling: `rgb_stride=5`, `max_rgb_frames=12000`
- segments: `max_gap_fill_ms=200`, `min_segment_seconds=5`
- sync minimum samples: `sync_min_samples=30`
- sync warn/diagnostic thresholds:
  - `sync_warn_ms=16`, `sync_fail_ms=33`
  - `sync_jitter_warn_ms=5`
  - `sync_drift_warn_ms_per_min=10`
  - stable-offset thresholds for std/jitter/drift
- depth fail thresholds:
  - `depth_fail_ratio_fail=0.5`
  - `depth_invalid_mean_fail=0.6`
- pass exposure evidence floor:
  - `pass_exposure_evidence_k=2`

Validation in `src/egologqa/config.py` enforces ranges and required positive values.

## 16) Key metrics map (selected)

Time/integrity:
- `expected_rgb_dt_ms`, `drop_ratio`
- `sync_p50_ms`, `sync_p95_ms`, `sync_max_ms`, `sync_fail_ratio`
- `sync_sample_count`, `sync_pattern`, `sync_offset_estimate_ms`
- `imu_accel_missing_ratio`, `imu_gyro_missing_ratio`, `imu_combined_missing_ratio`
- `out_of_order`

RGB pixel metrics:
- blur: `blur_*`, `blur_valid_frame_count`
- exposure: `exposure_bad_ratio`, `exposure_valid_frame_count`, `exposure_bad_reason_counts`, `high_clip_mean/p95`, `low_clip_mean/p95`, `p50_*`, `dynamic_range_*`, `contrast_*`

Depth metrics:
- `depth_invalid_mean`, `depth_invalid_p95`, `depth_fail_ratio`, `depth_valid_frame_count`

Decode accounting:
- `rgb_decode_attempt_count`, `rgb_decode_success_count`
- `depth_decode_attempt_count`, `depth_decode_success_count`

Coverage and segment basis:
- `integrity_ok_ratio`, `integrity_coverage_seconds_est`
- `vision_ok_ratio`, `vision_coverage_seconds_est`
- `segments_basis`
- `clean_segments_basis`, `clean_segments_path`, `clean_segments_nosync_path`

Preview and artifact pointers:
- `preview_count`
- `preview_relpaths`
- `*_path` and `*_frames_dir` artifact fields

## 17) Known diagnostic codes currently emitted

Configuration/runtime:
- `CONFIG_LOAD_ERROR`
- `ANALYSIS_EXCEPTION`
- `STREAM_TIMESTAMPS_MISSING`
- `TIMESTAMP_OUT_OF_ORDER_HIGH`
- `LEGACY_EXPOSURE_KEYS_IGNORED`

Sync availability:
- `SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING`
- `SYNC_INSUFFICIENT_SAMPLES`

Decode/compute:
- `RGB_DECODE_FAIL`
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`
- `DEPTH_UNEXPECTED_SHAPE`
- `DEPTH_DTYPE_NON_UINT16_SEEN`
- `EXPOSURE_COMPUTE_FAILED`
- `RGB_EXPOSURE_DEBUG_UNAVAILABLE`
- `BLUR_UNAVAILABLE_NO_DECODE`

## 18) Test coverage map (current)

Baseline command:
- `.venv/bin/python -m pytest -q`

Latest observed status:
- `98 passed`

Integration:
- `tests/integration/test_pipeline_inmemory.py`

Unit inventory:
- `test_artifact_paths_relative.py`
- `test_artifact_plots.py`
- `test_benchmarks_opt_in.py`
- `test_blur_denominator_pipeline.py`
- `test_blur_depth_debug_csv.py`
- `test_blur_evidence_selection.py`
- `test_blur_formula_lock.py`
- `test_blur_warn_logic.py`
- `test_clean_segments_counterfactual.py`
- `test_depth_dtype_warning.py`
- `test_drop_regions.py`
- `test_evidence_manifest_pipeline.py`
- `test_exposure_conditional_saturation.py`
- `test_exposure_debug_csv.py`
- `test_exposure_evidence_selection.py`
- `test_frame_flags_sync_alias.py`
- `test_gate.py`
- `test_gate_decision_precedence.py`
- `test_hf_fetch.py`
- `test_integrity_segments_exposure.py`
- `test_local_fs.py`
- `test_nearest_alignment.py`
- `test_out_of_order.py`
- `test_pass_exposure_evidence_pipeline.py`
- `test_preview_metrics.py`
- `test_reader_compat.py`
- `test_report.py`
- `test_sampling.py`
- `test_segments.py`
- `test_streamlit_kiosk_helpers.py`
- `test_sync_diagnostics.py`
- `test_sync_pattern_classification.py`
- `test_sync_unavailable_graceful.py`
- `test_sync_warn_vs_fail_integrity.py`
- `test_time_extraction.py`
- `test_timebase_diagnostics.py`
- `test_ui_text.py`

## 19) Repro runbook

Install:
```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Analyze:
```bash
.venv/bin/python -m egologqa analyze \
  --input /absolute/path/to/file.mcap \
  --config configs/microagi00_ros2.yaml \
  --output report_out
```

Run tests:
```bash
.venv/bin/python -m pytest -q
```

Run UI:
```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## 20) Practical troubleshooting

### 20.1 Reader tuple/object API mismatch

Symptom:
- `cannot unpack non-iterable McapROS2Message object`

Status:
- handled by compatibility extraction in `src/egologqa/io/reader.py`

### 20.2 Output path confusion

Symptom:
- report path appears nested like `report.json/report.json`

Cause:
- `--output` passed as filename instead of directory

Fix:
- pass directory path such as `report_out`

### 20.3 Missing shell aliases (`python`, `pip`)

Fix:
- use `.venv/bin/python -m ...` explicitly

### 20.4 Exposure interpretation confusion

Use:
- `metrics.exposure_bad_reason_counts`
- `debug/exposure_samples.csv`
- `p50_*`, `dynamic_range_*`, `high_clip_*`, `low_clip_*`

Guideline:
- bright-looking frame is `high_clip` only when clipped-highlight area exceeds threshold and brightness context condition holds

### 20.5 Depth metrics unavailable

Expected behavior:
- depth decode status may be unsupported
- depth metrics may be null
- optional depth decode warn enum only when enabled by config

## 21) Current output snapshot note

For concrete behavior examples, prefer fresh run artifacts under runtime output directories.
Avoid embedding long-lived numeric snapshots in this file.

## 22) Contract guardrails for future changes

Do not change without migration:
- top-level `report.json` keys
- FAIL/WARN enum names and ordering
- `errors[]` schema shape

If changing classifier/segment/gate behavior:
- update tests first
- preserve deterministic ordering and canonicalization
- verify non-critical diagnostics cannot erase integrity segments unexpectedly

If adding diagnostics:
- place under `metrics` or `errors[]`
- avoid top-level schema expansion

## 23) Open follow-up items

1. Evaluate retention/cleanup policy for `~/.cache/egologqa/runs`.
2. Keep `uv.lock` policy aligned with reproducibility goals.
3. If operator-only controls grow, consider dedicated admin UI surface.
4. Continue validating threshold portability across camera resolutions.

## 24) Quick file index (where to edit what)

Config schema/validation:
- `src/egologqa/models.py`
- `src/egologqa/config.py`
- `configs/microagi00_ros2.yaml`

Gate rules and reason ordering:
- `src/egologqa/constants.py`
- `src/egologqa/gate.py`

Reader and data source helpers:
- `src/egologqa/io/reader.py`
- `src/egologqa/io/hf_fetch.py`
- `src/egologqa/io/local_fs.py`
- `src/egologqa/reader.py`

Time and sync metrics:
- `src/egologqa/time.py`
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/drop_regions.py`

Pixel metrics:
- `src/egologqa/metrics/pixel_metrics.py`

Flags and segments:
- `src/egologqa/frame_flags.py`
- `src/egologqa/segments.py`

Pipeline orchestration:
- `src/egologqa/pipeline.py`

Reports and artifacts:
- `src/egologqa/report.py`
- `src/egologqa/artifacts.py`

CLI/UI text and rendering:
- `src/egologqa/cli.py`
- `src/egologqa/ui_text.py`
- `app/streamlit_app.py`

Tests:
- `tests/unit/*.py`
- `tests/integration/test_pipeline_inmemory.py`
