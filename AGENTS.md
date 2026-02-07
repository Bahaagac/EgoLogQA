# AGENTS.md

Project knowledge base for agents and developers working on EgoLogQA.

Last updated: 2026-02-07  
Repository root: `/Users/bahaagac/Documents/New project 2`  
Current branch: `main`  
Observed HEAD commit: `29933c4b1c0abad3786afd8e62e94edee4932f87`

This document is intentionally detailed and operational. It is meant to prevent re-discovery of project context and repeated regressions.

## 1) What this project is

EgoLogQA is a CLI-first quality-control analyzer for MicroAGI00-style ROS2 MCAP logs.

Primary deliverables per run:
- Deterministic `report.json` (always written)
- PASS/WARN/FAIL gate decision with fixed reason enum ordering
- Integrity-based clean segments (`segments[]`)
- Optional human and debug artifacts (`report.md`, previews, plots, exposure CSV)

Design intent implemented in code:
- Graceful degradation when pixel decoders fail
- Semantic stability (not cross-machine byte identity)
- Segment extraction based on temporal integrity signals
- Pixel quality as advisory diagnostics and WARN contributions

## 2) Repository layout (current)

Top-level:
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `configs/microagi00_ros2.yaml`
- `app/streamlit_app.py`
- `src/egologqa/...`
- `tests/...`
- `report_out/...` (generated runtime artifacts, intentionally git-ignored)

Source package:
- `src/egologqa/__init__.py`
- `src/egologqa/__main__.py`
- `src/egologqa/cli.py`
- `src/egologqa/pipeline.py`
- `src/egologqa/report.py`
- `src/egologqa/constants.py`
- `src/egologqa/models.py`
- `src/egologqa/config.py`
- `src/egologqa/time.py`
- `src/egologqa/topic_select.py`
- `src/egologqa/sampling.py`
- `src/egologqa/drop_regions.py`
- `src/egologqa/frame_flags.py`
- `src/egologqa/segments.py`
- `src/egologqa/gate.py`
- `src/egologqa/artifacts.py`
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/metrics/pixel_metrics.py`
- `src/egologqa/decoders/rgb.py`
- `src/egologqa/decoders/depth.py`
- `src/egologqa/io/reader.py`
- `src/egologqa/reader.py` (backward-compatible re-export shim)

Tests:
- `tests/integration/test_pipeline_inmemory.py`
- `tests/unit/*.py`
- `tests/conftest.py`

## 3) Runtime environments and command gotchas

Observed on this machine:
- `python` command: not found
- `pip` command: not found
- `python3` exists but points to system Python:
  - executable: `/Library/Developer/CommandLineTools/usr/bin/python3`
  - PyYAML: `6.0.3` (system site-packages)
- project venv Python:
  - executable: `/Users/bahaagac/Documents/New project 2/.venv/bin/python`
  - PyYAML: `6.0.2` (matches pinned dependency)

Practical rule:
- Use `.venv/bin/python` (or activate venv) for deterministic behavior.
- Use `.venv/bin/python -m egologqa ...` or installed script `.venv/bin/egologqa`.

## 4) Dependencies and pinning

Defined in `pyproject.toml`:
- Python: `>=3.11,<3.12`
- `mcap-ros2-support==0.5.5`
- `numpy==1.26.4`
- `opencv-python==4.10.0.84`
- `PyYAML==6.0.2`
- `streamlit==1.41.1`
- Dev: `pytest==8.3.4`

Lock file:
- `uv.lock` exists (minimal content currently, but present).

## 5) CLI contract

Entrypoint:
- Script: `egologqa` -> `egologqa.cli:main`
- Module: `python -m egologqa`

Command:
- `egologqa analyze --input <mcap> --output <output_dir> --config <yaml>`
- Optional topic overrides:
  - `--rgb-topic`
  - `--depth-topic`
  - `--imu-accel-topic`
  - `--imu-gyro-topic`

Important:
- `--output` is treated as a directory, not a file path.
- Passing `--output report.json` creates directory `report.json/` and writes `report.json/report.json`.

Exit codes:
- `0` PASS
- `10` WARN
- `20` FAIL
- `30` ERROR (or FAIL with `ANALYSIS_EXCEPTION`)

CLI stdout behavior:
- Prints human-readable summary lines:
  - `GATE STATUS`
  - `RECOMMENDED ACTION`
  - `FAIL REASONS`
  - `WARN REASONS`
  - WARN and ERROR entries grouped by severity
- Prints one final machine-readable JSON line:
  - `{"gate": "...", "recommended_action": "...", "report": "..."}`

## 6) UI contract (Streamlit)

File:
- `app/streamlit_app.py`

Behavior:
- Synchronous analysis with progress callback updates
- No threaded worker/queue model
- Uses:
  - `st.progress`
  - `st.status`
  - placeholders for partial updates

Displayed sections:
- Gate status
- Stream summary
- Key metrics including integrity/vision coverage and exposure diagnostics
- Integrity segments table
- Channel summary table
- Sync histogram image (if produced)
- Drop timeline image (if produced)
- RGB preview gallery
- Errors table

UX wording used:
- "Integrity Segments" (not "core segments")

## 7) Report schema contract (top-level keys are fixed)

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

No new top-level keys should be introduced.

### 7.1 `errors[]` schema

Each error entry shape:
- `severity`: `"WARN"` or `"ERROR"`
- `code`: string
- `message`: string
- `context`: object

This is the unified place for non-enum diagnostics, warnings, and internal failures.

### 7.2 Canonicalization and stability

Implemented in `src/egologqa/report.py`:
- JSON sorted keys (`sort_keys=True`)
- fixed separators (compact canonical formatting)
- floats rounded to 4 decimals
- NaN/Inf converted to `null`

Supported guarantee:
- semantic stability, not universal cross-machine byte identity

## 8) Gate logic and enum ordering (fixed)

Defined in `src/egologqa/constants.py` and enforced in `src/egologqa/gate.py`.

FAIL reason order:
1. `FAIL_NO_RGB_STREAM`
2. `FAIL_ANALYSIS_ERROR`
3. `FAIL_SYNC_P95_GT_FAIL`
4. `FAIL_DROP_RATIO_GT_FAIL`
5. `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`

WARN reason order:
1. `WARN_DEPTH_TIMESTAMP_MISSING`
2. `WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED`
3. `WARN_RGB_PIXEL_DECODE_UNSUPPORTED`
4. `WARN_SYNC_P95_GT_WARN`
5. `WARN_DROP_RATIO_GT_WARN`
6. `WARN_IMU_MISSING_RATIO_GT_WARN`
7. `WARN_BLUR_FAIL_RATIO_GT_WARN`
8. `WARN_EXPOSURE_BAD_RATIO_GT_WARN`
9. `WARN_DEPTH_INVALID_MEAN_GT_WARN`

Recommended action mapping:
- PASS -> `USE_FULL_SEQUENCE`
- WARN -> `USE_SEGMENTS_ONLY`
- FAIL -> `RECAPTURE_OR_SKIP`

WARN floor behavior:
- Config key: `gate.gate_warn_floor_error_codes`
- Default: `["TIMESTAMP_OUT_OF_ORDER_HIGH"]`
- If gate would be PASS and any `errors[]` WARN has code in that set, gate is raised to WARN.

## 9) Timestamp and integrity semantics

Timestamp extraction (`src/egologqa/time.py`):
- Prefer ROS header stamp if valid
- Valid timestamp: integer and `> 0`
- Fallback to `log_time_ns` if header invalid/zero
- If both invalid => `0` and source `"invalid"`

Presence rule used in pipeline:
- stream considered timestamp-present only when it has at least 2 valid timestamps

Out-of-order tracking:
- Each stream collector tracks:
  - `out_of_order_count`
  - `out_of_order_ratio`
  - inversion indices
- Threshold: `integrity.out_of_order_warn_ratio` (default `0.001`)
- Above threshold emits `TIMESTAMP_OUT_OF_ORDER_HIGH` WARN in `errors[]`

Sorting scope rule in implementation:
- Sorting used for nearest-neighbor alignment tasks
- Drop interval detection and segmenting use original sequence order semantics

## 10) Segment definition and extraction

Current segment basis:
- Integrity segments (`segments_basis: "integrity"`)
- Built from `frame_ok_integrity`, not vision flags

Integrity frame flags:
- `sync_ok AND rgb_drop_ok AND imu_ok`

Vision frame flags:
- `frame_ok_integrity AND blur_ok AND exposure_ok AND depth_ok`

Drop intervals:
- Built from RGB timestamp gaps (`compute_stream_gaps`)
- Interval boundary convention: `(t_prev, t_curr]` (left-open, right-closed)
- Membership implemented in `DropRegions.contains`

Segment extraction (`src/egologqa/segments.py`):
- Start on first `ok`
- Continue while gaps between ok points <= `max_gap_fill_ms`
- Hard break on forced inversion boundaries
- Drop segments shorter than `min_segment_seconds`

Forced inversion handling:
- Inversion-adjacent sampled positions are marked forced bad
- Also used as forced break positions in segment extraction

## 11) Exposure classifier semantics (current, v1.3 behavior)

Implemented in `src/egologqa/metrics/pixel_metrics.py`.

Per-frame exposure features on ROI:
- `low_clip` = fraction of pixels `<= low_clip_pixel_value`
- `high_clip` = fraction of pixels `>= high_clip_pixel_value`
- `p01`, `p05`, `p50`, `p95`, `p99`
- `contrast = p99 - p01` (diagnostic only)
- `dynamic_range = p95 - p05`

ROI:
- margin ratio from `thresholds.exposure_roi_margin_ratio` (default `0.05`)
- fallback to full frame if crop invalid

Exposure bad conditions:
- `bad_flat_and_dark`: `dynamic_range < dynamic_range_min` and `p50 < median_dark`
- `bad_flat_and_bright`: `dynamic_range < dynamic_range_min` and `p50 > median_bright`
- `bad_saturation_dark`: `low_clip > low_clip_warn` and `p50 < median_dark`
- `bad_saturation_bright`: `high_clip > high_clip_warn` and `p50 > median_bright`
- `exposure_bad = OR(all above)`

Critical semantic point:
- `low_clip` alone does not trigger exposure failure.
- `high_clip` alone does not trigger exposure failure.
- Brightness context (`p50`) is required.

Reason counting keys (fixed in metrics):
- `low_clip`
- `high_clip`
- `flat_and_dark`
- `flat_and_bright`

Denominator for `exposure_bad_ratio`:
- only frames with successful RGB decode and successful exposure feature computation

Exposure compute exceptions:
- produce `errors[]` ERROR code `EXPOSURE_COMPUTE_FAILED`
- frame exposure flag defaults non-blocking (`True`) so tool failure does not erase integrity segments

Legacy config keys:
- `low_clip_threshold`, `high_clip_threshold`, `contrast_min`
- retained for backward compatibility but ignored by classifier
- if values differ from defaults, pipeline emits WARN `LEGACY_EXPOSURE_KEYS_IGNORED`

## 12) Depth decoding semantics

Depth decoder (`src/egologqa/decoders/depth.py`):
- Search for PNG signature bytes inside payload
- Decode from signature onward using OpenCV `IMREAD_UNCHANGED`
- Accept only `uint16`

Depth decode error codes:
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`

Policy:
- Unsupported depth pixels do not block integrity segmentation
- Optional WARN enum `WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED` controlled by config:
  - `decode.warn_on_depth_pixel_decode_failure` (default `false`)

## 13) Reader compatibility and MCAP API shape handling

Message source abstraction:
- `MessageSource` protocol in `src/egologqa/io/reader.py`
- Production source: `MCapMessageSource`
- Test source: `InMemoryMessageSource`

Compatibility handling:
- `_extract_record_fields` supports:
  - tuple shape `(schema, channel, message, ros_msg)` (older API)
  - object shape (`McapROS2Message`-like) with `schema/channel/ros_msg/log_time_ns/...` (newer API)
- This prevents failures like:
  - `TypeError: cannot unpack non-iterable McapROS2Message object`

Compatibility shim:
- `src/egologqa/reader.py` re-exports source classes from `io/reader.py`

## 14) Pipeline flow (authoritative)

Implemented in `src/egologqa/pipeline.py`.

High-level stages:
1. Load config + apply topic overrides
2. Initialize empty report and set `config_used`
3. Scan topics and select active streams
4. Pass 1: timestamps-only collection
5. Compute integrity/time metrics and sampling plan
6. Pass 2: decode sampled RGB/depth and compute pixel metrics
7. Build frame flags
8. Extract integrity segments
9. Evaluate gate
10. Write artifacts (`report.md`, `report.json`, optional plots/previews/csv)

Progress callback phases emitted:
- `scan`
- `pass1`
- `pass2`
- `done`
- `error`

Artifacts generated by pipeline:
- `report.json` always
- `report.md` best effort
- previews, plots, exposure CSV when data/decoder conditions allow

## 15) Configuration schema (current)

Config file:
- `configs/microagi00_ros2.yaml`

Major sections:
- `topics`
- `expected_rates`
- `sampling`
- `thresholds`
- `segments`
- `gate`
- `integrity`
- `decode`
- `debug`

Key defaults:
- topic mode: explicit MicroAGI00 topics
- sampling: `rgb_stride=5`, `max_rgb_frames=12000`
- segmenting: `max_gap_fill_ms=200`, `min_segment_seconds=5`
- gate warn floor codes: `["TIMESTAMP_OUT_OF_ORDER_HIGH"]`
- depth decode WARN enum disabled by default
- debug exposure csv export enabled by default

Validation in `src/egologqa/config.py` includes:
- topics mode in `{explicit, auto}`
- stride/max frames positive
- out_of_order_warn_ratio >= 0
- segment min duration > 0
- exposure ROI margin in `[0, 0.5)`
- pixel clip values in `[0, 255]`
- clip warn thresholds in `[0, 1]`
- median and dynamic-range thresholds in `[0, 255]`

## 16) Metrics implemented (selected map)

Time/integrity:
- `expected_rgb_dt_ms`
- `drop_ratio`
- `sync_p50_ms`, `sync_p95_ms`, `sync_max_ms`, `sync_fail_ratio`
- `imu_accel_missing_ratio`, `imu_gyro_missing_ratio`, `imu_combined_missing_ratio`
- `out_of_order` object per stream

Pixel (RGB):
- blur: `blur_median`, `blur_threshold`, `blur_fail_ratio`
- exposure:
  - `exposure_bad_ratio`
  - `low_clip_mean`, `low_clip_p95`
  - `high_clip_mean`, `high_clip_p95`
  - `contrast_mean`, `contrast_p05` (diagnostic)
  - `dynamic_range_mean`, `dynamic_range_p05`
  - `p50_mean`, `p50_p05`, `p50_p95`
  - `dark_frame_ratio`
  - `low_clip_when_dark_mean`
  - `exposure_bad_reason_counts`
  - `exposure_debug_csv_path`

Pixel (depth):
- `depth_invalid_mean`
- `depth_invalid_p95`
- `depth_fail_ratio`

Coverage:
- `integrity_ok_ratio`
- `integrity_coverage_seconds_est`
- `vision_ok_ratio`
- `vision_coverage_seconds_est`
- `segments_basis` (`"integrity"`)

## 17) Known error/warn codes currently emitted

From config/pipeline/decoders/CLI:
- `CONFIG_LOAD_ERROR`
- `ANALYSIS_EXCEPTION`
- `FAIL_ANALYSIS_ERROR` (gate enum reason)
- `STREAM_TIMESTAMPS_MISSING`
- `TIMESTAMP_OUT_OF_ORDER_HIGH`
- `LEGACY_EXPOSURE_KEYS_IGNORED`
- `RGB_DECODE_FAIL`
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`
- `RGB_EXPOSURE_DEBUG_UNAVAILABLE`
- `EXPOSURE_COMPUTE_FAILED`

Severity conventions:
- Data quality / availability diagnostics typically `WARN`
- Internal compute/path failures typically `ERROR`

## 18) Current test coverage map

Test command that currently passes:
- `.venv/bin/python -m pytest -q`
- Result observed: `25 passed in 1.61s`

Unit tests and what they lock:
- `test_reader_compat.py`
  - both tuple/object mcap message shapes
- `test_time_extraction.py`
  - header stamp priority, fallback, invalid behavior
- `test_sampling.py`
  - deterministic sampling behavior and target count
- `test_drop_regions.py`
  - boundary semantics `(left, right]`
- `test_nearest_alignment.py`
  - nearest-delta and nearest-index correctness
- `test_out_of_order.py`
  - inversion counting ratio
- `test_segments.py`
  - forced-break behavior in segment extraction
- `test_gate.py`
  - fixed reason ordering + warn floor behavior
- `test_report.py`
  - float rounding and NaN/Inf sanitization
- `test_exposure_conditional_saturation.py`
  - low_clip not enough alone, dark/bright/flat cases
- `test_exposure_debug_csv.py`
  - CSV export enabled/disabled/unavailable behavior
- `test_integrity_segments_exposure.py`
  - exposure cannot erase integrity segments
  - exposure compute failure recorded without segment wipeout

Integration:
- `test_pipeline_inmemory.py`
  - in-memory source smoke test
  - top-level report keys unchanged
  - expected metrics fields present
  - `errors[]` schema shape check

## 19) Repro runbook

Install:
```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Run analysis:
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

### 20.1 `cannot unpack non-iterable McapROS2Message object`

Cause:
- Reader expecting old tuple API only.

Current fix status:
- Resolved by `_extract_record_fields` in `src/egologqa/io/reader.py` that supports both APIs.

### 20.2 Output path confusion

Symptom:
- Final JSON output shows `report: report.json/report.json`.

Cause:
- `--output` expects directory; user passed file-like path.

Fix:
- pass `--output report_out` or another directory path.

### 20.3 Command not found (`python`, `pip`, `pytest`)

Cause:
- environment PATH on this machine.

Fix:
- use `.venv/bin/python -m ...` directly.

### 20.4 Exposure WARN noise diagnosis

Primary debug tools:
- `metrics.exposure_bad_reason_counts`
- `metrics.p50_*`, `dark_frame_ratio`, `low_clip_when_dark_mean`
- `debug/exposure_samples.csv` (if enabled)

Interpretation guideline:
- high `low_clip` with normal `p50` should not mark exposure_bad by current classifier.

### 20.5 Depth metrics unavailable

Expected behavior:
- `streams.decode_status.depth_pixels = "unsupported"`
- depth metrics fields `null`
- optional WARN enum only when `decode.warn_on_depth_pixel_decode_failure=true`
- UI should explicitly report unavailability with decoder code when available.

## 21) Current output example snapshot (from local `report_out`)

Latest observed run in local runtime artifacts:
- Gate: `WARN`
- Warn reasons: `["WARN_BLUR_FAIL_RATIO_GT_WARN"]`
- No `errors[]` entries
- Segments basis: `integrity`
- Non-empty segments present
- Exposure bad ratio low (`0.0167`)

This indicates exposure semantic fix is active and not over-triggering in the sampled local run.

## 22) Contract guardrails for future changes

Do not change without explicit migration:
- top-level `report.json` keys
- FAIL/WARN enum names/order
- `errors[]` object shape

If changing any classifier/segment logic:
- update or add tests first
- preserve deterministic ordering and rounding
- verify no regression where non-critical pixel metrics can erase integrity segments

If adding new diagnostics:
- place under `metrics` or `errors[]`
- avoid top-level schema changes

If modifying reader behavior:
- keep tuple/object compatibility tests green
- preserve no-ROS runtime requirement

## 23) Open technical debt / follow-up opportunities

1. Runtime artifacts are ignored by policy (`report_out/`, `out/`, `tmp_reports/`). If a static demo artifact is needed, copy it intentionally into a dedicated tracked folder (for example `examples/`).
2. `uv.lock` is very small; validate lock strategy if strict reproducibility across machines is required.
3. Streamlit currently writes to fixed output directory by default (`out/streamlit`); concurrent sessions may collide.
4. `sync_ok` is currently false when depth timestamps missing; long-duration recordings with no depth can fail via no-segments path even if RGB/IMU are healthy. Keep in mind when broadening dataset support.

## 24) Quick file index (where to edit what)

Config schema and validation:
- `src/egologqa/models.py`
- `src/egologqa/config.py`
- `configs/microagi00_ros2.yaml`

Gate rules/order:
- `src/egologqa/constants.py`
- `src/egologqa/gate.py`

Reader and mcap compatibility:
- `src/egologqa/io/reader.py`
- `src/egologqa/reader.py`

Exposure/blur metrics:
- `src/egologqa/metrics/pixel_metrics.py`

Time metrics / gaps / sync:
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/drop_regions.py`

Frame flags and segments:
- `src/egologqa/frame_flags.py`
- `src/egologqa/segments.py`

Pipeline orchestration:
- `src/egologqa/pipeline.py`

Report writing:
- `src/egologqa/report.py`
- `src/egologqa/artifacts.py`

CLI:
- `src/egologqa/cli.py`

UI:
- `app/streamlit_app.py`

Tests:
- `tests/unit/*.py`
- `tests/integration/test_pipeline_inmemory.py`
