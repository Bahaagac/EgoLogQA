# EgoLogQA Master Documentation and Technical Specification

Last audited: 2026-02-07 (local workspace state)  
Repository root: `/Users/bahaagac/Documents/New project 2`  
Audited branch: `main`  
Audited commit: `1bb95f30e6da377f5cfc12b9e69bb3b2c96bdbb3`

## 0) Scope and Accuracy Contract

This document is a code-grounded specification for the exact repository state at commit `1bb95f30e6da377f5cfc12b9e69bb3b2c96bdbb3`.

Accuracy meaning in this document:
- Behavior statements are derived from current source and tests in this repository.
- Values and thresholds are listed exactly as implemented.
- "Exact" applies to this commit only. Later commits can diverge.

This is not a conceptual overview only. It is a build-and-behavior spec intended to let a new developer reconstruct an equivalent system.

## 1) Current Project Status Snapshot

### 1.1 Git and Workspace Status

- Current branch: `main`
- HEAD: `1bb95f30e6da377f5cfc12b9e69bb3b2c96bdbb3`
- Working tree: clean (`git status --short --branch` -> `## main...origin/main`)
- Recent commits:
  - `1bb95f3` Add local file listing helper
  - `feb35a9` Clarify blur thresholds and warn
  - `3cacfa2` Ignore generated analysis artifacts and untrack report_out outputs
  - `29933c4` Fix exposure classifier conditional saturation semantics

### 1.2 Test Status

Command run:

```bash
.venv/bin/python -m pytest -q
```

Result:
- `68 passed in 0.80s`

### 1.3 Runtime/Toolchain Observed on This Machine

Path probes:
- `python`: not found on PATH
- `pip`: not found on PATH
- `python3`: `/opt/homebrew/bin/python3`
- project venv python: `.venv/bin/python`

Versions observed:
- `python3 -V` -> `Python 3.14.3`
- `.venv/bin/python -V` -> `Python 3.11.14`
- `PyYAML 6.0.2`
- `numpy 1.26.4`
- `opencv 4.10.0`
- `streamlit 1.41.1`
- `huggingface_hub 0.27.1`
- `mcap_ros2 0.5.5`

Operational rule:
- Use `.venv/bin/python ...` and `.venv/bin/egologqa ...` for deterministic execution.

### 1.4 Runtime Artifacts Present in Workspace

Generated directories present:
- `report_out/`
- `out/streamlit/`

Observed snapshots:
- `report_out/report.json`
  - gate: `WARN`
  - warn reasons: `["WARN_SYNC_P95_GT_WARN"]`
  - fail reasons: `[]`
  - errors: `0`
  - segments: `4`
  - metrics: `segments_basis="integrity"`, `blur_fail_ratio=0.0133`, `exposure_bad_ratio=0.0`
- `out/streamlit/report.json`
  - gate: `PASS`
  - warn reasons: `[]`
  - fail reasons: `[]`
  - errors: `0`
  - segments: `2`
  - metrics: `segments_basis="integrity"`, `blur_fail_ratio=0.1833`, `exposure_bad_ratio=0.0167`

Sizes observed:
- `report_out`: `40M`
- `out/streamlit`: `27M`

## 2) What EgoLogQA Is

EgoLogQA is a deterministic quality-control analyzer for MicroAGI00-style ROS2 MCAP logs.

Primary outputs per analysis run:
- `report.json` (always produced)
- PASS/WARN/FAIL gate decision
- clean segments list (`segments[]`) based on integrity signals
- optional human/debug artifacts (`report.md`, preview images, plots, CSVs, evidence frames)

Core design direction:
- CLI-first deterministic analysis pipeline
- Streamlit kiosk UI wrapper over same pipeline
- graceful degradation when pixel decode fails
- integrity segmentation cannot be erased by non-critical vision heuristics
- semantic stability over machine-level byte identity

## 3) Why This Project Exists (Purpose and Benefits)

Purpose:
- Automatically evaluate MCAP sequence usability for downstream use.
- Quantify synchronization, drops/gaps, IMU coverage, and sampled pixel quality.
- Produce machine-consumable and operator-consumable outputs in one run.

Benefits:
- Deterministic gate reasons and ordering.
- Standardized report schema and canonical JSON serialization.
- Segment extraction that prioritizes temporal/integrity validity.
- Clear WARN/ERROR diagnostics in structured `errors[]`.
- Supports both offline CLI workflows and kiosk-style UI workflows.

## 4) High-Level System Architecture

Main components:
- CLI entrypoint (`egologqa analyze`)
- Streamlit UI (`app/streamlit_app.py`)
- Analysis pipeline (`src/egologqa/pipeline.py`)
- IO adapters (MCAP source, HF listing/downloading, local file listing)
- Metrics engines (time/integrity + pixel)
- Gate evaluator
- Artifact writers
- Canonical report writer

Two-pass pipeline pattern:
1. Pass 1 collects timestamps and computes integrity/time metrics.
2. Pass 2 decodes sampled RGB/depth frames and computes pixel metrics.
3. Frame flags + segment extraction + gate evaluation + artifact persistence.

## 5) Repository Layout (Current, Relevant)

Top-level key files:
- `README.md`
- `AGENTS.md`
- `pyproject.toml`
- `uv.lock`
- `.gitignore`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml`
- `configs/microagi00_ros2.yaml`
- `app/streamlit_app.py`

Package source:
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
- `src/egologqa/kiosk_helpers.py`
- `src/egologqa/metrics/time_metrics.py`
- `src/egologqa/metrics/pixel_metrics.py`
- `src/egologqa/decoders/rgb.py`
- `src/egologqa/decoders/depth.py`
- `src/egologqa/io/reader.py`
- `src/egologqa/io/hf_fetch.py`
- `src/egologqa/io/local_fs.py`
- `src/egologqa/reader.py` (backward-compatible re-export shim)

Tests:
- `tests/integration/test_pipeline_inmemory.py`
- `tests/unit/*.py` (67 unit tests total)

## 6) Packaging and Dependency Contract

`pyproject.toml`:
- project name: `egologqa`
- version: `0.1.0`
- Python range: `>=3.11,<3.12`
- script entrypoint: `egologqa = egologqa.cli:main`
- dependencies:
  - `mcap-ros2-support==0.5.5`
  - `huggingface-hub==0.27.1`
  - `numpy==1.26.4`
  - `opencv-python==4.10.0.84`
  - `PyYAML==6.0.2`
  - `streamlit==1.41.1`
- dev dependency:
  - `pytest==8.3.4`

`uv.lock`:
- present and populated (`789` lines observed)
- requires python `==3.11.*`

## 7) Runbook (Canonical Commands)

Install:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Run CLI analysis:

```bash
.venv/bin/python -m egologqa analyze \
  --input /absolute/path/to/file.mcap \
  --config configs/microagi00_ros2.yaml \
  --output report_out
```

Equivalent console-script:

```bash
.venv/bin/egologqa analyze --input ... --config ... --output ...
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run UI:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## 8) CLI Specification (`src/egologqa/cli.py`)

### 8.1 Parser Contract

Program: `egologqa`

Subcommands:
- `analyze`

Arguments for `analyze`:
- `--input` (required): MCAP file path
- `--output` (required): output directory path
- `--config` (optional, default `None` -> fallback default config path)
- `--rgb-topic`
- `--depth-topic`
- `--imu-accel-topic`
- `--imu-gyro-topic`

### 8.2 Behavior

- If command is missing: prints help, exits `0`.
- Builds `TopicOverrides`.
- Loads config via `load_config`.

If config load fails:
- creates minimal report with `CONFIG_LOAD_ERROR`
- sets gate to FAIL with `FAIL_ANALYSIS_ERROR`
- writes `report.json` to output dir
- prints machine JSON line `{"gate":"FAIL","report":"..."}`
- exits with code `30`

If analysis succeeds:
- calls `analyze_file(...)`
- prints:
  - `GATE STATUS`
  - `RECOMMENDED ACTION`
  - `FAIL REASONS`
  - `WARN REASONS`
  - WARN entries list
  - ERROR entries list
  - artifact paths if present
  - final machine-readable JSON line:
    - `{"gate":"...","recommended_action":"...","report":"..."}`

### 8.3 Exit Code Mapping

- `0` -> PASS
- `10` -> WARN
- `20` -> FAIL
- `30` -> ERROR class
  - includes case where gate is FAIL and `ANALYSIS_EXCEPTION` exists in `errors[]`

## 9) Streamlit UI Specification (`app/streamlit_app.py`)

### 9.1 UI Mode

Single-page kiosk UI with tabs:
- `Hugging Face`
- `Local disk`

Global title/caption text:
- title: `EgoLogQA`
- caption: `MicroAGI00 ROS2 MCAP quality gate`
- instruction: `Choose an MCAP file to analyze.`

### 9.2 Environment Variables Consumed

HF defaults and overrides:
- `EGOLOGQA_HF_REPO_ID` (default `MicroAGI-Labs/MicroAGI00`)
- `EGOLOGQA_HF_REVISION` (default `main`)
- `EGOLOGQA_HF_PREFIX` (default `raw_mcaps/`)
- `EGOLOGQA_HF_CACHE_DIR` (default `~/.cache/egologqa/hf_mcaps`)

Run output directory:
- `EGOLOGQA_RUNS_DIR` (default from helper -> `~/.cache/egologqa/runs`)

Advanced/dev UI:
- `EGOLOGQA_UI_ADVANCED=1` enables advanced expander with manual HF refresh

Local list cap:
- `EGOLOGQA_LOCAL_MAX_FILES` (default `500`, coerced to positive int)

Token:
- `HF_TOKEN` from env, fallback to Streamlit secrets key `HF_TOKEN`

### 9.3 Streamlit Cache Usage

- `_cached_hf_file_list(...)`
  - `@st.cache_data(ttl=300)`
  - key includes repo/revision/prefix/token_digest
- `_cached_local_file_list(...)`
  - `@st.cache_data(ttl=300)`
  - key includes normalized dir + nonce + max files

### 9.4 Source Tabs Behavior

Hugging Face tab:
- placeholder-first selectbox (`None` option: `Select an MCAP file`)
- display label uses basename + size string
- analyze button disabled until non-placeholder selected
- if `huggingface_hub` missing, warning is shown

Local disk tab:
- folder preset selector (`~/Downloads`, `~/Desktop`, `~/.cache/egologqa`, optional `Last used`, `Other...`)
- custom folder path when `Other...`
- manual `Refresh local list` increments nonce
- placeholder-first file selectbox
- analyze button disabled until selection exists
- validates file readability before analysis

### 9.5 Run Directory Handling

For each analysis:
- ensures runs base dir writable
- allocates unique run dir via timestamped basename + collision suffixes
- updates `latest_run.txt` pointer in runs base dir
- runs analysis into that per-run directory
- stores latest report and output_dir in session state
- renders full report panel in same page

### 9.6 Progress and Status UX

- Uses `st.progress` and `st.status`
- maps pipeline phases:
  - `scan`, `pass1`, `pass2`, `done`, `error`
- HF download occupies early fraction (`0.0` to `0.2`)
- analysis progress scales from `analysis_start` to `1.0`

### 9.7 Results Rendering

Sections shown:
- gate summary
- core metrics table
- exposure/blur/depth metrics table
- exposure reason counts JSON
- segments table
- errors table
- artifact table (absolute resolved display paths)
- images (sync histogram/drop timeline)
- preview frames (up to 8 shown)
- blur fail/pass evidence galleries (up to 8)
- raw `report.json` expander

## 10) Configuration Schema (Dataclasses, Defaults, Validation)

Source: `src/egologqa/models.py` + `src/egologqa/config.py`

### 10.1 Full Field List with Defaults

`topics.mode` = `"explicit"`  
`topics.rgb_topic` = `None`  
`topics.depth_topic` = `None`  
`topics.imu_accel_topic` = `None`  
`topics.imu_gyro_topic` = `None`  
`topics.auto.rgb_regex` = `"color.*compressed"`  
`topics.auto.depth_regex` = `"depth.*compressedDepth"`  
`topics.auto.imu_regex` = `"imu.*sample"`  

`expected_rates.image_hz` = `30.0`  
`expected_rates.imu_hz` = `200.0`  

`sampling.rgb_stride` = `5`  
`sampling.max_rgb_frames` = `12000`  

`thresholds.image_gap_factor` = `2.5`  
`thresholds.imu_gap_factor` = `5.0`  
`thresholds.sync_warn_ms` = `16.0`  
`thresholds.sync_fail_ms` = `33.0`  
`thresholds.imu_window_ms` = `20.0`  
`thresholds.blur_threshold_min` = `80.0`  
`thresholds.blur_roi_margin_ratio` = `0.05`  
`thresholds.low_clip_threshold` = `0.05` (legacy/ignored by v1.3 exposure classifier)  
`thresholds.high_clip_threshold` = `0.05` (legacy/ignored)  
`thresholds.contrast_min` = `25.0` (legacy/ignored)  
`thresholds.depth_invalid_threshold` = `0.35`  
`thresholds.drop_warn_ratio` = `0.05`  
`thresholds.drop_fail_ratio` = `0.10`  
`thresholds.imu_missing_warn_ratio` = `0.10`  
`thresholds.blur_fail_warn_ratio` = `0.20`  
`thresholds.exposure_bad_warn_ratio` = `0.20`  
`thresholds.depth_invalid_mean_warn` = `0.35`  
`thresholds.low_clip_pixel_value` = `5`  
`thresholds.high_clip_pixel_value` = `250`  
`thresholds.exposure_roi_margin_ratio` = `0.05`  
`thresholds.low_clip_warn` = `0.20`  
`thresholds.high_clip_warn` = `0.20`  
`thresholds.dynamic_range_min` = `10.0`  
`thresholds.median_dark` = `40.0`  
`thresholds.median_bright` = `215.0`  

`segments.max_gap_fill_ms` = `200.0`  
`segments.min_segment_seconds` = `5.0`  

`gate.fail_if_no_segments_min_duration_s` = `30.0`  
`gate.gate_warn_floor_error_codes` = `["TIMESTAMP_OUT_OF_ORDER_HIGH"]`  

`integrity.out_of_order_warn_ratio` = `0.001`  

`decode.warn_on_depth_pixel_decode_failure` = `False`  

`debug.export_exposure_csv` = `True`  
`debug.export_blur_csv` = `True`  
`debug.export_evidence_frames` = `False`  
`debug.export_evidence_on_warn` = `True`  
`debug.evidence_frames_k` = `16`  
`debug.export_preview_frames` = `True`

### 10.2 Validation Rules

`load_config` / `_validate_config` enforces:
- `topics.mode` in `{"explicit", "auto"}`
- `sampling.rgb_stride > 0`
- `sampling.max_rgb_frames > 0`
- `integrity.out_of_order_warn_ratio >= 0`
- `segments.min_segment_seconds > 0`
- `thresholds.exposure_roi_margin_ratio in [0, 0.5)`
- `thresholds.blur_roi_margin_ratio in [0, 0.5)`
- `thresholds.low_clip_pixel_value in [0, 255]`
- `thresholds.high_clip_pixel_value in [0, 255]`
- `thresholds.low_clip_warn in [0, 1]`
- `thresholds.high_clip_warn in [0, 1]`
- `thresholds.median_dark in [0, 255]`
- `thresholds.median_bright in [0, 255]`
- `thresholds.dynamic_range_min in [0, 255]`
- `debug.evidence_frames_k in [1, 64]`

### 10.3 Topic Override Rules

`apply_topic_overrides(...)`:
- non-null CLI overrides replace config topics
- if any override is provided, `topics.mode` forced to `"explicit"`
- merged config is re-validated

## 11) Topic Selection and Stream Modes

Source: `src/egologqa/topic_select.py`

Explicit mode:
- uses configured topic names directly
- IMU mode resolution:
  - both accel+gyro set and equal -> `single_topic_assumed_both`
  - both set and different -> `dual_topics`
  - only one set -> copy to both, mode `single_topic_assumed_both`
  - neither -> `none`

Auto mode:
- regex matching over scanned topic names
- candidate score tuple: `(-message_count, rate_distance, topic_lexical)`
- selects:
  - one RGB
  - one depth
  - up to two IMU topics
- one IMU candidate means accel=gyro same topic

## 12) Timestamp Semantics

Source: `src/egologqa/time.py`

`extract_stamp_ns(message, fallback_ns) -> (timestamp_ns, source, used_fallback)`

Rules:
1. Read `header.stamp.sec` + `header.stamp.nanosec`.
2. If header-derived ns is valid (`int > 0`), use it with source `"header"`.
3. Else if fallback `log_time_ns` valid (`int > 0`), use fallback with source `"log_time"`.
4. Else return `(0, "invalid", True)`.

Presence rule downstream:
- stream has usable timestamps only if at least 2 valid timestamps (`len(times_ns) >= 2`)

## 13) Reader and MCAP Compatibility

Source: `src/egologqa/io/reader.py`

Abstractions:
- `MessageSource` protocol:
  - `scan_topics()`
  - `iter_messages(topics: Optional[set[str]])`
- implementations:
  - `InMemoryMessageSource` for tests
  - `MCapMessageSource` for production

Compatibility bridge:
- `_extract_record_fields(item)` supports:
  - old tuple shape `(schema, channel, message, ros_msg)`
  - newer object shape with `schema/channel/ros_msg/log_time_ns/publish_time_ns`
  - nested `message.log_time` / `message.publish_time` fallback

Failure mode:
- if `mcap_ros2.reader.read_ros2_messages` import fails:
  - raises `RuntimeError("mcap-ros2-support is required to read MCAP files")`

Shim:
- `src/egologqa/reader.py` re-exports reader classes for backward import compatibility.

## 14) Decoding Semantics

### 14.1 RGB Decoder (`src/egologqa/decoders/rgb.py`)

`decode_rgb_message(msg) -> (frame_bgr | None, error_code | None)`

Logic:
- imports `cv2` lazily
- reads `msg.data`
- decodes with `cv2.imdecode(..., cv2.IMREAD_COLOR)`
- returns:
  - success: `(frame, None)`
  - failure: `(None, "RGB_DECODE_FAIL")`

### 14.2 Depth Decoder (`src/egologqa/decoders/depth.py`)

`decode_depth_message(msg) -> (depth_u16_2d | None, error_code | None)`

Logic:
- searches payload for PNG signature bytes `\x89PNG\r\n\x1a\n`
- decodes from signature onward with `IMREAD_UNCHANGED`
- accepts only `uint16` and 2D

Possible codes:
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`
- `DEPTH_UNEXPECTED_SHAPE`

## 15) Time/Integrity Metric Algorithms

Source: `src/egologqa/metrics/time_metrics.py`

### 15.1 Out-of-Order Ratio

`compute_out_of_order_ratio(times_ms)`:
- inversions = count of `arr[i] < arr[i-1]`
- ratio = inversions / `(n-1)` if `n >= 2`, else `0.0`

### 15.2 Gap/Drop Detection

`compute_stream_gaps(times_ms, gap_factor)`:
- computes positive deltas only (`dt > 0`)
- expected dt:
  - p10/p90 bounds over positive deltas
  - inner set = deltas within `[p10, p90]`
  - expected = median(inner) if inner non-empty else median(positive)
- gap mask: `dt > gap_factor * expected_dt`
- gap intervals recorded as `(t_prev, t_curr)`
- gap ratio: `gap_count / (n-1)`
- if insufficient data -> expected dt `None`, ratio `0.0`, intervals `[]`

Boundary semantics consumed elsewhere:
- drop intervals interpreted as left-open/right-closed: `(left, right]`

### 15.3 Nearest Alignment

`nearest_abs_delta(query, sorted_ref)`:
- searchsorted left/right neighbors
- delta = min(abs(left-query), abs(right-query))

`nearest_indices(query, sorted_ref)`:
- same neighbor selection with left tie-break
- returns int64 indices in sorted ref array

### 15.4 Sync Metrics

`compute_sync_metrics(rgb_times_ms, depth_times_ms_for_index, sync_fail_ms)`:
- returns `None` metrics if either side empty
- else computes deltas via nearest-depth matching
- outputs:
  - `sync_p50_ms`
  - `sync_p95_ms`
  - `sync_max_ms`
  - `sync_fail_ratio` (`mean(delta > sync_fail_ms)`)

### 15.5 IMU Coverage

`compute_imu_coverage(rgb_times_ms, imu_times_ms, window_ms)`:
- if no RGB times -> `[]`
- if no IMU times -> `[False]*len(rgb)`
- for each RGB time `t`:
  - coverage true if any IMU in `[t-window_ms, t+window_ms]`

## 16) Pixel Metric Algorithms

Source: `src/egologqa/metrics/pixel_metrics.py`

## 16.1 RGB Metrics

Function:
- `compute_rgb_pixel_metrics(rgb_frames, thresholds, sample_indices, sample_times_ms)`

Returns tuple:
1. metrics dict
2. `blur_ok` list aligned to input `rgb_frames`
3. `exposure_ok` list aligned to input `rgb_frames`
4. `exposure_rows` for debug CSV
5. `exposure_compute_errors`

If `cv2` unavailable or no frames:
- returns metric dict with `None` metrics and zero counts
- returns empty flag lists and empty rows/errors

Per-frame blur:
- `gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
- blur ROI from `_blur_roi(gray, thresholds.blur_roi_margin_ratio)`
- if ROI invalid -> fallback full frame and margin `0.0`
- blur value:
  - `cv2.Laplacian(blur_roi, cv2.CV_64F).var()`

Blur aggregate:
- valid blur scores = non-None scores only
- threshold = `thresholds.blur_threshold_min` (fixed scalar)
- `blur_fail_ratio = mean(score < threshold)` over valid scores only
- `blur_ok[i] = True` when score missing, else `score >= threshold`

Per-frame exposure features (on `_exposure_roi`):
- `low_clip = mean(roi <= low_clip_pixel_value)`
- `high_clip = mean(roi >= high_clip_pixel_value)`
- percentiles `p01,p05,p50,p95,p99`
- `contrast = p99 - p01`
- `dynamic_range = p95 - p05`

Exposure-bad conditions:
- `bad_flat_and_dark`: `dynamic_range < dynamic_range_min AND p50 < median_dark`
- `bad_flat_and_bright`: `dynamic_range < dynamic_range_min AND p50 > median_bright`
- `bad_saturation_dark`: `low_clip > low_clip_warn AND p50 < median_dark`
- `bad_saturation_bright`: `high_clip > high_clip_warn AND p50 > median_bright`
- final `bad = OR(all four)`

Reason labels:
- `low_clip`
- `high_clip`
- `flat_and_dark`
- `flat_and_bright`

Key semantic constraint:
- high low_clip or high high_clip alone does not mark bad unless p50 context condition is also true.

Exception handling inside per-frame loop:
- exposure flag defaults non-blocking (`True`) on compute exception
- collects structured error row in `exposure_compute_errors`

Derived exposure aggregates:
- `exposure_bad_ratio` over successfully computed exposure rows only
- first/last bad sample indices
- clip and contrast and dynamic-range and p50 aggregates
- dark frame ratio and low_clip_when_dark_mean
- reason counts dict

### 16.2 Depth Metrics

Function:
- `compute_depth_pixel_metrics(depth_frames, thresholds)`

Per frame:
- invalid ratio = `mean(depth == 0)`

Aggregate:
- `depth_invalid_mean`
- `depth_invalid_p95`
- `depth_fail_ratio = mean(invalid_ratio > depth_invalid_threshold)`
- returns `depth_ok` list where pass means invalid ratio <= threshold

No frames:
- returns metric dict with `None`s and empty `depth_ok`.

## 17) Frame Flag Construction

Source: `src/egologqa/frame_flags.py`

`build_frame_flags(...) -> FrameFlags`

Inputs include:
- sampled times/indices
- sync deltas + thresholds
- drop regions
- IMU coverage arrays
- pixel pass arrays
- decode/timestamp support booleans
- forced bad positions

Per sampled position:
- sync flags:
  - if sync unavailable globally: `sync_ok_fail=True`, `sync_ok_warn=True`, `sync_available=False`
  - if sync available for position: compare delta to fail/warn thresholds
  - if global true but per-position missing: `sync_ok_fail=False`, `sync_ok_warn=False`, `sync_available=False`
- drop flag:
  - `rgb_drop_ok = not drop_regions.contains(t_ms)`
- IMU flag:
  - if IMU exists and coverage arrays exist, require accel AND gyro
  - else `True`
- pixel flags:
  - blur: default true if missing
  - exposure: forced true if RGB pixels unsupported
  - depth: forced true if depth pixels unsupported

Composite flags:
- `integrity_ok = sync_ok_fail AND rgb_drop_ok AND imu_ok`
- `vision_ok = integrity_ok AND blur_ok AND exposure_ok AND depth_ok`
- forced-bad positions set both to `False`

Return fields include alias:
- `sync_ok` is deprecated alias of `sync_ok_fail`

## 18) Segment Extraction

Source: `src/egologqa/segments.py`

`extract_segments(sampled_times_ns, frame_ok, max_gap_fill_ms, min_segment_seconds, forced_break_positions)`

Algorithm:
- iterate sampled positions
- starts segment on first `ok=True`
- extends while subsequent ok samples satisfy:
  - not forced break at current or previous ok position
  - not monotonic inversion (`t_ns < last_ok_ns`)
  - inter-ok gap <= `max_gap_fill_ms`
- on break, closes prior segment at last ok time
- appends only if duration >= `min_segment_seconds`
- returns list of objects:
  - `start_ns` (int)
  - `end_ns` (int)
  - `duration_s` (float)

## 19) Gate Evaluation Contract

Source: `src/egologqa/gate.py` + `src/egologqa/constants.py`

### 19.1 FAIL Reason Order (fixed)

1. `FAIL_NO_RGB_STREAM`
2. `FAIL_ANALYSIS_ERROR`
3. `FAIL_SYNC_P95_GT_FAIL`
4. `FAIL_DROP_RATIO_GT_FAIL`
5. `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`

### 19.2 WARN Reason Order (fixed)

1. `WARN_DEPTH_TIMESTAMP_MISSING`
2. `WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED`
3. `WARN_RGB_PIXEL_DECODE_UNSUPPORTED`
4. `WARN_SYNC_P95_GT_WARN`
5. `WARN_DROP_RATIO_GT_WARN`
6. `WARN_IMU_MISSING_RATIO_GT_WARN`
7. `WARN_BLUR_FAIL_RATIO_GT_WARN`
8. `WARN_EXPOSURE_BAD_RATIO_GT_WARN`
9. `WARN_DEPTH_INVALID_MEAN_GT_WARN`

### 19.3 Recommended Action Mapping

- PASS -> `USE_FULL_SEQUENCE`
- WARN -> `USE_SEGMENTS_ONLY`
- FAIL -> `RECAPTURE_OR_SKIP`

### 19.4 Trigger Rules

FAIL set:
- missing RGB timestamps presence -> `FAIL_NO_RGB_STREAM`
- any `errors[]` with severity ERROR -> `FAIL_ANALYSIS_ERROR`
- `sync_p95_ms > thresholds.sync_fail_ms` -> `FAIL_SYNC_P95_GT_FAIL`
- `drop_ratio > thresholds.drop_fail_ratio` -> `FAIL_DROP_RATIO_GT_FAIL`
- no segments and duration >= `gate.fail_if_no_segments_min_duration_s` -> `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`

WARN set:
- depth timestamps missing -> `WARN_DEPTH_TIMESTAMP_MISSING`
- optional depth pixel unsupported warn controlled by `decode.warn_on_depth_pixel_decode_failure`
- RGB pixel unsupported -> `WARN_RGB_PIXEL_DECODE_UNSUPPORTED`
- `sync_p95_ms > thresholds.sync_warn_ms` -> `WARN_SYNC_P95_GT_WARN`
- `drop_ratio > thresholds.drop_warn_ratio` -> `WARN_DROP_RATIO_GT_WARN`
- IMU combined missing above threshold -> `WARN_IMU_MISSING_RATIO_GT_WARN`
- blur fail ratio above threshold -> `WARN_BLUR_FAIL_RATIO_GT_WARN`
- exposure bad ratio above threshold -> `WARN_EXPOSURE_BAD_RATIO_GT_WARN`
- depth invalid mean above threshold -> `WARN_DEPTH_INVALID_MEAN_GT_WARN`

Gate finalization:
- FAIL if fail reasons non-empty
- else WARN if warn reasons non-empty
- else PASS

WARN floor:
- if gate currently PASS and any WARN error code in `gate_warn_floor_error_codes`, force gate to WARN

## 20) Pipeline Flow (Authoritative)

Source: `src/egologqa/pipeline.py`

### 20.1 Function Signature

`analyze_file(input_path, output_dir, config, overrides=None, source=None, progress_cb=None) -> AnalysisResult`

### 20.2 Ordered Execution

1. Normalize paths and apply topic overrides.
2. Initialize empty report with tool/input defaults.
3. Store resolved config under `config_used`.
4. Append legacy exposure warning if legacy keys differ from defaults.
5. Emit progress phase `scan`.
6. Build source object (`MCapMessageSource` unless injected source).
7. Scan topics and select active streams.
8. Populate `streams` topic metadata and per-topic stats.
9. Pass 1 timestamp collection over selected topics.
10. Populate timestamp presence and decode status for depth timestamps.
11. Emit `STREAM_TIMESTAMPS_MISSING` WARN per stream lacking >=2 valid timestamps despite having messages.
12. Compute duration, sampling settings, out-of-order metrics.
13. Emit `TIMESTAMP_OUT_OF_ORDER_HIGH` WARN for streams over configured ratio.
14. Compute RGB gap metrics and instantiate `DropRegions`.
15. Compute sync metrics (alignment context may sort sequences for nearest-neighbor use).
16. Compute IMU coverage and missing ratios.
17. Compute sampled RGB indices.
18. Build forced-break sampled positions around inversion indices.
19. Build nearest depth index map for sampled RGB positions.
20. Emit phase `pass2`.
21. Pass 2 decode loop over RGB/depth topics:
    - count decode attempts/successes
    - capture decode error maps by sample position
22. Run RGB and depth pixel metric calculators on decoded subsets.
23. Expand per-position pixel flags back to full sampled count defaults.
24. Store decode counters and depth valid frame count.
25. Set `streams.decode_status` pixel support.
26. Emit decode unavailability warnings:
    - RGB unsupported -> dominant decode code WARN
    - depth unsupported with depth timestamps -> dominant decode code WARN
27. Aggregate depth dtype non-uint16 warning:
    - `DEPTH_DTYPE_NON_UINT16_SEEN`
28. Translate per-frame exposure compute exceptions into `EXPOSURE_COMPUTE_FAILED` ERROR entries.
29. Emit blur unavailable warning if blur valid frame count is zero:
    - `BLUR_UNAVAILABLE_NO_DECODE`
30. Optional exposure CSV export; if enabled but rows unavailable -> `RGB_EXPOSURE_DEBUG_UNAVAILABLE`.
31. Build blur/depth debug rows and export CSVs when enabled.
32. Decide blur evidence export:
    - explicit debug flag OR auto-on-warn trigger
33. Optional blur fail/pass evidence frame writing and metric paths.
34. Build sampled IMU coverage arrays.
35. Emit sync unavailability WARN if depth timestamps missing globally or partially.
36. Build frame flags.
37. Compute integrity/vision ratios and estimated coverage seconds.
38. Set `segments_basis = "integrity"`.
39. Extract integrity segments with forced breaks.
40. Store `errors` and optionally `sync_sample_count`.
41. Evaluate gate and store gate object.
42. Optional preview frame export.
43. Optional sync histogram and drop timeline plot export.
44. Store artifact paths as output-relative POSIX strings.
45. Emit progress `done`.

Exception path:
- any exception in main try block:
  - append `ANALYSIS_EXCEPTION` ERROR
  - gate forced FAIL with `FAIL_ANALYSIS_ERROR`
  - emit phase `error`

Finalization always attempted:
- best-effort write `report.md` (exceptions swallowed)
- write canonical `report.json` (always)
- return populated `AnalysisResult`

### 20.3 Progress Callback Contract

Phases emitted:
- `scan`
- `pass1`
- `pass2`
- `done`
- `error`

Event shape:
- `phase`
- `progress` (0.0..1.0)
- `message`
- optional `partial` object

## 21) Report Schema Specification

Source: `src/egologqa/report.py` + pipeline updates

### 21.1 Top-Level Keys (fixed)

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

### 21.2 `tool` Object

- `name`
- `version`
- `git_commit`

### 21.3 `input` Object

- `file_path`
- `file_size_bytes`
- `analyzed_at_utc`

### 21.4 `streams` Object

- `rgb_topic`
- `depth_topic`
- `imu_accel_topic`
- `imu_gyro_topic`
- `imu_mode`
- `topic_stats` (topic -> message_count/approx_rate_hz/duration_s)
- `depth_topic_present`
- `depth_timestamps_present`
- `rgb_timestamps_present`
- `decode_status`:
  - `rgb_pixels` (`supported|unsupported`)
  - `depth_pixels` (`supported|unsupported`)
  - `depth_timestamps` (`present|missing`)

### 21.5 `time` Object

- `time_base` (`"ns"`)
- `duration_s`

### 21.6 `sampling` Object

- `rgb_stride`
- `max_rgb_frames`
- `frames_analyzed`

### 21.7 `metrics` Object (Current Key Set)

53 baseline keys in empty report:
- `sync_p50_ms`
- `sync_p95_ms`
- `sync_max_ms`
- `sync_fail_ratio`
- `expected_rgb_dt_ms`
- `drop_ratio`
- `imu_accel_missing_ratio`
- `imu_gyro_missing_ratio`
- `imu_combined_missing_ratio`
- `blur_median`
- `blur_threshold`
- `blur_fail_ratio`
- `blur_p10`
- `blur_p50`
- `blur_p90`
- `blur_valid_frame_count`
- `exposure_bad_ratio`
- `exposure_valid_frame_count`
- `exposure_bad_first_sample_i`
- `exposure_bad_last_sample_i`
- `low_clip_mean`
- `low_clip_p95`
- `high_clip_mean`
- `high_clip_p95`
- `contrast_mean`
- `contrast_p05`
- `dynamic_range_mean`
- `dynamic_range_p05`
- `p50_mean`
- `p50_p05`
- `p50_p95`
- `dark_frame_ratio`
- `low_clip_when_dark_mean`
- `exposure_bad_reason_counts`
- `exposure_debug_csv_path`
- `blur_debug_csv_path`
- `depth_debug_csv_path`
- `blur_fail_frames_dir`
- `blur_pass_frames_dir`
- `rgb_decode_attempt_count`
- `rgb_decode_success_count`
- `depth_decode_attempt_count`
- `depth_decode_success_count`
- `depth_valid_frame_count`
- `integrity_ok_ratio`
- `integrity_coverage_seconds_est`
- `vision_ok_ratio`
- `vision_coverage_seconds_est`
- `segments_basis`
- `depth_invalid_mean`
- `depth_invalid_p95`
- `depth_fail_ratio`
- `out_of_order`

Additional keys may be added by pipeline when data exists:
- `preview_count`
- `sync_histogram_path`
- `drop_timeline_path`
- `sync_sample_count`

### 21.8 `gate` Object

- `gate` (`PASS|WARN|FAIL`)
- `recommended_action`
- `fail_reasons` (ordered list)
- `warn_reasons` (ordered list)

### 21.9 `segments` Array

Each item:
- `start_ns`
- `end_ns`
- `duration_s`

### 21.10 `errors[]` Schema

Each item strictly:
- `severity` (`WARN|ERROR`)
- `code` (string)
- `message` (string)
- `context` (object)

## 22) Canonical JSON and Stability Rules

Source: `src/egologqa/report.py`

`write_report_json` behavior:
- creates output directory if needed
- writes to `report.json`
- serializes with:
  - `sort_keys=True`
  - separators `(",", ":")`
  - `ensure_ascii=True`
  - trailing newline

Sanitization:
- recursively processes dict/list
- floats:
  - NaN/Inf -> `null`
  - finite float -> rounded to 4 decimals

Guarantee model:
- semantic stability for same input/config
- not guaranteed byte-identical across all environments

## 23) Artifacts and Path Semantics

Source: `src/egologqa/artifacts.py` + pipeline path normalization

Produced artifacts (condition-dependent):
- always:
  - `report.json`
- best-effort:
  - `report.md`
- optional:
  - `previews/rgb_####_sample_######.png` (up to 12)
  - `plots/sync_histogram.png`
  - `plots/drop_timeline.png`
  - `debug/exposure_samples.csv`
  - `debug/blur_samples.csv`
  - `debug/depth_samples.csv`
  - `debug/blur_fail_frames/*.jpg`
  - `debug/blur_pass_frames/*.jpg`

Path serialization in report:
- pipeline stores artifact paths via `_relative_path(...)`
- preferred form: output-relative POSIX path
- fallback form if relative conversion fails: normalized slash string

CSV formatting:
- float values formatted to six decimal places in artifact writer
- rows sorted deterministically by sample index/time

Blur evidence selection:
- fail rows sorted ascending `(blur_value, sample_i)` and take first `k`
- pass rows sorted descending `(blur_value)` tie-broken by `sample_i` ascending
- deterministic filenames include rank/sample/time/blur

## 24) Code/Reason Catalog

### 24.1 Gate Enum Reasons

FAIL:
- `FAIL_NO_RGB_STREAM`
- `FAIL_ANALYSIS_ERROR`
- `FAIL_SYNC_P95_GT_FAIL`
- `FAIL_DROP_RATIO_GT_FAIL`
- `FAIL_NO_CLEAN_SEGMENTS_LONG_ENOUGH`

WARN:
- `WARN_DEPTH_TIMESTAMP_MISSING`
- `WARN_DEPTH_PIXEL_DECODE_UNSUPPORTED`
- `WARN_RGB_PIXEL_DECODE_UNSUPPORTED`
- `WARN_SYNC_P95_GT_WARN`
- `WARN_DROP_RATIO_GT_WARN`
- `WARN_IMU_MISSING_RATIO_GT_WARN`
- `WARN_BLUR_FAIL_RATIO_GT_WARN`
- `WARN_EXPOSURE_BAD_RATIO_GT_WARN`
- `WARN_DEPTH_INVALID_MEAN_GT_WARN`

### 24.2 Diagnostics/Error Codes Emitted

- `CONFIG_LOAD_ERROR`
- `ANALYSIS_EXCEPTION`
- `STREAM_TIMESTAMPS_MISSING`
- `TIMESTAMP_OUT_OF_ORDER_HIGH`
- `LEGACY_EXPOSURE_KEYS_IGNORED`
- `RGB_DECODE_FAIL`
- `DEPTH_PNG_SIGNATURE_NOT_FOUND`
- `DEPTH_PNG_IMDECODE_FAIL`
- `DEPTH_UNEXPECTED_DTYPE`
- `DEPTH_UNEXPECTED_SHAPE`
- `DEPTH_DTYPE_NON_UINT16_SEEN`
- `RGB_EXPOSURE_DEBUG_UNAVAILABLE`
- `EXPOSURE_COMPUTE_FAILED`
- `BLUR_UNAVAILABLE_NO_DECODE`
- `SYNC_UNAVAILABLE_DEPTH_TIMESTAMPS_MISSING`

## 25) Module-by-Module API Reference

### 25.1 `src/egologqa/cli.py`

- `build_parser()`
- `main(argv=None)`

### 25.2 `src/egologqa/config.py`

- `load_config(config_path)`
- `apply_topic_overrides(config, overrides)`
- `config_to_dict(config)`
- `_merge_dataclass(obj, patch)`
- `_validate_config(config)`

### 25.3 `src/egologqa/pipeline.py`

- `StreamCollector` class
- `analyze_file(...)`
- helpers:
  - `_duration_seconds`
  - `_max_error_code`
  - `_error`
  - `_relative_path`
  - `_append_legacy_exposure_keys_warning`
  - `_emit`

### 25.4 `src/egologqa/report.py`

- `now_utc_iso()`
- `git_commit_or_unknown(cwd=None)`
- `write_report_json(report, output_dir)`
- `sanitize_json_value(value)`
- `empty_report(input_path, file_size_bytes=None, commit="unknown")`

### 25.5 `src/egologqa/gate.py`

- `evaluate_gate(config, metrics, streams, duration_s, segments, errors)`

### 25.6 `src/egologqa/time.py`

- `extract_stamp_ns(message, fallback_ns)`
- `is_valid_timestamp_ns(value)`
- `_nested_attr(obj, dotted)`

### 25.7 `src/egologqa/topic_select.py`

- `select_topics(config, stats)`
- `_select_explicit(config)`
- `_select_auto(config, stats)`
- `_pick_topic(stats, regex, expected_rate_hz)`
- `_pick_topics(stats, regex, expected_rate_hz)`

### 25.8 `src/egologqa/sampling.py`

- `sample_rgb_indices(total_frames, stride, max_frames)`

### 25.9 `src/egologqa/drop_regions.py`

- `DropRegions` class
  - `contains(t_ms)` with `(left,right]` semantics

### 25.10 `src/egologqa/frame_flags.py`

- `FrameFlags` dataclass
- `build_frame_flags(...)`

### 25.11 `src/egologqa/segments.py`

- `extract_segments(...)`
- `_append_segment(...)`

### 25.12 `src/egologqa/artifacts.py`

- `write_report_markdown(...)`
- `write_rgb_previews(...)`
- `write_sync_histogram(...)`
- `write_drop_timeline(...)`
- `write_exposure_debug_csv(...)`
- `write_blur_debug_csv(...)`
- `write_depth_debug_csv(...)`
- `write_blur_evidence_frames(...)`
- `_write_evidence_set(...)`
- `_format_csv_value(...)`

### 25.13 `src/egologqa/metrics/time_metrics.py`

- `compute_out_of_order_ratio(...)`
- `compute_stream_gaps(...)`
- `compute_sync_metrics(...)`
- `nearest_abs_delta(...)`
- `nearest_indices(...)`
- `compute_imu_coverage(...)`

### 25.14 `src/egologqa/metrics/pixel_metrics.py`

- `compute_rgb_pixel_metrics(...)`
- `compute_depth_pixel_metrics(...)`
- `_exposure_roi(...)`
- `_blur_roi(...)`

### 25.15 `src/egologqa/decoders/rgb.py`

- `decode_rgb_message(msg)`

### 25.16 `src/egologqa/decoders/depth.py`

- `PNG_SIGNATURE` constant
- `decode_depth_message(msg)`

### 25.17 `src/egologqa/io/reader.py`

- `MessageSource` protocol
- `InMemoryMessageSource`
- `MCapMessageSource`
- `_extract_record_fields(item)`
- `_to_int(value, default)`
- `_to_optional_int(value)`

### 25.18 `src/egologqa/io/hf_fetch.py`

- `_load_hf_clients()`
- `_as_dict(entry)`
- `list_mcap_files(...)`
- `_raise_for_http_error(...)`
- `download_to_temp(...)`
- `resolve_cached_file(...)`

### 25.19 `src/egologqa/io/local_fs.py`

- exception classes:
  - `LocalDirNotFound`
  - `LocalDirNotReadable`
  - `TooManyFiles`
- `_resolve_dir_path(dir_path)`
- `list_mcap_files_in_dir(dir_path, max_files=500)`
- `is_readable_file(path)`

### 25.20 `src/egologqa/kiosk_helpers.py`

- `resolve_runs_base_dir`
- `sanitize_component`
- `build_run_basename`
- `allocate_run_dir`
- `write_latest_run_pointer`
- `ensure_writable_dir`
- `human_bytes`
- `build_hf_display_label`
- `make_local_option_label`
- `build_timestamped_run_basename`
- `resolve_source_kind`
- `map_error_bucket`

## 26) Test Matrix and Locked Behaviors

Full suite currently passing: 68 tests.

Coverage map:
- `tests/integration/test_pipeline_inmemory.py`
  - report top-level key set locked
  - core metrics presence checks
  - `errors[]` object shape checks

- `tests/unit/test_reader_compat.py`
  - old/new MCAP API shape compatibility

- `tests/unit/test_time_extraction.py`
  - header priority, fallback, invalid cases

- `tests/unit/test_sampling.py`
  - deterministic sampling behavior

- `tests/unit/test_drop_regions.py`
  - `(left,right]` boundary correctness

- `tests/unit/test_nearest_alignment.py`
  - nearest delta/index correctness

- `tests/unit/test_out_of_order.py`
  - inversion counting

- `tests/unit/test_segments.py`
  - forced-break segment split behavior

- `tests/unit/test_gate.py`
  - reason ordering and WARN floor behavior

- `tests/unit/test_report.py`
  - float rounding + NaN/Inf sanitization

- `tests/unit/test_exposure_conditional_saturation.py`
  - low/high clip requires p50 context
  - flat dark/bright triggers

- `tests/unit/test_exposure_debug_csv.py`
  - exposure CSV on/off/unavailable behavior

- `tests/unit/test_integrity_segments_exposure.py`
  - exposure cannot erase integrity segments
  - exposure compute failure recorded without segment wipeout

- `tests/unit/test_blur_denominator_pipeline.py`
  - blur denominator uses decoded RGB frames only

- `tests/unit/test_blur_depth_debug_csv.py`
  - blur/depth CSV schema and sort order

- `tests/unit/test_blur_evidence_selection.py`
  - deterministic evidence selection and naming

- `tests/unit/test_blur_formula_lock.py`
  - exact blur formula lock (grayscale, ROI, Laplacian variance)

- `tests/unit/test_blur_warn_logic.py`
  - blur warn only when ratio exists and exceeds threshold

- `tests/unit/test_depth_dtype_warning.py`
  - aggregated non-uint16 depth warning behavior

- `tests/unit/test_frame_flags_sync_alias.py`
  - `sync_ok` alias equals `sync_ok_fail`

- `tests/unit/test_hf_fetch.py`
  - HF list filtering/sorting
  - download progress and HTTP/network error mapping
  - cache hit/miss behavior and auth error mapping

- `tests/unit/test_local_fs.py`
  - local listing sort/filter/limits/readability errors

- `tests/unit/test_streamlit_kiosk_helpers.py`
  - helper utilities determinism, formatting, directory handling

- `tests/unit/test_sync_unavailable_graceful.py`
  - missing depth timestamps does not erase integrity segments

- `tests/unit/test_sync_warn_vs_fail_integrity.py`
  - sync warn-level offsets do not trigger fail-level shredding

## 27) Known Practical Caveats

- `--output` is directory-oriented; passing file-like value (e.g., `report.json`) creates folder `report.json/` containing `report.json`.
- If `cv2` unavailable, pixel metrics become unavailable/degraded but analysis continues.
- Streamlit requires writable runs/cache directories; helper probes write access explicitly.
- Depth decode unsupported does not block integrity segmentation by design.
- `src/egologqa.egg-info/PKG-INFO` can lag source README/UI wording if editable metadata not regenerated.

## 28) Rebuild-the-Project Blueprint (From Scratch)

This section specifies a deterministic reconstruction plan for this exact project behavior.

### Step 1: Initialize package and tooling

1. Create package name `egologqa`.
2. Set Python requirement to `>=3.11,<3.12`.
3. Add dependencies exactly:
   - `mcap-ros2-support==0.5.5`
   - `huggingface-hub==0.27.1`
   - `numpy==1.26.4`
   - `opencv-python==4.10.0.84`
   - `PyYAML==6.0.2`
   - `streamlit==1.41.1`
4. Add dev dependency `pytest==8.3.4`.
5. Configure console script `egologqa = egologqa.cli:main`.

### Step 2: Implement datamodel and config loading

1. Recreate all dataclasses in `models.py` with exact fields/defaults in section 10.
2. Implement recursive dataclass merge from YAML.
3. Implement validation checks exactly as listed.
4. Implement topic override merge behavior and explicit-mode coercion.

### Step 3: Implement core utility modules

1. `time.py` timestamp extraction semantics.
2. `sampling.py` deterministic sampling.
3. `drop_regions.py` with `(left,right]` `contains`.
4. `topic_select.py` scoring and IMU mode behavior.
5. `constants.py` fixed reason ordering and action map.

### Step 4: Implement decoders and time metrics

1. RGB decoder returns only `RGB_DECODE_FAIL` on failure.
2. Depth decoder supports embedded-PNG extraction and strict uint16/2D checks.
3. Implement `time_metrics.py` functions exactly (gap expected dt, nearest matching, IMU coverage).

### Step 5: Implement pixel metrics

1. Implement blur formula exactly:
   - grayscale conversion
   - ROI margin
   - Laplacian variance
2. Implement exposure conditional classifier exactly with p50 context dependencies.
3. Ensure exposure compute exceptions do not set blocking false flags.
4. Return all metrics keys and debug rows expected by pipeline.

### Step 6: Implement frame flags and segments

1. Integrity definition: sync_fail-level + drop + IMU.
2. Vision definition: integrity + blur + exposure + depth.
3. Preserve `sync_ok` alias behavior.
4. Implement forced-break and monotonic-break segment splitting.

### Step 7: Implement gate evaluator

1. Evaluate fail/warn condition sets.
2. Emit reasons in fixed order arrays only.
3. Implement WARN floor from configured warn codes.
4. Return gate with recommended action mapping.

### Step 8: Implement report writer

1. Empty report with fixed top-level structure.
2. Canonical JSON dump (`sort_keys`, fixed separators, ASCII).
3. float rounding and NaN/Inf to null sanitization.

### Step 9: Implement artifacts module

1. Markdown report writer.
2. RGB preview writer with deterministic naming.
3. Sync histogram and drop timeline image generators.
4. Exposure/blur/depth CSV writers with sorted rows and fixed headers.
5. Blur evidence selector and deterministic naming.

### Step 10: Implement IO adapters

1. Reader protocol + in-memory and MCAP implementations.
2. MCAP record shape compatibility extraction helper.
3. HF list/download/cache resolver utilities and error mapping.
4. Local folder listing/readability utilities and error classes.

### Step 11: Implement pipeline orchestration

1. Follow section 20 order exactly.
2. Keep two-pass pattern and sampled decode mapping logic.
3. Keep code emission behavior and severity conventions.
4. Ensure artifact paths in report are output-relative POSIX strings.
5. Keep exception path with `ANALYSIS_EXCEPTION` and forced FAIL gate.

### Step 12: Implement CLI and Streamlit UI

1. CLI parser and exit code contract.
2. Streamlit kiosk layout with two source tabs and placeholder-first selects.
3. Advanced panel gated by `EGOLOGQA_UI_ADVANCED=1`.
4. Cache behavior and environment variable wiring.
5. Progress/status rendering and same-page results view.

### Step 13: Recreate tests and acceptance gate

1. Recreate all unit and integration tests listed in section 26.
2. Ensure `pytest -q` passes with 68 tests.
3. Validate top-level report keys unchanged.
4. Validate reason ordering and schema contracts.

## 29) Acceptance Checklist for "Exact Same Project"

Use this to validate reconstruction equivalence:

- Packaging and dependency pins match section 6.
- Config defaults and validation rules match section 10.
- Gate reason order and action mapping match section 19.
- Exposure classifier semantics match section 16.
- Segment basis is integrity and cannot be erased by exposure-only failures.
- Report top-level keys exactly match section 21.1.
- `errors[]` entries always use `{severity, code, message, context}`.
- Artifact path fields are output-relative POSIX style when present.
- CLI exit codes match section 8.3.
- Streamlit env contracts match section 9.2.
- Full test suite passes (68/68 on this commit).

## 30) Appendix: Files That Define Most Contracts

- Config/schema:
  - `src/egologqa/models.py`
  - `src/egologqa/config.py`
  - `configs/microagi00_ros2.yaml`
- Pipeline/gating:
  - `src/egologqa/pipeline.py`
  - `src/egologqa/gate.py`
  - `src/egologqa/constants.py`
- Metrics/segments:
  - `src/egologqa/metrics/time_metrics.py`
  - `src/egologqa/metrics/pixel_metrics.py`
  - `src/egologqa/frame_flags.py`
  - `src/egologqa/segments.py`
- IO/decoding:
  - `src/egologqa/io/reader.py`
  - `src/egologqa/io/hf_fetch.py`
  - `src/egologqa/io/local_fs.py`
  - `src/egologqa/decoders/rgb.py`
  - `src/egologqa/decoders/depth.py`
- Output/artifacts:
  - `src/egologqa/report.py`
  - `src/egologqa/artifacts.py`
- Interfaces:
  - `src/egologqa/cli.py`
  - `app/streamlit_app.py`

