# EgoLogQA

EgoLogQA is a CLI-first quality-control gate for MicroAGI00-style ROS2 MCAP logs.

## Stability Contract

EgoLogQA guarantees semantic stability for the same input/config:

- Gate and enum reason codes match exactly.
- Segment boundaries are compared with a 10 ms tolerance.
- Float metrics are rounded to 4 decimals in `report.json` and compared post-rounding.

`report.json` is written with canonical JSON formatting (`sort_keys=true`, fixed separators) and invalid floats are converted to `null`.

Byte-identical output across different machines/OS builds is not guaranteed.

## Install

```bash
pip install -e ".[dev]"
```

## Analyze

```bash
egologqa analyze \
  --input /path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml
```

Default shipped config: `configs/microagi00_ros2.yaml` (explicit MicroAGI00 ROS2 topics).

Segment extraction is integrity-based (`sync + drop/gap + IMU`) so noisy pixel
heuristics cannot erase all usable time. Vision quality is reported separately
through metrics and WARN reasons.

Exposure diagnostics are exported to `debug/exposure_samples.csv` by default.

Note: `thresholds.contrast_min`, `thresholds.low_clip_threshold`, and
`thresholds.high_clip_threshold` remain in config for backward compatibility but
are not used by the v1.3 exposure classifier.
Use `thresholds.low_clip_pixel_value` and `thresholds.high_clip_pixel_value`
to control clip-value cutoffs for exposure diagnostics/classification.

Output directory contains:

- `report.json` (always)
- `report.md`
- `previews/` (sample RGB PNGs when decode succeeds)
- `plots/sync_histogram.png` (when sync deltas are available)
- `plots/drop_timeline.png` (when RGB timestamps are available)
- `debug/exposure_samples.csv` (when RGB decode succeeds and debug export is enabled)
- `debug/blur_samples.csv` (when blur/depth debug CSV export is enabled)
- `debug/depth_samples.csv` (when blur/depth debug CSV export is enabled)
- `debug/blur_fail_frames/*.jpg` (when evidence export is enabled or blur WARN auto-export triggers)
- `debug/blur_pass_frames/*.jpg` (when evidence export is enabled or blur WARN auto-export triggers)

All artifact paths stored in `report.json` are output-directory-relative POSIX paths (for example `plots/sync_histogram.png`).

## Blur Metric Contract

Blur is computed deterministically with the following exact formula:

- `gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)` (no resize, no normalization)
- ROI uses `thresholds.blur_roi_margin_ratio` with pixel margin `m = int(min(H, W) * ratio)`
- fallback to full frame when ROI would be empty
- `blur_value = cv2.Laplacian(gray_roi, cv2.CV_64F).var()`
- `blur_ok = (blur_value >= thresholds.blur_threshold_min)` (fixed threshold, no adaptive gating)

`blur_threshold_min` default (`80.0`) is tuned for 1080p MicroAGI00-style streams and may require retuning for other resolutions.

Blur WARN rule:

- `WARN_BLUR_FAIL_RATIO_GT_WARN` triggers only when
  `blur_fail_ratio > thresholds.blur_fail_warn_ratio`
- blur never triggers FAIL in v1.3
- if no valid decoded RGB frames are available for blur, `blur_fail_ratio` is null and
  `errors[]` includes `BLUR_UNAVAILABLE_NO_DECODE`

Integrity segments remain the source of FAIL/no-segment decisions; vision metrics are advisory/WARN-only.

## Git Hygiene

Runtime analysis artifacts are intentionally local and not tracked by git.

- Ignored runtime output directories include `report_out/` and `out/`.
- Generated previews (`previews/*.png`), plots, CSV debug files, and reports should not be committed.
- If you need a static artifact for documentation, copy it intentionally into a dedicated non-runtime location (for example `examples/`).

Exit codes:

- `0` PASS
- `10` WARN
- `20` FAIL
- `30` ERROR

## Streamlit UI

```bash
./.venv/bin/streamlit run app/streamlit_app.py
```

Run from repo root so local `.streamlit/` config is loaded.

The default UI is kiosk-style with two tabs:

- One instruction line
- `Hugging Face` tab:
  - `.mcap` dropdown (with file-size labels)
  - tab-specific Analyze button
- `Local disk` tab:
  - folder selector (`~/Downloads`, `~/Desktop`, `~/.cache/egologqa`, `Last used`, or `Other...`)
  - `.mcap` dropdown (with file-size labels)
  - tab-specific Analyze button
- Progress + full same-page results after each run

No dataset/revision/prefix/token/cache controls are shown by default.
Both source dropdowns use placeholder-first behavior (`Select an MCAP file`) and `Analyze` is disabled until you explicitly pick a file.
Local disk mode analyzes the selected file in place (no browser upload and no local-file copy into run output).

Defaults:

- HF dataset id: `MicroAGI-Labs/MicroAGI00`
- HF revision: `main`
- HF prefix: `raw_mcaps/`
- HF cache dir: `~/.cache/egologqa/hf_mcaps`
- Runs dir: `~/.cache/egologqa/runs`

Env overrides:

- `EGOLOGQA_HF_REPO_ID`
- `EGOLOGQA_HF_REVISION`
- `EGOLOGQA_HF_PREFIX`
- `EGOLOGQA_HF_CACHE_DIR`
- `EGOLOGQA_RUNS_DIR`
- `EGOLOGQA_LOCAL_MAX_FILES` (default `500`)
- `HF_TOKEN` (optional, not shown in UI)

Developer-only advanced panel:

- Set `EGOLOGQA_UI_ADVANCED=1`
- Enables manual Hugging Face list refresh and error details

Streamlit log noise suppression:

- `.streamlit/config.toml` sets logger level to `error`
- `.streamlit/secrets.toml` is intentionally present (empty)
- Fallback launch flag if needed: `--logger.level=error`
