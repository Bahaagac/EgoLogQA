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
EgoLogQA analyze \
  --input /path/to/file.mcap \
  --output out/run1 \
  --config configs/microagi00_ros2.yaml
```

Default shipped config: `configs/microagi00_ros2.yaml` (explicit MicroAGI00 ROS2 topics).

Segment extraction is integrity-based (`sync + drop/gap + IMU`) so noisy pixel
heuristics cannot erase all usable time. Vision quality is reported separately
through metrics and WARN reasons.

Exposure diagnostics are exported to `debug/exposure_samples.csv` by default.

Recommended actions:

- `USE_FULL_SEQUENCE`
- `USE_SEGMENTS_ONLY`
- `FIX_TIME_ALIGNMENT`
- `RECAPTURE_OR_SKIP`

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
- `debug/exposure_low_clip_frames/*.jpg` (when exposure evidence export is active)
- `debug/exposure_high_clip_frames/*.jpg` (when exposure evidence export is active)
- `debug/exposure_flat_and_dark_frames/*.jpg` (when exposure evidence export is active)
- `debug/exposure_flat_and_bright_frames/*.jpg` (when exposure evidence export is active)
- `debug/exposure_evidence_error.txt` (when exposure evidence selection falls back)
- `debug/blur_samples.csv` (when blur/depth debug CSV export is enabled)
- `debug/depth_samples.csv` (when blur/depth debug CSV export is enabled)
- `debug/blur_fail_frames/*.jpg` (when evidence export is enabled or blur WARN auto-export triggers)
- `debug/blur_pass_frames/*.jpg` (when evidence export is enabled or blur WARN auto-export triggers)
- `debug/blur_fail_frames_annotated/*.jpg` (when `debug.write_annotated_evidence=true`)
- `debug/blur_pass_frames_annotated/*.jpg` (when `debug.write_annotated_evidence=true`)
- `debug/clean_segments.json` (WARN-strict clean segments)
- `debug/clean_segments_nosync.json` (counterfactual clean segments with sync forced-good)
- `debug/evidence_manifest.json` (when `debug.write_evidence_manifest=true`)
- `debug/benchmarks.json` (when `--bench` or `debug.benchmarks_enabled=true`)

All artifact paths stored in `report.json` are output-directory-relative POSIX paths (for example `plots/sync_histogram.png`).

Optional additive diagnostics:

- `EgoLogQA analyze --bench ...` writes benchmark timings to `debug/benchmarks.json`.
- `debug.write_evidence_manifest` writes a deterministic evidence manifest.
- `debug.write_annotated_evidence` writes annotated evidence copies without modifying raw evidence frames.

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

Sync diagnostics:

- signed sync delta is `depth_time - rgb_time`
- positive signed offset means depth is later than RGB
- when stable offset is detected, action can be `FIX_TIME_ALIGNMENT`

Integrity segments remain available for context. FAIL/no-clean decisions are based on WARN-strict clean segments.

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
  - browser upload (`st.file_uploader`) for one `.mcap` file
  - uploaded filename + size preview
  - tab-specific Analyze button
- Progress + full same-page results after each run

No dataset/revision/prefix/token/cache controls are shown by default.
`Analyze` is disabled until a valid source is selected (HF dropdown choice or Local upload).
Local disk mode stages the uploaded file under the run directory before analysis.

Defaults:

- HF dataset id: `MicroAGI-Labs/MicroAGI00`
- HF revision: `main`
- HF prefix: `raw_mcaps/`
- HF cache dir: `~/.cache/EgoLogQA/hf_mcaps`
- Runs dir: `~/.cache/EgoLogQA/runs`

Env overrides:

- `EGOLOGQA_HF_REPO_ID`
- `EGOLOGQA_HF_REVISION`
- `EGOLOGQA_HF_PREFIX`
- `EGOLOGQA_HF_CACHE_DIR`
- `EGOLOGQA_RUNS_DIR`
- `EGOLOGQA_AI_SUMMARY_ENABLED` (`1` by default, set `0` to disable AI summary)
- `EGOLOGQA_GEMINI_MODEL` (optional override; default is `DEFAULT_GEMINI_MODEL` in `src/egologqa/ai_summary.py`)
- `HF_TOKEN` (optional, not shown in UI)
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` (optional, for Gemini summary)

Developer-only advanced panel:

- Set `EGOLOGQA_UI_ADVANCED=1`
- Enables manual Hugging Face list refresh and error details

### AI Summary (Gemini)

The main results page includes a quick summary with three AI lines plus one deterministic action line per completed run:

- Line 1 (AI): outcome summary (`summary_line`)
- Line 2 (AI): primary-cause explanation (`explanation_line`)
- Line 3 (AI): secondary/unusual signal (`insight_line`)
- Action line (deterministic): gate-action guidance (`action_line`)

Behavior:

- The feature is UI-only and does not modify `report.json`.
- Gemini receives a project brief plus the full analysis `report.json` content for context.
- If API key, SDK, network, or model response fails, the app falls back to deterministic summary, explanation, and insight lines.
- API key precedence is: `GOOGLE_API_KEY`, `GEMINI_API_KEY`, then `st.secrets`.
- Model resolution precedence is: explicit override argument, then `EGOLOGQA_GEMINI_MODEL`, then `DEFAULT_GEMINI_MODEL`.

## Validation Pack

Deterministic validation tooling lives under `validation/`:

- `validation/run_validation.py`
- `validation/summarize_results.py`
- `validation/labels_template.csv`

Streamlit log noise suppression:

- `.streamlit/config.toml` sets logger level to `error`
- `.streamlit/config.toml` sets `server.maxUploadSize = 500` (MB)
- `.streamlit/secrets.toml` must stay local/untracked; configure production secrets in Streamlit Community Cloud -> App Settings -> Secrets
- Fallback launch flag if needed: `--logger.level=error`

Deployment note:

- Streamlit Community Cloud is the supported free hosting target for this app.
- Vercel is not a target host for this Streamlit runtime.

### Troubleshooting: Deployed app shows no images

When the hosted UI renders no previews/evidence images, verify decode health first:

- Check `metrics.rgb_decode_success_count` in `report.json` (must be greater than `0` for image artifacts).
- Check `errors[]` contexts for decode warnings (`RGB_DECODE_FAIL`, `DEPTH_PNG_IMDECODE_FAIL`, `BLUR_UNAVAILABLE_NO_DECODE`) and inspect `cv2_available` / `cv2_import_error`.
- Use Python `3.11` on Streamlit Community Cloud and redeploy after the `opencv-python-headless` dependency switch.
