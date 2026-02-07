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
streamlit run app/streamlit_app.py
```

The UI runs analysis in synchronous chunks and updates progress continuously.
