from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from egologqa.config import load_config
from egologqa.pipeline import analyze_file


st.set_page_config(page_title="EgoLogQA", layout="wide")
st.title("EgoLogQA")
st.caption("MicroAGI00 ROS2 MCAP quality gate")

uploaded = st.file_uploader("MCAP file", type=["mcap"])
config_path = st.text_input("Config", value="configs/microagi00_ros2.yaml")
output_dir = st.text_input("Output directory", value="out/streamlit")

run = st.button("Analyze", type="primary", disabled=uploaded is None)

progress = st.progress(0)
status_box = st.status("Idle", expanded=False)
summary_box = st.empty()
detail_box = st.empty()


def _on_progress(event: dict) -> None:
    progress.progress(min(max(float(event.get("progress", 0.0)), 0.0), 1.0))
    status_box.update(label=event.get("message", "..."), state="running", expanded=True)
    partial = event.get("partial")
    if partial:
        detail_box.json(partial)


if run and uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mcap") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        cfg = load_config(config_path)
        result = analyze_file(
            input_path=tmp_path,
            output_dir=output_dir,
            config=cfg,
            progress_cb=_on_progress,
        )
        gate = result.report["gate"]["gate"]
        if gate == "PASS":
            st.success(f"Gate: {gate}")
        elif gate == "WARN":
            st.warning(f"Gate: {gate}")
        else:
            st.error(f"Gate: {gate}")

        summary_box.json(
            {
                "gate": result.report["gate"],
                "streams": result.report["streams"],
                "metrics": {
                    "integrity_ok_ratio": result.report["metrics"]["integrity_ok_ratio"],
                    "integrity_coverage_seconds_est": result.report["metrics"][
                        "integrity_coverage_seconds_est"
                    ],
                    "vision_ok_ratio": result.report["metrics"]["vision_ok_ratio"],
                    "vision_coverage_seconds_est": result.report["metrics"][
                        "vision_coverage_seconds_est"
                    ],
                    "exposure_bad_ratio": result.report["metrics"]["exposure_bad_ratio"],
                    "exposure_bad_reason_counts": result.report["metrics"][
                        "exposure_bad_reason_counts"
                    ],
                    "sync_p95_ms": result.report["metrics"]["sync_p95_ms"],
                    "drop_ratio": result.report["metrics"]["drop_ratio"],
                    "imu_combined_missing_ratio": result.report["metrics"][
                        "imu_combined_missing_ratio"
                    ],
                    "depth_invalid_mean": result.report["metrics"]["depth_invalid_mean"],
                },
            }
        )
        if result.report["metrics"].get("exposure_debug_csv_path"):
            st.caption(
                "Exposure debug CSV: "
                f"{result.report['metrics']['exposure_debug_csv_path']}"
            )

        if result.report["metrics"]["depth_invalid_mean"] is None:
            depth_error_code = None
            for err in result.report.get("errors", []):
                if str(err.get("code", "")).startswith("DEPTH_"):
                    depth_error_code = err.get("code")
                    break
            st.info(
                "Depth pixel metrics unavailable "
                f"(status={result.report['streams']['decode_status']['depth_pixels']}, "
                f"code={depth_error_code or 'N/A'})."
            )

        st.subheader("Integrity Segments")
        st.dataframe(result.report["segments"], use_container_width=True)
        st.subheader("Channel Summary")
        st.dataframe(
            [
                {"topic": topic, **stats}
                for topic, stats in result.report["streams"].get("topic_stats", {}).items()
            ],
            use_container_width=True,
        )
        hist_path = result.report["metrics"].get("sync_histogram_path")
        if hist_path and Path(hist_path).exists():
            st.subheader("Sync Histogram")
            st.image(str(hist_path), use_container_width=True)
        drop_path = result.report["metrics"].get("drop_timeline_path")
        if drop_path and Path(drop_path).exists():
            st.subheader("Drop Timeline")
            st.image(str(drop_path), use_container_width=True)

        preview_dir = Path(output_dir) / "previews"
        if preview_dir.exists():
            preview_files = sorted(preview_dir.glob("*.png"))[:8]
            if preview_files:
                st.subheader("RGB Previews")
                cols = st.columns(4)
                for i, img in enumerate(preview_files):
                    cols[i % 4].image(str(img), caption=img.name, use_container_width=True)

        st.subheader("Errors")
        st.dataframe(result.report["errors"], use_container_width=True)
        st.caption(f"Report written to {result.output_path}")
        status_box.update(label="Done", state="complete", expanded=False)
    except Exception as exc:
        status_box.update(label="Failed", state="error", expanded=True)
        st.exception(exc)
