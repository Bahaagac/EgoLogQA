from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from egologqa.config import load_config
from egologqa.pipeline import analyze_file


st.set_page_config(page_title="EgoLogQA", layout="wide")
st.title("EgoLogQA")
st.caption("MicroAGI00 ROS2 MCAP quality gate")

mode = st.sidebar.selectbox(
    "Mode",
    ["Analyze MCAP", "Open Output Directory"],
)
config_path = st.sidebar.text_input("Config", value="configs/microagi00_ros2.yaml")
output_dir_input = st.sidebar.text_input("Output directory", value="out/streamlit")

debug_export_exposure_csv = st.sidebar.toggle("Export exposure CSV", value=True)
debug_export_blur_csv = st.sidebar.toggle("Export blur/depth CSV", value=True)
debug_export_evidence_frames = st.sidebar.toggle("Export blur evidence frames", value=False)
debug_export_evidence_on_warn = st.sidebar.toggle("Auto-export evidence on blur WARN", value=True)
debug_evidence_frames_k = int(
    st.sidebar.number_input("Evidence frames K", min_value=1, max_value=64, value=16, step=1)
)
debug_export_preview_frames = st.sidebar.toggle("Export preview frames", value=True)

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


def _artifact_path(output_dir: Path, rel: str | None) -> Path | None:
    if not rel:
        return None
    return output_dir / rel


def _render_artifact_line(label: str, output_dir: Path, rel: str | None) -> None:
    path = _artifact_path(output_dir, rel)
    if rel and path is not None and path.exists():
        st.caption(f"{label}: `{rel}`")
    elif rel:
        st.caption(f"{label}: `{rel}` (artifact not found)")
    else:
        st.caption(f"{label}: unavailable")


def _render_report(report: dict[str, Any], output_dir: Path) -> None:
    gate = report.get("gate", {}).get("gate")
    if gate == "PASS":
        st.success(f"Gate: {gate}")
    elif gate == "WARN":
        st.warning(f"Gate: {gate}")
    else:
        st.error(f"Gate: {gate}")

    metrics = report.get("metrics", {})
    summary_box.json(
        {
            "gate": report.get("gate", {}),
            "streams": report.get("streams", {}),
            "metrics": {
                "integrity_ok_ratio": metrics.get("integrity_ok_ratio"),
                "integrity_coverage_seconds_est": metrics.get("integrity_coverage_seconds_est"),
                "vision_ok_ratio": metrics.get("vision_ok_ratio"),
                "vision_coverage_seconds_est": metrics.get("vision_coverage_seconds_est"),
                "sync_p95_ms": metrics.get("sync_p95_ms"),
                "drop_ratio": metrics.get("drop_ratio"),
                "blur_fail_ratio": metrics.get("blur_fail_ratio"),
                "blur_threshold": metrics.get("blur_threshold"),
                "exposure_bad_ratio": metrics.get("exposure_bad_ratio"),
                "exposure_bad_reason_counts": metrics.get("exposure_bad_reason_counts"),
                "depth_invalid_mean": metrics.get("depth_invalid_mean"),
                "rgb_decode_attempt_count": metrics.get("rgb_decode_attempt_count"),
                "rgb_decode_success_count": metrics.get("rgb_decode_success_count"),
                "depth_decode_attempt_count": metrics.get("depth_decode_attempt_count"),
                "depth_decode_success_count": metrics.get("depth_decode_success_count"),
            },
        }
    )

    st.subheader("Integrity Segments")
    st.dataframe(report.get("segments", []), use_container_width=True)

    st.subheader("Channel Summary")
    st.dataframe(
        [
            {"topic": topic, **stats}
            for topic, stats in report.get("streams", {}).get("topic_stats", {}).items()
        ],
        use_container_width=True,
    )

    hist_rel = metrics.get("sync_histogram_path")
    hist_path = _artifact_path(output_dir, hist_rel)
    if hist_path is not None and hist_path.exists():
        st.subheader("Sync Histogram")
        st.image(str(hist_path), use_container_width=True)
    elif hist_rel:
        st.info(f"Sync histogram artifact missing: `{hist_rel}`")

    drop_rel = metrics.get("drop_timeline_path")
    drop_path = _artifact_path(output_dir, drop_rel)
    if drop_path is not None and drop_path.exists():
        st.subheader("Drop Timeline")
        st.image(str(drop_path), use_container_width=True)
    elif drop_rel:
        st.info(f"Drop timeline artifact missing: `{drop_rel}`")

    st.subheader("Vision Evidence")
    _render_artifact_line("Exposure CSV", output_dir, metrics.get("exposure_debug_csv_path"))
    _render_artifact_line("Blur CSV", output_dir, metrics.get("blur_debug_csv_path"))
    _render_artifact_line("Depth CSV", output_dir, metrics.get("depth_debug_csv_path"))

    fail_dir_rel = metrics.get("blur_fail_frames_dir")
    pass_dir_rel = metrics.get("blur_pass_frames_dir")
    _render_artifact_line("Blur fail frames", output_dir, fail_dir_rel)
    _render_artifact_line("Blur pass frames", output_dir, pass_dir_rel)

    fail_dir = _artifact_path(output_dir, fail_dir_rel)
    pass_dir = _artifact_path(output_dir, pass_dir_rel)
    if fail_dir is not None and fail_dir.exists():
        fail_files = sorted(fail_dir.glob("*.jpg"))[:8]
        if fail_files:
            st.caption("Blur fail evidence")
            cols = st.columns(4)
            for i, img in enumerate(fail_files):
                cols[i % 4].image(str(img), caption=img.name, use_container_width=True)
    if pass_dir is not None and pass_dir.exists():
        pass_files = sorted(pass_dir.glob("*.jpg"))[:8]
        if pass_files:
            st.caption("Blur pass evidence")
            cols = st.columns(4)
            for i, img in enumerate(pass_files):
                cols[i % 4].image(str(img), caption=img.name, use_container_width=True)

    preview_dir = output_dir / "previews"
    if preview_dir.exists():
        preview_files = sorted(preview_dir.glob("*.png"))[:8]
        if preview_files:
            st.subheader("RGB Previews")
            cols = st.columns(4)
            for i, img in enumerate(preview_files):
                cols[i % 4].image(str(img), caption=img.name, use_container_width=True)

    st.subheader("Errors")
    st.dataframe(report.get("errors", []), use_container_width=True)


if mode == "Analyze MCAP":
    uploaded = st.file_uploader("MCAP file", type=["mcap"])
    run = st.button("Analyze", type="primary", disabled=uploaded is None)

    if run and uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mcap") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            cfg = load_config(config_path)
            cfg.debug.export_exposure_csv = debug_export_exposure_csv
            cfg.debug.export_blur_csv = debug_export_blur_csv
            cfg.debug.export_evidence_frames = debug_export_evidence_frames
            cfg.debug.export_evidence_on_warn = debug_export_evidence_on_warn
            cfg.debug.evidence_frames_k = debug_evidence_frames_k
            cfg.debug.export_preview_frames = debug_export_preview_frames

            out_dir = Path(output_dir_input)
            result = analyze_file(
                input_path=tmp_path,
                output_dir=out_dir,
                config=cfg,
                progress_cb=_on_progress,
            )
            _render_report(result.report, out_dir)
            st.caption(f"Report written to {result.output_path}")
            status_box.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label="Failed", state="error", expanded=True)
            st.exception(exc)
else:
    st.subheader("Open Existing Output Directory")
    load = st.button("Load", type="primary")
    if load:
        out_dir = Path(output_dir_input)
        report_path = out_dir / "report.json"
        if not report_path.exists():
            st.error(f"report.json not found at `{report_path}`")
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                _render_report(report, out_dir)
                status_box.update(label="Loaded", state="complete", expanded=False)
            except Exception as exc:
                status_box.update(label="Failed", state="error", expanded=True)
                st.exception(exc)
