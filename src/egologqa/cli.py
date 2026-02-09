from __future__ import annotations

import argparse
import json
from pathlib import Path

from egologqa.config import load_config
from egologqa.models import TopicOverrides
from egologqa.pipeline import analyze_file
from egologqa.report import empty_report, git_commit_or_unknown, write_report_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egologqa")
    sub = parser.add_subparsers(dest="command")
    analyze = sub.add_parser("analyze", help="Analyze an MCAP file.")
    analyze.add_argument("--input", required=True, help="Path to MCAP file.")
    analyze.add_argument("--output", required=True, help="Output directory.")
    analyze.add_argument("--config", required=False, default=None, help="YAML config path.")
    analyze.add_argument("--rgb-topic", default=None)
    analyze.add_argument("--depth-topic", default=None)
    analyze.add_argument("--imu-accel-topic", default=None)
    analyze.add_argument("--imu-gyro-topic", default=None)
    analyze.add_argument(
        "--bench",
        action="store_true",
        help="Write debug/benchmarks.json and include benchmarks_path in metrics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "analyze":
        parser.print_help()
        return 0

    output_dir = Path(args.output)
    overrides = TopicOverrides(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        imu_accel_topic=args.imu_accel_topic,
        imu_gyro_topic=args.imu_gyro_topic,
    )
    try:
        config = load_config(args.config)
        if args.bench:
            config.debug.benchmarks_enabled = True
    except Exception as exc:
        report = empty_report(
            input_path=args.input,
            file_size_bytes=Path(args.input).stat().st_size if Path(args.input).exists() else None,
            commit=git_commit_or_unknown(Path.cwd()),
        )
        report["errors"] = [
            {
                "severity": "ERROR",
                "code": "CONFIG_LOAD_ERROR",
                "message": str(exc),
                "context": {"config": args.config},
            }
        ]
        report["gate"]["gate"] = "FAIL"
        report["gate"]["fail_reasons"] = ["FAIL_ANALYSIS_ERROR"]
        path = write_report_json(report, output_dir)
        print(json.dumps({"gate": "FAIL", "report": str(path)}))
        return 30

    result = analyze_file(
        input_path=args.input,
        output_dir=output_dir,
        config=config,
        overrides=overrides,
    )
    print(f"GATE STATUS: {result.gate}")
    print(f"RECOMMENDED ACTION: {result.recommended_action}")
    print(f"FAIL REASONS: {result.fail_reasons}")
    print(f"WARN REASONS: {result.warn_reasons}")
    warn_entries = [err for err in result.errors if err.get("severity") == "WARN"]
    error_entries = [err for err in result.errors if err.get("severity") == "ERROR"]
    print(f"WARN ENTRIES ({len(warn_entries)}):")
    for item in warn_entries:
        print(f"- {item.get('code')}: {item.get('message')}")
    print(f"ERROR ENTRIES ({len(error_entries)}):")
    for item in error_entries:
        print(f"- {item.get('code')}: {item.get('message')}")
    metrics = result.report.get("metrics", {})
    artifact_keys = [
        ("report", str(result.report_path or result.output_path)),
        ("sync_histogram_path", metrics.get("sync_histogram_path")),
        ("drop_timeline_path", metrics.get("drop_timeline_path")),
        ("exposure_debug_csv_path", metrics.get("exposure_debug_csv_path")),
        ("exposure_low_clip_frames_dir", metrics.get("exposure_low_clip_frames_dir")),
        ("exposure_high_clip_frames_dir", metrics.get("exposure_high_clip_frames_dir")),
        ("exposure_flat_and_dark_frames_dir", metrics.get("exposure_flat_and_dark_frames_dir")),
        ("exposure_flat_and_bright_frames_dir", metrics.get("exposure_flat_and_bright_frames_dir")),
        ("exposure_evidence_error_path", metrics.get("exposure_evidence_error_path")),
        ("blur_debug_csv_path", metrics.get("blur_debug_csv_path")),
        ("depth_debug_csv_path", metrics.get("depth_debug_csv_path")),
        ("blur_fail_frames_dir", metrics.get("blur_fail_frames_dir")),
        ("blur_pass_frames_dir", metrics.get("blur_pass_frames_dir")),
        ("blur_fail_frames_annotated_dir", metrics.get("blur_fail_frames_annotated_dir")),
        ("blur_pass_frames_annotated_dir", metrics.get("blur_pass_frames_annotated_dir")),
        ("clean_segments_path", metrics.get("clean_segments_path")),
        ("clean_segments_nosync_path", metrics.get("clean_segments_nosync_path")),
        ("evidence_manifest_path", metrics.get("evidence_manifest_path")),
        ("benchmarks_path", metrics.get("benchmarks_path")),
    ]
    print("ARTIFACTS:")
    for name, value in artifact_keys:
        if value:
            print(f"- {name}: {value}")
    print(
        json.dumps(
            {
                "gate": result.gate,
                "recommended_action": result.recommended_action,
                "report": str(result.report_path or result.output_path),
            }
        )
    )
    if result.gate == "PASS":
        return 0
    if result.gate == "WARN":
        return 10
    if result.gate == "FAIL":
        if any(err.get("code") == "ANALYSIS_EXCEPTION" for err in result.report.get("errors", [])):
            return 30
        return 20
    return 30
