from __future__ import annotations

from pathlib import Path

import pytest

from egologqa.kiosk_helpers import (
    allocate_run_dir,
    build_hf_display_label,
    build_timestamped_run_basename,
    build_run_basename,
    ensure_writable_dir,
    human_bytes,
    make_local_option_label,
    map_error_bucket,
    resolve_runs_base_dir,
    resolve_source_kind,
)


def test_resolve_runs_base_dir_default_and_override() -> None:
    default_dir = resolve_runs_base_dir(None)
    assert str(default_dir).endswith(".cache/EgoLogQA/runs")
    override = resolve_runs_base_dir("~/custom_runs")
    assert override == Path("~/custom_runs").expanduser()


def test_build_run_basename_is_stable() -> None:
    one = build_run_basename(
        repo_id="MicroAGI-Labs/MicroAGI00",
        revision="main",
        hf_path="raw_mcaps/Bakery_Food_Preparation_15f719ff.mcap",
    )
    two = build_run_basename(
        repo_id="MicroAGI-Labs/MicroAGI00",
        revision="main",
        hf_path="raw_mcaps/Bakery_Food_Preparation_15f719ff.mcap",
    )
    three = build_run_basename(
        repo_id="MicroAGI-Labs/MicroAGI00",
        revision="main",
        hf_path="raw_mcaps/AFTER_MEAL_CLEANUP_3bcd1460.mcap",
    )
    assert one == two
    assert one != three
    assert one.startswith("Bakery_Food_Preparation_15f719ff_main_")


def test_allocate_run_dir_adds_suffix_on_collision(tmp_path: Path) -> None:
    first = allocate_run_dir(tmp_path, "demo_run")
    second = allocate_run_dir(tmp_path, "demo_run")
    third = allocate_run_dir(tmp_path, "demo_run")
    assert first.name == "demo_run"
    assert second.name == "demo_run_2"
    assert third.name == "demo_run_3"


def test_map_error_bucket() -> None:
    assert "huggingface_hub" in map_error_bucket(RuntimeError("No module named 'huggingface_hub'"))
    assert "Dataset requires auth" in map_error_bucket(RuntimeError("Authentication failed (HTTP 401)"))
    assert "Dataset/revision/path not found" in map_error_bucket(RuntimeError("HTTP 404"))
    assert "Cannot reach huggingface.co" in map_error_bucket(RuntimeError("Failed to resolve 'huggingface.co'"))


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (None, "unknown"),
        (-1, "unknown"),
        (43, "43 B"),
        (43 * 1024, "43 KB"),
        (812_300_000, "774.7 MB"),
        (5_368_709_120, "5.0 GB"),
    ],
)
def test_human_bytes(size_bytes: int | None, expected: str) -> None:
    assert human_bytes(size_bytes) == expected


def test_build_hf_display_label_strips_prefix_and_formats_size() -> None:
    label = build_hf_display_label("raw_mcaps/AFTER_MEAL_CLEANUP_3bcd1460.mcap", "raw_mcaps/", 1024**3)
    assert label == "AFTER_MEAL_CLEANUP_3bcd1460.mcap (1.0 GB)"
    unknown = build_hf_display_label("raw_mcaps/clip.mcap", "raw_mcaps/", None)
    assert unknown == "clip.mcap (unknown)"


def test_make_local_option_label() -> None:
    assert make_local_option_label("clip.mcap", 1024) == "clip.mcap (1 KB)"
    assert make_local_option_label("clip.mcap", None) == "clip.mcap (unknown)"


def test_build_timestamped_run_basename_with_suffix() -> None:
    name = build_timestamped_run_basename("Bakery_Food_Preparation_15f719ff.mcap", suffix="ab12")
    assert name.endswith("_ab12")
    assert "Bakery_Food_Preparation_15f719ff_" in name


def test_resolve_source_kind_precedence() -> None:
    assert resolve_source_kind(local_uploaded=True, hf_selected=True) == "local"
    assert resolve_source_kind(local_uploaded=False, hf_selected=True) == "hf"
    assert resolve_source_kind(local_uploaded=False, hf_selected=False) is None


def test_ensure_writable_dir(tmp_path: Path) -> None:
    out = ensure_writable_dir(tmp_path / "nested", "runs")
    assert out.exists()
