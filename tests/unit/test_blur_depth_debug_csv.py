from __future__ import annotations

from pathlib import Path

from egologqa.artifacts import write_blur_debug_csv, write_depth_debug_csv


def test_blur_debug_csv_schema_and_order(tmp_path: Path) -> None:
    rows = [
        {
            "sample_i": 2,
            "t_ms": 2.0,
            "roi_margin_ratio": 0.05,
            "decode_ok": 1,
            "blur_value": 90.0,
            "blur_threshold": 80.0,
            "blur_ok": 1,
            "decode_error_code": "",
        },
        {
            "sample_i": 0,
            "t_ms": 0.0,
            "roi_margin_ratio": 0.05,
            "decode_ok": 0,
            "blur_value": None,
            "blur_threshold": 80.0,
            "blur_ok": None,
            "decode_error_code": "RGB_DECODE_FAIL",
        },
    ]
    path = write_blur_debug_csv(rows, tmp_path)
    assert path is not None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample_i,t_ms,roi_margin_ratio,decode_ok,blur_value,blur_threshold,blur_ok,decode_error_code"
    assert lines[1].startswith("0,")
    assert lines[2].startswith("2,")


def test_depth_debug_csv_schema_and_order(tmp_path: Path) -> None:
    rows = [
        {
            "sample_i": 3,
            "t_ms": 3.0,
            "decode_ok": 1,
            "invalid_ratio": 0.1,
            "min_depth": 100,
            "max_depth": 2000,
            "dtype": "uint16",
            "error_code": "",
        },
        {
            "sample_i": 1,
            "t_ms": 1.0,
            "decode_ok": 0,
            "invalid_ratio": None,
            "min_depth": None,
            "max_depth": None,
            "dtype": "",
            "error_code": "DEPTH_PNG_IMDECODE_FAIL",
        },
    ]
    path = write_depth_debug_csv(rows, tmp_path)
    assert path is not None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample_i,t_ms,decode_ok,invalid_ratio,min_depth,max_depth,dtype,error_code"
    assert lines[1].startswith("1,")
    assert lines[2].startswith("3,")
