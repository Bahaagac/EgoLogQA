from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from egologqa.artifacts import write_rgb_previews
from egologqa.io.reader import InMemoryMessageSource
from egologqa.models import MessageRecord, QAConfig
from egologqa.pipeline import analyze_file
from tests.conftest import make_message_from_ns


def _cfg() -> QAConfig:
    cfg = QAConfig()
    cfg.topics.mode = "explicit"
    cfg.topics.rgb_topic = "/rgb"
    cfg.topics.depth_topic = "/depth"
    cfg.topics.imu_accel_topic = "/imu"
    cfg.topics.imu_gyro_topic = "/imu"
    cfg.sampling.rgb_stride = 1
    cfg.sampling.max_rgb_frames = 100
    cfg.debug.export_preview_frames = True
    return cfg


def _records() -> list[MessageRecord]:
    out: list[MessageRecord] = []
    base = 1_000_000_000
    for i in range(6):
        t = base + i * 50_000_000
        out.append(MessageRecord("/rgb", t, t, make_message_from_ns(t, data=b"rgb")))
        out.append(MessageRecord("/depth", t, t, make_message_from_ns(t, data=b"depth")))
        out.append(MessageRecord("/imu", t, t, make_message_from_ns(t)))
    return out


def test_preview_relpaths_match_preview_count_and_are_sorted() -> None:
    cfg = _cfg()
    source = InMemoryMessageSource(_records())

    def fake_decode_rgb(_msg):
        return np.full((16, 16, 3), 127, dtype=np.uint8), None

    def fake_decode_depth(_msg):
        return np.full((16, 16), 100, dtype=np.uint16), None

    with TemporaryDirectory() as d, patch(
        "egologqa.pipeline.decode_rgb_message", side_effect=fake_decode_rgb
    ), patch("egologqa.pipeline.decode_depth_message", side_effect=fake_decode_depth):
        result = analyze_file("dummy.mcap", Path(d), cfg, source=source)

    preview_relpaths = result.report["metrics"]["preview_relpaths"]
    preview_count = result.report["metrics"]["preview_count"]
    assert isinstance(preview_relpaths, list)
    assert preview_count == len(preview_relpaths)
    assert preview_relpaths == sorted(preview_relpaths)
    assert all(not Path(path).is_absolute() for path in preview_relpaths)


def test_write_rgb_previews_skips_excluded_positions_when_possible() -> None:
    frames = {i: np.zeros((4, 4, 3), dtype=np.uint8) for i in range(8)}

    class _FakeCV2:
        @staticmethod
        def imwrite(path: str, _frame: np.ndarray) -> bool:
            Path(path).write_bytes(b"ok")
            return True

    with TemporaryDirectory() as d, patch.dict("sys.modules", {"cv2": _FakeCV2}):
        out = write_rgb_previews(
            frames_by_pos=frames,
            output_dir=Path(d),
            max_previews=6,
            exclude_positions={0, 1, 2},
        )

    selected_positions = [
        int(Path(path).stem.split("_sample_")[-1])
        for path in out
        if "_sample_" in Path(path).stem
    ]
    assert selected_positions[:5] == [3, 4, 5, 6, 7]
    assert len(selected_positions) == 6
