from __future__ import annotations

from egologqa.sampling import sample_rgb_indices


def test_sampling_small_uses_stride() -> None:
    out = sample_rgb_indices(total_frames=12, stride=5, max_frames=12000)
    assert out == [0, 5, 10]


def test_sampling_large_is_deterministic_and_full_target() -> None:
    out1 = sample_rgb_indices(total_frames=20_000, stride=5, max_frames=1_000)
    out2 = sample_rgb_indices(total_frames=20_000, stride=5, max_frames=1_000)
    assert out1 == out2
    assert len(out1) == 1_000
    assert out1 == sorted(out1)
    assert len(set(out1)) == 1_000
