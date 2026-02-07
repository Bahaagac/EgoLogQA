from __future__ import annotations

import numpy as np


def sample_rgb_indices(total_frames: int, stride: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(0, total_frames, stride))

    target = max_frames
    raw = np.rint(np.linspace(0, total_frames - 1, target)).astype(np.int64)
    sampled: list[int] = []
    seen: set[int] = set()
    for idx in raw.tolist():
        if idx not in seen:
            seen.add(idx)
            sampled.append(idx)
    if len(sampled) < target:
        for idx in range(total_frames):
            if idx in seen:
                continue
            seen.add(idx)
            sampled.append(idx)
            if len(sampled) >= target:
                break
    sampled.sort()
    return sampled
