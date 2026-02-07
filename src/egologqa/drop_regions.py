from __future__ import annotations

from bisect import bisect_left


class DropRegions:
    """Represents bad intervals using (left, right] boundary convention."""

    def __init__(self, intervals_ms: list[tuple[float, float]]):
        self.intervals_ms = sorted(intervals_ms, key=lambda item: item[0])
        self._starts = [s for s, _ in self.intervals_ms]
        self._ends = [e for _, e in self.intervals_ms]

    def contains(self, t_ms: float) -> bool:
        if not self.intervals_ms:
            return False
        idx = bisect_left(self._starts, t_ms)
        candidates = []
        if idx < len(self.intervals_ms):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        for cidx in candidates:
            left, right = self.intervals_ms[cidx]
            if t_ms > left and t_ms <= right:
                return True
        return False
