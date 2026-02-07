from __future__ import annotations

from egologqa.drop_regions import DropRegions


def test_drop_region_boundary_is_left_open_right_closed() -> None:
    regions = DropRegions([(100.0, 200.0)])
    assert regions.contains(100.0) is False
    assert regions.contains(150.0) is True
    assert regions.contains(200.0) is True
    assert regions.contains(200.1) is False
