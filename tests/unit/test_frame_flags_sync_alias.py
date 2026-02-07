from __future__ import annotations

from egologqa.drop_regions import DropRegions
from egologqa.frame_flags import build_frame_flags


def test_sync_ok_alias_matches_sync_ok_fail() -> None:
    flags = build_frame_flags(
        sampled_rgb_times_ms=[1000.0, 1033.3, 1066.6],
        sampled_rgb_indices=[0, 1, 2],
        sync_deltas_ms=[10.0, 20.0, 30.0],
        sync_fail_ms=33.0,
        sync_warn_ms=16.0,
        sync_available_globally=True,
        drop_regions=DropRegions([]),
        imu_accel_coverage=[True, True, True],
        imu_gyro_coverage=[True, True, True],
        blur_ok=[True, True, True],
        exposure_ok=[True, True, True],
        depth_ok=[True, True, True],
        rgb_pixels_supported=True,
        depth_pixels_supported=True,
        depth_timestamps_present=True,
        imu_exists=True,
    )

    assert flags.sync_ok == flags.sync_ok_fail
    assert flags.sync_ok_warn == [True, False, False]
