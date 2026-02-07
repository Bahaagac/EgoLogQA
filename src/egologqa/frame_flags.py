from __future__ import annotations

from dataclasses import dataclass

from egologqa.drop_regions import DropRegions


@dataclass
class FrameFlags:
    frame_ok: list[bool]
    frame_ok_integrity: list[bool]
    frame_ok_vision: list[bool]
    sync_ok: list[bool]
    rgb_drop_ok: list[bool]
    imu_ok: list[bool]
    blur_ok: list[bool]
    exposure_ok: list[bool]
    depth_ok: list[bool]


def build_frame_flags(
    sampled_rgb_times_ms: list[float],
    sampled_rgb_indices: list[int],
    sync_deltas_ms: list[float] | None,
    sync_warn_ms: float,
    drop_regions: DropRegions,
    imu_accel_coverage: list[bool] | None,
    imu_gyro_coverage: list[bool] | None,
    blur_ok: list[bool] | None,
    exposure_ok: list[bool] | None,
    depth_ok: list[bool] | None,
    rgb_pixels_supported: bool,
    depth_pixels_supported: bool,
    depth_timestamps_present: bool,
    imu_exists: bool,
    forced_bad_sample_positions: set[int] | None = None,
) -> FrameFlags:
    forced = forced_bad_sample_positions or set()
    n = len(sampled_rgb_times_ms)
    sync: list[bool] = []
    rgb_drop: list[bool] = []
    imu: list[bool] = []
    blur_flags: list[bool] = []
    exposure_flags: list[bool] = []
    depth_flags: list[bool] = []
    frame_ok_integrity: list[bool] = []
    frame_ok_vision: list[bool] = []

    for pos, t_ms in enumerate(sampled_rgb_times_ms):
        if depth_timestamps_present and sync_deltas_ms is not None:
            sync_ok = sync_deltas_ms[pos] <= sync_warn_ms
        else:
            sync_ok = False
        drop_ok = not drop_regions.contains(t_ms)
        if imu_exists and imu_accel_coverage is not None and imu_gyro_coverage is not None:
            imu_ok = imu_accel_coverage[pos] and imu_gyro_coverage[pos]
        else:
            imu_ok = True

        blur_i = True if blur_ok is None or pos >= len(blur_ok) else blur_ok[pos]
        if not rgb_pixels_supported:
            exposure_i = True
        else:
            exposure_i = (
                True if exposure_ok is None or pos >= len(exposure_ok) else exposure_ok[pos]
            )
        if not depth_pixels_supported:
            depth_i = True
        else:
            depth_i = True if depth_ok is None or pos >= len(depth_ok) else depth_ok[pos]

        integrity_ok = sync_ok and drop_ok and imu_ok
        vision_ok = integrity_ok and blur_i and exposure_i and depth_i
        if pos in forced:
            integrity_ok = False
            vision_ok = False
        sync.append(sync_ok)
        rgb_drop.append(drop_ok)
        imu.append(imu_ok)
        blur_flags.append(blur_i)
        exposure_flags.append(exposure_i)
        depth_flags.append(depth_i)
        frame_ok_integrity.append(integrity_ok)
        frame_ok_vision.append(vision_ok)

    return FrameFlags(
        frame_ok=frame_ok_integrity,
        frame_ok_integrity=frame_ok_integrity,
        frame_ok_vision=frame_ok_vision,
        sync_ok=sync,
        rgb_drop_ok=rgb_drop,
        imu_ok=imu,
        blur_ok=blur_flags,
        exposure_ok=exposure_flags,
        depth_ok=depth_flags,
    )
