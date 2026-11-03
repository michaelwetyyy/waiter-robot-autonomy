"""Sensor-fusion primitives: geometry + semantics, bound by SLAM pose.

The companion doc is [[Sensor Fusion]]. This module is the actual API the
decision layer consumes; everything here is a pure function so the unit
tests can hit it without any of the live perception models running.

Three groups:

  1. Geometry (LiDAR): `lidar_emergency` -- 360 degree immediate-stop rule.
  2. Binding (the glue): `pixel_to_world_at_z` -- takes any camera pixel,
     the SLAM pose, and a known world-Z (e.g. the table plane or cup
     centre height) and returns the world (x, y) of the thing in that
     pixel. This is how a hand-gesture pixel becomes "table 3" and how
     a YOLO bbox becomes a position to avoid.
  3. Intent (camera semantics): `table_summoning`, `confident_collect` --
     drop-in functions that operate on whatever the gesture / detection
     pipeline produces. They're built for MediaPipe Hands + ArUco +
     YOLO inputs but only require duck-typed dicts, so they can be
     exercised before those models are wired in.

What's NOT wired into `decision/state_machine.py` yet (deliberately,
pending sim validation): the LiDAR emergency stop. The existing
AVOID + DWA collision logic already covers obstacles via the inflated
grid. Adding `lidar_emergency` as a parallel 360° immediate-stop layer
is sound (it catches things the forward camera and the planner's
inflation can't), but introducing it without a Webots run risks
fighting the existing flow. The function is ready to drop into
`_run_perception` once the sim confirms behaviour.
"""

from __future__ import annotations
import math
from typing import Any, Iterable, Optional, Tuple

import numpy as np


# 1. Geometry: LiDAR 360-degree emergency stop ---------------------------
def lidar_emergency(scan, stop_radius_m: float = 0.30,
                    ignore_within_m: float = 0.05) -> bool:
    """True if any LiDAR return is inside `stop_radius_m` in any direction.

    Returns within `ignore_within_m` are treated as self-reflections (the
    LiDAR's own housing, the robot body, or carried cups too close to the
    scan plane) and ignored, which keeps the rule from false-firing on the
    robot itself. Tune `ignore_within_m` to just clear the chassis radius.

    Wraps the camera-only emergency stop with a 360° check; the camera
    sees ~62° forward, the LiDAR sees the rest. The decision layer ORs
    the two: `lidar_emergency(scan) or camera_emergency_stop(...)`.
    """
    if scan is None:
        return False
    r = np.asarray(scan.ranges, dtype=np.float64)
    finite = np.isfinite(r) & (r > ignore_within_m)
    return bool((r[finite] < stop_radius_m).any())


# 2. Binding: pixel -> world (the glue between camera and LiDAR map) -----
def pixel_to_world_at_z(
    px: float, py: float,
    intrinsics: Any,
    bot_pose: Tuple[float, float, float],
    world_z: float,
) -> Optional[Tuple[float, float]]:
    """Back-project an image pixel onto a world plane at `world_z`.

    `intrinsics` is `CONFIG.camera` (CameraIntrinsics with .K, .dist, .tilt,
    .mount_x, .mount_y, .mount_z). `bot_pose` is the SLAM-corrected
    (x, y, yaw) in the world. Returns the (wx, wy) where the camera ray
    through (px, py) intersects the horizontal plane at `world_z`, or
    None if the ray is parallel to / above the plane.

    Use cases:
      - gesture pixel -> world XY at table-top height -> nearest table marker
      - YOLO bbox bottom -> world XY at floor height -> planner avoidance pos

    Math: ray in camera frame from K^-1 @ [px, py, 1]; rotate by camera
    tilt + bot yaw into the world frame; offset by bot pose + camera mount;
    solve t such that the ray hits z = world_z.
    """
    K = np.asarray(intrinsics.K, dtype=np.float64)
    bx, by, byaw = bot_pose
    mx, my, mz = (float(intrinsics.mount_x),
                  float(intrinsics.mount_y),
                  float(intrinsics.mount_z))
    tilt = float(intrinsics.tilt)

    # Camera-frame ray (camera looks down +Z conventionally; we'll use the
    # OpenCV convention: x right, y down, z forward).
    ray_cam = np.linalg.solve(K, np.array([float(px), float(py), 1.0]))

    # Rotate ray: camera pitch (tilt) about the body X axis, then yaw the
    # body into the world. Pitch maps (xc, yc, zc) -> (xc, yc*cos - zc*sin,
    # yc*sin + zc*cos); body x = z_cam, body y = -x_cam, body z = -y_cam
    # for a typical forward-pointing-camera mount.
    xc, yc, zc = ray_cam
    ct, st = math.cos(tilt), math.sin(tilt)
    # camera -> body (rotate about body X by tilt)
    bx_dir = zc * ct + yc * st
    by_dir = -xc
    bz_dir = -yc * ct + zc * st
    # body -> world (yaw about Z)
    cy, sy = math.cos(byaw), math.sin(byaw)
    wx_dir = cy * bx_dir - sy * by_dir
    wy_dir = sy * bx_dir + cy * by_dir
    wz_dir = bz_dir

    # Ray origin in world: bot pose + camera mount (camera mount is in body
    # frame, rotate by yaw too).
    ox = bx + cy * mx - sy * my
    oy = by + sy * mx + cy * my
    oz = mz

    if abs(wz_dir) < 1e-9:
        return None
    t = (world_z - oz) / wz_dir
    if t <= 0:                          # plane behind the camera
        return None
    return (ox + t * wx_dir, oy + t * wy_dir)


# 3. Intent: which table is summoning? -----------------------------------
def table_summoning(
    gesture: Optional[Any],
    visible_markers: Iterable[Any],
    table_from_id: dict[int, Any],
    max_norm_dist: float = 0.30,
    vertical_tol: float = 0.20,
) -> Optional[Any]:
    """Pair a hand-down gesture event with the nearest visible table marker.

    `gesture` has `.image_position = (x, y)` in NORMALISED image coords
    [0, 1]; pass None when no gesture fired.
    `visible_markers` are duck-typed objects with `.id`, `.image_x`,
    `.image_y` (also normalised).
    Returns the table id, or None when no marker is unambiguously closest.

    Skip markers that are vertically far from the hand (different image
    row -- they can't be the diner's table even if x-close. Threshold
    `max_norm_dist` controls "unambiguous": a gesture that's not clearly
    near a marker is discarded, not guessed.
    """
    if gesture is None:
        return None
    hx, hy = gesture.image_position
    best, best_d = None, float("inf")
    for m in visible_markers:
        if abs(m.image_y - hy) > vertical_tol:
            continue
        d = math.hypot(m.image_x - hx, m.image_y - hy)
        if d < best_d:
            best, best_d = m, d
    if best is not None and best_d < max_norm_dist:
        return table_from_id.get(best.id)
    return None


def confident_collect(
    gesture: Optional[Any],
    detections: Iterable[Any],
    visible_markers: Iterable[Any],
    table_from_id: dict[int, Any],
    near_norm: float = 0.15,
) -> Optional[Any]:
    """Higher-confidence variant: only fire when the table marker AND a
    `glass_empty` AND a `person_seated` all sit near the gesture pixel.

    Reduces false fires (someone reaching across an empty table). Skip for
    the bare demo (`table_summoning` alone) if MediaPipe + ArUco are
    already reliable in the test set.
    """
    table = table_summoning(gesture, visible_markers, table_from_id)
    if table is None:
        return None
    hx, hy = gesture.image_position
    def _near(det) -> bool:
        cx, cy = 0.5 * (det.bbox[0] + det.bbox[2]), 0.5 * (det.bbox[1] + det.bbox[3])
        return math.hypot(cx - hx, cy - hy) < near_norm
    has_empty  = any(d.cls == "glass_empty"   and _near(d) for d in detections)
    has_seated = any(d.cls == "person_seated" and _near(d) for d in detections)
    return table if (has_empty and has_seated) else None
