"""LiDAR scan-to-map matching for odometry drift correction.

`ingest_lidar` stamps each scan in the bot's *odometry* pose. Odometry
drifts (wheel slip, IMU bias), so over time walls get stamped a few cm
off their true position and smear into phantoms; the `lidar_walls`
docstring calls these out directly. This module corrects the pose BEFORE
stamping, so every scan re-registers onto the map the bot has already
built.

Method: distance-transform Gauss-Newton scan matching (the Hector-SLAM
technique). Build a distance field D over the map: D(cell) = metres to
the nearest occupied cell. A scan endpoint that truly lies on a wall has
D = 0; drift pushes it off, giving D > 0. We find the small rigid
transform (Δx, Δy, Δθ) of the bot pose that drives Σ D(endpoint)² to a
minimum, by Gauss-Newton on the field and its gradient.

Why this counts as point-to-line: near a wall the distance field's
iso-contours run parallel to the wall and ∇D is the wall normal, so each
residual penalises only the wall-perpendicular error, exactly what
point-to-line ICP does, but with no explicit line extraction or nearest-
neighbour search. Dependencies stay numpy + cv2 (already used lib-wide),
so it runs unchanged in Webots and on the Mac.

Safety: corrections are capped (`max_trans_m`, `max_rot_rad`). A solution
that wants to move the bot further than that is treated as a bad match
(featureless area, too few points, or genuine kidnap) and REJECTED; the
original odometry pose is returned untouched. Drift correction must only
ever nudge; it must never teleport the robot.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from ..config import GridConfig


@dataclass
class MatchResult:
    pose:        Tuple[float, float, float]  # corrected (x, y, yaw)
    accepted:    bool                        # False -> original pose returned
    iters:       int
    mean_resid_m: float                      # mean |D| over points, post-fit
    corr_trans_m: float                      # applied translation magnitude
    corr_rot_rad: float                      # applied rotation magnitude


def _bilinear(arr: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Bilinear sample of `arr` (indexed [x, y]) at float cell coords."""
    dim_x, dim_y = arr.shape
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, dim_x - 2)
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, dim_y - 2)
    x1, y1 = x0 + 1, y0 + 1
    wx = np.clip(fx - x0, 0.0, 1.0)
    wy = np.clip(fy - y0, 0.0, 1.0)
    return (arr[x0, y0] * (1 - wx) * (1 - wy)
            + arr[x1, y0] * wx * (1 - wy)
            + arr[x0, y1] * (1 - wx) * wy
            + arr[x1, y1] * wx * wy)


def match_scan(
    occ: np.ndarray,
    scan,
    pose: Tuple[float, float, float],
    cfg: GridConfig,
    *,
    max_iters: int = 20,
    min_points: int = 25,
    min_wall_cells: int = 40,
    max_trans_m: float = 0.08,
    max_rot_rad: float = 0.12,
    damping: float = 1e-3,
    converge_eps: float = 1e-4,
) -> MatchResult:
    """Align `scan` (captured at `pose`) to the occupancy map `occ`.

    Returns a MatchResult; on rejection `pose` is the unchanged input and
    `accepted` is False. The caller decides whether to trust an accepted
    correction (e.g. blend a fraction of it); this function applies the
    full Gauss-Newton solution and only vetoes implausible jumps.
    """
    x0p, y0p, yaw0 = pose
    reject = lambda: MatchResult((x0p, y0p, yaw0), False, 0, float("inf"), 0.0, 0.0)

    if scan is None or int(occ.sum()) < min_wall_cells:
        return reject()

    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if ranges.size == 0:
        return reject()
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
    finite = np.isfinite(ranges) & (ranges > 0.05)
    if int(finite.sum()) < min_points:
        return reject()
    r  = ranges[finite]
    a  = angles[finite]
    px = r * np.cos(a)            # endpoints in the sensor/body frame
    py = r * np.sin(a)

    # Distance field (metres) + its world-frame gradient, built once.
    free = (~occ).view(np.uint8)
    # Exact-Euclidean DT (DIST_MASK_PRECISE): smoother sub-cell gradients
    # than the 3x3 approximation, which matters for fine pose convergence.
    D = cv2.distanceTransform(free, cv2.DIST_L2,
                              cv2.DIST_MASK_PRECISE).astype(np.float64) * cfg.cell_m
    gX = np.gradient(D, cfg.cell_m, axis=0)   # ∂D/∂x_world  (per metre)
    gY = np.gradient(D, cfg.cell_m, axis=1)   # ∂D/∂y_world
    half, cell = cfg.half_m, cfg.cell_m

    x, y, yaw = x0p, y0p, yaw0
    iters = 0
    mean_resid = float("inf")
    for iters in range(1, max_iters + 1):
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * px - s * py
        wy = y + s * px + c * py
        # -0.5 puts the sample on cell CENTRES: cell (i,j)'s centre is at
        # fractional coord (i+0.5, j+0.5), but the field arrays are indexed
        # so that D[i,j] *is* that centre's value. Without the shift the
        # bilinear sampler reads half a cell off, biasing the fit by ~1 cell.
        fx = (wx + half) / cell - 0.5
        fy = (wy + half) / cell - 0.5

        resid = _bilinear(D,  fx, fy)
        gx    = _bilinear(gX, fx, fy)
        gy    = _bilinear(gY, fx, fy)
        # Jacobian of D wrt (x, y, yaw): the yaw column comes from
        # d(world)/dyaw = (-(wy - y), (wx - x)).
        j_th = -gx * (wy - y) + gy * (wx - x)

        # Normal equations H Δ = -b with Levenberg-Marquardt damping.
        J = np.stack([gx, gy, j_th], axis=1)          # (M, 3)
        H = J.T @ J
        b = J.T @ resid
        H[np.diag_indices(3)] += damping * np.diag(H).clip(min=1e-9)
        try:
            delta = -np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            return reject()

        x   += float(delta[0])
        y   += float(delta[1])
        yaw += float(delta[2])
        mean_resid = float(np.mean(np.abs(resid)))
        if np.linalg.norm(delta) < converge_eps:
            break

    corr_trans = math.hypot(x - x0p, y - y0p)
    corr_rot   = abs(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0)))
    if corr_trans > max_trans_m or corr_rot > max_rot_rad or not math.isfinite(mean_resid):
        return reject()

    return MatchResult((x, y, yaw), True, iters, mean_resid, corr_trans, corr_rot)
