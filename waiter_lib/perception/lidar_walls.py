"""LiDAR wall ingest: fold each fresh LidarScan into a voting occupancy map.

One scan = one geometric snapshot of the maze from the bot's pose-at-
scan-time. Rather than stamping/clearing `wall_status` on a single
observation (which made distant walls flicker on/off as angular aliasing
landed the same wall in slightly different cells each revolution), every
cell carries a confidence vote in `world.wall_vote` and `wall_status`
flips off it with hysteresis:

  - HIT: a finite ray's endpoint cell. Vote += WALL_VOTE_HIT_GAIN
           (clamped to WALL_VOTE_MAX). A cell becomes a wall the first
           time its vote reaches WALL_VOTE_ON, and repeated hits saturate
           the vote so an established wall is very sticky.
  - MISS: a cell a ray PASSED THROUGH on the way to its return (free
           space). Vote -= a range-weighted amount: near misses subtract
           a lot (free space updates quickly), distant misses subtract
           little (a far wall needs many consecutive misses to be erased).
           A wall clears only once its vote falls below WALL_VOTE_OFF.

Occlusion is handled by *where* misses are applied. A FINITE return clears
free space only up to its hit (the nearest obstacle), so cells in the
shadow of a closer wall are never touched and the wall stands. A NO-RETURN
(infinite) ray is NOT proof of empty space. A thin / angled /
low-reflectance occluder can swallow the beam and return nothing, so it
clears only up to the FIRST already-mapped wall on its path and never
carves past it. Without that rule, a no-return ray would sweep straight
through an intermittently-undetected occluder and erase the wall behind
it (the bug that made occluded walls vanish). The upshot is the intended
rule: "assume an occluded wall is still there until a ray reaches that
spot and comes back empty." When the occluder clears, finite returns reach
the cell again; if nothing is there, the misses accumulate and the wall
is eventually removed.

Each cell gets at most ONE vote per scan: a cell hit by any ray this
revolution is never also missed this revolution (hits win), and a cell
swept by several rays still counts as a single miss. This keeps one
revolution = one observation, which is what the vote thresholds are tuned
against.

Returns telemetry counts (n_status_added, n_status_cleared) for logging.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np

from ..config import GridConfig

# Voting parameters -------------------------------------------------------
WALL_VOTE_MAX      = 10     # confidence ceiling (also used for the
                            # out-of-band confirmed anchor-footprint stamp)
WALL_VOTE_ON       = 5      # a hit pushing the vote to >= this confirms a wall
WALL_VOTE_OFF      = 2      # vote below this clears the wall (hysteresis gap)
WALL_VOTE_HIT_GAIN = 5      # vote added per hit (one hit confirms; more saturate)

# Range-weighted miss penalties. Near rays are dense + reliable, so a miss
# is strong free-space evidence; far rays are sparse + noisy, so each miss
# counts for little and a distant wall persists across many revolutions.
WALL_MISS_NEAR   = 3
WALL_MISS_MID    = 2
WALL_MISS_FAR    = 1
WALL_MISS_D_NEAR = 1.5      # m: at/below this, use WALL_MISS_NEAR
WALL_MISS_D_FAR  = 3.5      # m: at/above this, use WALL_MISS_FAR


def ingest_lidar(
    world,
    scan,
    pose: Tuple[float, float, float],
    grid: GridConfig,
    max_range_m: float = 8.0,
    sample_carved: Optional[list] = None,
    sweep_motion: Optional[Tuple[float, float, float]] = None,
    deskew_dir: float = 1.0,
    deskew_phase: float = 0.0,
) -> Tuple[int, int]:
    """Apply one LidarScan revolution to the voting map in-place.

    Args:
        world:       WorldState (touches wall_status, wall_status_tof,
                     wall_discovered_tick, wall_vote, tick).
        scan:        LidarScan duck-type (ranges, angle_min, angle_increment).
        pose:        (x, y, yaw) where the scan was captured (the sweep END).
        grid:        GridConfig (cell_m, half_m).
        max_range_m: clamp for no-return rays (match the world's maxRange).
        sweep_motion: signed (dx, dy, dyaw) the bot moved over the revolution,
                     for motion de-skew of a spinning LiDAR. None -> all rays
                     stamped at `pose` (correct for a fixed-instant sensor like
                     Webots; the historical behaviour).
        deskew_dir:  +1/-1 spin-sense selector for the de-skew (see below).
        deskew_phase: 0..1 alignment of the revolution wrap (forgiving).

    Returns: (n_status_added, n_status_cleared) wall_status transitions
    this scan. For logging; the FSM ignores them.
    """
    if scan is None:
        return 0, 0
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if ranges.size == 0:
        return 0, 0

    sx, sy, syaw = pose
    cell_m = grid.cell_m
    half_m = grid.half_m
    ws    = world.wall_status
    tof   = world.wall_status_tof
    disc  = world.wall_discovered_tick
    vote  = world.wall_vote
    dim_x, dim_y = ws.shape

    n = ranges.size
    local_angles = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment

    # Per-ray sensor pose (motion de-skew). A spinning LiDAR captures each ray
    # at a different instant across the ~1/scan_hz sweep, so on a moving/turning
    # bot every ray has a slightly different true pose. Stamping them all at one
    # pose smears walls — worst during turns, which is exactly when it paints
    # phantom walls. When the caller passes the sweep motion we interpolate the
    # pose across the revolution; `pose` is the sweep-END snapshot, so earlier
    # rays rewind a fraction of the motion. `deskew_dir` (+1/-1) selects the
    # spin sense — the ONE knob that can make de-skew worse if wrong (flip it if
    # turning degrades the map); `deskew_phase` aligns the wrap and is forgiving
    # (a phase error is just a small constant rotation the voting map absorbs).
    # Webots' LiDAR is a fixed-instant sensor (no skew) -> the sim passes no
    # motion and this whole branch is skipped there.
    if sweep_motion is not None:
        dx, dy, dyaw = sweep_motion
        f = (deskew_dir * ((np.arange(n, dtype=np.float64) + 0.5) / n)
             + deskew_phase) % 1.0           # capture-time fraction, 0=start 1=end
        back = 1.0 - f                        # how far to rewind from the end pose
        sx_i   = sx   - back * dx
        sy_i   = sy   - back * dy
        syaw_i = syaw - back * dyaw
    else:
        sx_i   = np.full(n, sx)
        sy_i   = np.full(n, sy)
        syaw_i = np.full(n, syaw)
    world_yaws = syaw_i + local_angles

    # Webots returns 0 for "ray not yet swept this revolution" AND for any
    # return below minRange -- neither is informative. Filter both alongside
    # +inf (no-return).
    valid    = ranges > 0.05
    finite_h = valid & np.isfinite(ranges)

    # HIT cells -> boolean grid mask (finite-return endpoints) -----------
    hit_mask = np.zeros((dim_x, dim_y), dtype=bool)
    if finite_h.any():
        hr   = ranges[finite_h]
        hyaw = world_yaws[finite_h]
        hgxs = ((sx_i[finite_h] + hr * np.cos(hyaw) + half_m) / cell_m).astype(np.int32)
        hgys = ((sy_i[finite_h] + hr * np.sin(hyaw) + half_m) / cell_m).astype(np.int32)
        in_b = (hgxs >= 0) & (hgxs < dim_x) & (hgys >= 0) & (hgys < dim_y)
        hit_mask[hgxs[in_b], hgys[in_b]] = True

    # MISS cells -> boolean grid mask (free space each ray traversed) ----
    # Walk every ray from the sensor to its OWN return (finite) or to
    # maxRange (no-return); endpoint=False so the hit cell isn't walked.
    # Cells past a hit are never reached (the occlusion guarantee). A grid
    # mask auto-dedups cells swept by many rays into one miss, and
    # `&= ~hit_mask` makes hits win over misses this revolution.
    # Fully vectorised across all carve rays at once (the per-ray Python loop
    # was ~13 ms/scan; at 5.5 Hz that spiked the FSM tick and starved the
    # camera-consumption cadence). Sample every ray at a fixed half-cell step
    # in one (K_rays x S_steps) grid; `within = d < r` enforces endpoint=False
    # (the hit cell isn't walked -> occlusion of cells past a hit is automatic).
    miss_mask = np.zeros((dim_x, dim_y), dtype=bool)
    ray_ranges = np.minimum(np.where(np.isfinite(ranges), ranges, max_range_m),
                            max_range_m)
    half_cell = cell_m * 0.5
    ci = np.where(ray_ranges > cell_m)[0]
    if ci.size:
        r    = ray_ranges[ci]                                  # (K,)
        yaw  = world_yaws[ci]                                  # (K,)
        sxc  = sx_i[ci][:, None]                               # (K,1) per-ray origin
        syc  = sy_i[ci][:, None]
        steps = np.arange(int(max_range_m / half_cell) + 1)    # (S,)
        d    = steps * half_cell                               # (S,) distances
        within = d[None, :] < r[:, None]                       # (K,S) endpoint=False
        gx = ((sxc + np.cos(yaw)[:, None] * d[None, :] + half_m) / cell_m).astype(np.int32)
        gy = ((syc + np.sin(yaw)[:, None] * d[None, :] + half_m) / cell_m).astype(np.int32)
        valid = within & (gx >= 0) & (gx < dim_x) & (gy >= 0) & (gy < dim_y)
        # NO-RETURN (infinite) rays are NOT proof of empty space. A thin /
        # angled / low-reflectance occluder can swallow the beam and return
        # nothing. So a no-return ray clears only up to the FIRST already-
        # mapped wall on its path, never past it (else it'd erase the wall
        # hiding behind an undetected occluder). A finite return IS proof the
        # beam reached its hit (the nearest obstacle), so it clears straight
        # through stale cells up to that hit. Implement the no-return stop by
        # masking off cells at/after the first on-ray mapped wall via cumsum.
        blocked = ws[np.clip(gx, 0, dim_x - 1), np.clip(gy, 0, dim_y - 1)] & valid
        before_wall = np.cumsum(blocked, axis=1) == 0          # cells before 1st wall
        keep = valid.copy()
        inf_rows = ~finite_h[ci]
        keep[inf_rows] &= before_wall[inf_rows]
        miss_mask[gx[keep], gy[keep]] = True
    miss_mask &= ~hit_mask

    # Apply one vote per cell, flip wall_status with hysteresis ----------
    hx, hy = np.where(hit_mask)
    vote[hx, hy] = np.minimum(WALL_VOTE_MAX, vote[hx, hy] + WALL_VOTE_HIT_GAIN)
    new_wall = (~ws[hx, hy]) & (vote[hx, hy] >= WALL_VOTE_ON)
    nhx, nhy = hx[new_wall], hy[new_wall]
    ws[nhx, nhy]   = True
    tof[nhx, nhy]  = True             # dashboard paints these magenta
    disc[nhx, nhy] = world.tick
    n_added = int(new_wall.sum())

    mx, my = np.where(miss_mask)
    # Range-weighted miss penalty: near misses subtract a lot, far misses
    # little, so distant walls persist across many revolutions.
    d   = np.hypot((mx + 0.5) * cell_m - half_m - sx,
                   (my + 0.5) * cell_m - half_m - sy)
    pen = np.select([d <= WALL_MISS_D_NEAR, d >= WALL_MISS_D_FAR],
                    [WALL_MISS_NEAR, WALL_MISS_FAR], default=WALL_MISS_MID)
    vote[mx, my] = np.maximum(0, vote[mx, my] - pen)
    cleared = ws[mx, my] & (vote[mx, my] < WALL_VOTE_OFF)
    cmx, cmy = mx[cleared], my[cleared]
    ws[cmx, cmy]   = False
    tof[cmx, cmy]  = False
    disc[cmx, cmy] = -1
    n_cleared = int(cleared.sum())
    if sample_carved is not None:
        for gx, gy in zip(cmx[:6].tolist(), cmy[:6].tolist()):
            sample_carved.append({"gx": gx, "gy": gy, "vote": int(vote[gx, gy])})

    return n_added, n_cleared
