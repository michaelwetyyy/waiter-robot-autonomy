"""Dynamic Window Approach local planner.

Replaces the per-tick waypoint chaser. Same pipeline shape as Fox /
Burgard / Thrun (1997): sample a grid of candidate (v, ω) pairs from
the dynamic window, forward-simulate each one against the inflated
occupancy grid, reject collisions, score survivors, pick the best.

Why we use it here:
  - The waypoint chaser decelerates into every grid-cell corner of the
    A* path. With 2.5 cm cells that's ~50 decelerations across the
    maze. DWA scores a SMOOTH velocity that carves through cells.
  - Rollout collision-checks against the same inflated grid A* plans
    over, so the two stay consistent.
  - Replaces SEEK's spin-down-then-pivot-then-accelerate at every
    corner with a single (v, ω) per tick. Far smoother + faster.

Scoring is a "pure pursuit + endpoint safety" hybrid (third iteration):

    - carrot_weight  · ‖rollout_end - carrot‖    # pull along path
    - end_dev_weight · cross-track at endpoint   # land on the polyline
    - path_dev_weight· RMS cross-track           # don't bow off mid-rollout
    - heading_weight · |final_yaw - tangent|     # break v=0 ties

The CARROT is a point on the DENSE path at a fixed arc-length ahead of
the bot's current path projection. Two notes on why this version
finally works after two failed cuts:

1. The carrot used to live on the SMOOTHED path with a cell-count
   lookahead; on an L-route the smoothed path has 3-4 points and
   `closest_idx + 8` snapped to the goal, so the carrot pulled across
   corners. Now it lives on the DENSE path at a fixed arc-distance
   ahead, so it always sits ON the polyline ~15 cm forward.

2. The cross-track-only scoring (no carrot, pure penalties) froze the
   bot whenever its initial yaw was misaligned with the path tangent.
   Converging onto the path requires traversing some transient
   cross-track, and cross-track-only scoring punishes that the same
   as drifting AWAY. Pure pursuit naturally rewards convergence
   because moving toward the carrot is the same as moving toward the
   path; `end_dev` then blocks chord-across-corner shortcuts that
   would otherwise win on raw carrot-distance.

Pure math + numpy. No platform dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..config import GridConfig, RobotMechanics
from ..planning.grid import grid_to_world


@dataclass(frozen=True)
class DWAConfig:
    """All knobs in one place. Defaults tuned for the e-puck at
    cruise_rad_s ≈ 5.5 (≈ 0.11 m/s linear, ≈ 4 rad/s spin) on a 2.5 cm
    grid; bump `horizon_s` if cruise rises substantially.

    The weights are a balance: too much endpoint adherence and the
    bot freezes the moment its yaw drifts off the path; too much
    carrot pull and it cuts corners. The sanity check that ties them
    together: for a typical 5 cm rollout converging from a 15° yaw
    error, the curving trajectory's d_carrot advantage (~5 cm) must
    outweigh its end_dev cost (~1.5 cm x 15 = 0.225). 4 x 0.05 = 0.20
    progress reward versus 0.225 endpoint penalty, which is marginal. Bump
    carrot_weight to 4.0 to give convergence a clear win, and the
    behaviour locks in.
    """
    horizon_s:        float = 0.5     # forward simulation horizon
    rollout_dt:       float = 0.05    # rollout step (≈ 10 steps per trajectory)
    n_v_samples:      int   = 7       # forward velocities, includes 0
    n_w_samples:      int   = 21      # angular velocities, symmetric about 0

    carrot_arc_m:     float = 0.15    # carrot lives at this arc-length
                                       # ahead of the bot's path projection.
                                       # Larger = smoother / lazier
                                       # convergence; smaller = tighter
                                       # tracking but more prone to
                                       # oscillation near corners.
    carrot_weight:    float = 4.0     # pull toward the carrot point.
                                       # The ONLY term that rewards
                                       # forward motion; everything else
                                       # is a penalty, so without this
                                       # the algorithm prefers v=0.
    end_dev_weight:   float = 15.0    # cross-track at rollout endpoint.
                                       # This is the corner-cut blocker:
                                       # a chord-across-corner trajectory
                                       # ends ON the inside of the corner
                                       # (off-path), so end_dev punishes
                                       # it heavily while d_carrot would
                                       # actually reward it.
    path_dev_weight:  float = 3.0     # RMS cross-track over the rollout.
                                       # Light; exists to discourage
                                       # trajectories that bow far off
                                       # the path even if they return at
                                       # the endpoint. Too high here and
                                       # the freeze bug returns: any
                                       # transient deviation during
                                       # convergence gets over-penalised.
    heading_weight:   float = 0.2     # |final_yaw - path_tangent|.
                                       # Tie-breaker among v=0 candidates
                                       # so the bot pivots toward the
                                       # path tangent when stationary.

    path_search_pad:  int   = 4       # cells either side of the bot's
                                       # dense_closest that participate
                                       # in the cross-track / nearest-
                                       # path-point lookups. Plus the
                                       # rollout reach window; see
                                       # `select_velocity`.


def advance_path_index(
    path: List[Tuple[int, int]],
    bot_x: float, bot_y: float,
    last_idx: int,
    grid: GridConfig,
) -> int:
    """Monotonic progress along the A* path.

    Returns the index of the path cell currently CLOSEST to the bot,
    constrained to be ≥ `last_idx` so progress never reverses if the
    bot loops near an earlier path cell.

    Early-exits once we've moved past the local minimum (the path is
    a polyline; once distance starts growing it keeps growing).
    """
    best_idx = last_idx
    best_d2  = math.inf
    for i in range(last_idx, len(path)):
        wx, wy = grid_to_world(path[i][0], path[i][1], grid)
        d2 = (wx - bot_x) * (wx - bot_x) + (wy - bot_y) * (wy - bot_y)
        if d2 < best_d2:
            best_d2  = d2
            best_idx = i
        elif d2 > best_d2 + 0.09:   # 30 cm past the minimum; stop scanning
            break
    return best_idx


def select_velocity(
    bot_x: float, bot_y: float, bot_yaw: float,
    path: List[Tuple[int, int]],
    closest_idx: int,
    inflated_grid: np.ndarray,
    grid: GridConfig,
    robot: RobotMechanics,
    cruise_rad_s: float,
    dwa: DWAConfig = DWAConfig(),
) -> Optional[Tuple[float, float, float, float]]:
    """Pick the best (v, ω) from the dynamic window.

    Returns (v_l, v_r, v, ω) ready for set_wheel_velocities, plus the
    chosen body-frame linear / angular velocity for logging. Returns
    None if every sampled trajectory collides with the inflated grid
    (caller should fall back to reverse-out or survey behaviour).

    Scoring (signs included; each term is added to `score`):
        - carrot_weight    · ‖rollout_end - carrot‖       # pull along path
        - end_dev_weight   · cross-track at rollout endpoint
        - path_dev_weight  · RMS cross-track over rollout
        - heading_weight   · |final_yaw - tangent|

    The CARROT is a point on the DENSE path at arc-length
    `bot_arc + carrot_arc_m` (clamped to the path's last point).
    Distance to the carrot is the only positive-direction force:
    trajectories that advance toward it (along the path's actual
    centreline) score better than ones that don't. `end_dev` is the
    safety net; it blocks corner-cutting trajectories that beat
    d_carrot via the chord across a bend, because those trajectories
    end OFF the polyline.

    `closest_idx` is retained as a parameter for backwards compat with
    callers but is unused; the dense-path nearest index is recomputed
    internally each call (cheap, O(N_dense), one argmin).
    """
    R = robot.wheel_radius_m
    B = robot.wheel_base_m
    v_max = cruise_rad_s * R
    # Maximum body angular velocity = one wheel at +max, the other at
    # -max; |Δv_wheel| = 2·v_max, so ω_max = 2·v_max / B.
    w_max = 2.0 * v_max / B

    # Densify the smoothed A* polyline before computing the deviation
    # cost. `smooth_path` collapses long straight runs into just their
    # endpoints (so a 1.8 m open-area path can come through as 2 cells:
    # start and goal), and "distance to NEAREST path point" against
    # such a sparse path is the wrong measure of "distance to the
    # path's centreline": a trajectory driving along the segment
    # moves AWAY from the start point faster than it approaches the
    # goal point, so the deviation cost wrongly penalises forward
    # motion. Resampling at ~one point per grid cell turns the polyline
    # into a uniform sample cloud where nearest-point ≈ true centreline
    # distance.
    cell_m = grid.cell_m
    half_m = grid.half_m
    raw_xs = (np.asarray([p[0] for p in path], dtype=np.float64) + 0.5) * cell_m - half_m
    raw_ys = (np.asarray([p[1] for p in path], dtype=np.float64) + 0.5) * cell_m - half_m
    dense_xs_list: List[float] = [float(raw_xs[0])]
    dense_ys_list: List[float] = [float(raw_ys[0])]
    for i in range(1, len(path)):
        x0, y0 = float(raw_xs[i - 1]), float(raw_ys[i - 1])
        x1, y1 = float(raw_xs[i]),     float(raw_ys[i])
        seg_d  = math.hypot(x1 - x0, y1 - y0)
        n_sub  = max(1, int(math.ceil(seg_d / cell_m)))
        for k in range(1, n_sub + 1):
            t = k / n_sub
            dense_xs_list.append(x0 + t * (x1 - x0))
            dense_ys_list.append(y0 + t * (y1 - y0))
    path_xs = np.asarray(dense_xs_list, dtype=np.float64)
    path_ys = np.asarray(dense_ys_list, dtype=np.float64)

    # Cumulative arc-length per dense point. Used to place the carrot
    # at a fixed arc-distance ahead of the bot, regardless of how the
    # path bends in between. Chord-distance ahead would let the
    # carrot snap across a corner the moment the bot enters the
    # corner cell, exactly the failure mode we're trying to prevent.
    seg_dx     = np.diff(path_xs)
    seg_dy     = np.diff(path_ys)
    seg_lens   = np.hypot(seg_dx, seg_dy)
    arc_cum    = np.concatenate(([0.0], np.cumsum(seg_lens)))

    # Find the bot's nearest dense index up-front. O(N_dense), once
    # per call, cheap vs the per-trajectory inner loop.
    bot_to_dense  = np.hypot(path_xs - bot_x, path_ys - bot_y)
    dense_closest = int(bot_to_dense.argmin())
    bot_arc       = float(arc_cum[dense_closest])

    # Locate the carrot: the dense point whose arc-length is closest
    # to bot_arc + carrot_arc_m. searchsorted gives us O(log N) instead
    # of a second argmin.
    target_arc    = bot_arc + dwa.carrot_arc_m
    carrot_idx    = int(np.searchsorted(arc_cum, target_arc))
    if carrot_idx >= len(path_xs):
        carrot_idx = len(path_xs) - 1
    carrot_x      = float(path_xs[carrot_idx])
    carrot_y      = float(path_ys[carrot_idx])

    # Window for the cross-track lookup. Must cover (a) one rollout's
    # worth of arc forward (so the rollout endpoint can find its
    # nearest path point) and (b) the carrot's index. Pad each side
    # by `path_search_pad` cells.
    reach_m       = v_max * dwa.horizon_s
    reach_cells   = max(1, int(reach_m / cell_m) + 2)
    win_hi_target = max(carrot_idx + 1, dense_closest + reach_cells + 1)
    win_lo        = max(0, dense_closest - dwa.path_search_pad)
    win_hi        = min(len(path_xs), win_hi_target + dwa.path_search_pad)
    win_xs        = path_xs[win_lo:win_hi]
    win_ys        = path_ys[win_lo:win_hi]

    n_steps  = max(1, int(dwa.horizon_s / dwa.rollout_dt))
    dt       = dwa.rollout_dt
    ts       = np.arange(1, n_steps + 1, dtype=np.float64) * dt
    horizon  = n_steps * dt
    inv_n    = 1.0 / n_steps

    # Heading target = path tangent at the bot's nearest dense point
    # via FORWARD difference (not central). The earlier central-
    # difference version had a corner-cutting bug: at a corner cell
    # `i`, `path[i+1] - path[i-1]` is the bisector of the incoming
    # and outgoing legs, so the heading penalty rewarded trajectories
    # aimed diagonally THROUGH the corner, exactly the wall-clipping
    # we were trying to prevent. Forward diff at a corner cell points
    # along the OUTGOING leg, so the bot starts rotating into the new
    # leg's direction as soon as it reaches the corner.
    i = dense_closest
    n_dense = len(path_xs)
    if n_dense >= 2:
        if i < n_dense - 1:
            tdx = float(path_xs[i + 1] - path_xs[i])
            tdy = float(path_ys[i + 1] - path_ys[i])
        else:                                  # at goal: fall back to the
            tdx = float(path_xs[i] - path_xs[i - 1])    # last segment's
            tdy = float(path_ys[i] - path_ys[i - 1])    # direction.
    else:
        tdx, tdy = math.cos(bot_yaw), math.sin(bot_yaw)
    target_yaw = math.atan2(tdy, tdx)

    v_samples = np.linspace(0.0,    v_max, dwa.n_v_samples)
    w_samples = np.linspace(-w_max, w_max, dwa.n_w_samples)

    dim       = inflated_grid.shape[0]

    best_score = -math.inf
    best:      Optional[Tuple[float, float]] = None

    for v in v_samples:
        for w in w_samples:
            # Unicycle rollout, closed-form for the whole horizon. For
            # ω≈0 the arc collapses to a straight line; guard the
            # division explicitly so we don't get inf/nan at zero spin.
            if abs(w) < 1e-6:
                xs = bot_x + v * math.cos(bot_yaw) * ts
                ys = bot_y + v * math.sin(bot_yaw) * ts
            else:
                yaws = bot_yaw + w * ts
                xs = bot_x + (v / w) * (np.sin(yaws) - math.sin(bot_yaw))
                ys = bot_y - (v / w) * (np.cos(yaws) - math.cos(bot_yaw))

            # Vectorised collision check: any cell along the rollout
            # that's inflated-occupied OR clamped (off-grid) kills it.
            gxs = np.clip(((xs + half_m) / cell_m).astype(np.int32), 0, dim - 1)
            gys = np.clip(((ys + half_m) / cell_m).astype(np.int32), 0, dim - 1)
            if inflated_grid[gxs, gys].any():
                continue

            # Cross-track distance² from each rollout step to its
            # nearest dense window point. Used twice: RMS over the
            # full rollout (mid-flight bowing), and the endpoint
            # value (the "did the trajectory land on the line" check).
            dxs2  = (xs[:, None] - win_xs[None, :]) ** 2
            dys2  = (ys[:, None] - win_ys[None, :]) ** 2
            d2    = dxs2 + dys2
            per_step_min_sq = d2.min(axis=1)
            rms_dev = math.sqrt(float(per_step_min_sq.sum()) * inv_n)
            end_dev = math.sqrt(float(per_step_min_sq[-1]))

            # Pull toward the carrot. The carrot is on the polyline at
            # arc-length `carrot_arc_m` ahead, so trajectories that
            # advance ALONG the path get a smaller d_carrot than ones
            # that stand still or wander. Chord-across-corner cheaters
            # also get a small d_carrot, but end_dev kills them: their
            # endpoint sits in the inside of the corner, off-path.
            d_carrot = math.hypot(float(xs[-1]) - carrot_x,
                                  float(ys[-1]) - carrot_y)

            # Heading alignment of the trajectory's FINAL yaw with the
            # path tangent. Closed-form: ω is constant over the
            # rollout, so final_yaw = bot_yaw + ω·horizon.
            final_yaw  = bot_yaw + w * horizon
            yaw_err    = target_yaw - final_yaw
            # Wrap to [-π, π] without importing; modulo is cheap and
            # avoids an extra function call inside the hot loop.
            yaw_err    = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
            head_pen   = abs(yaw_err)

            score = (- dwa.carrot_weight   * d_carrot
                     - dwa.end_dev_weight  * end_dev
                     - dwa.path_dev_weight * rms_dev
                     - dwa.heading_weight  * head_pen)
            if score > best_score:
                best_score = score
                best       = (float(v), float(w))

    if best is None:
        return None
    v, w = best
    # Body (v, ω) -> individual wheel-tip linear speeds -> angular speeds.
    v_l = (v - w * B / 2.0) / R
    v_r = (v + w * B / 2.0) / R
    return v_l, v_r, v, w
