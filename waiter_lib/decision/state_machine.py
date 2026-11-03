"""The waiter robot's main FSM: hybrid map-and-plan + reactive avoidance.

Pure logic; reads sensors + perception output, writes wheel commands
via `SensorSource`. No Webots imports. Same code runs on Webots and Pi.

Algorithm: A* over a partial occupancy grid + reactive Bug-style fallback.

  - Perception accumulates only NEAR-FIELD wall detections (within 25 cm)
    into the occupancy grid. These are high-confidence: at close range the
    bottom-edge inverse-perspective projection is reliable because nothing
    can occlude the floor right in front of the bot.
  - Each tick, A* attempts to plan a path from the bot to the cup.
    Initially the grid is empty so A* draws a straight line to the cup;
    as the bot encounters and maps walls, A* routes around the known ones.
  - If a wall the bot doesn't know about yet appears in the near field
    (AVOID_TRIGGER), the FSM steps in: it rotates toward the side with
    more clearance, exposing the wall to the camera at close range so it
    gets added to the map. Once that wall is in the map, A* on the next
    tick naturally routes around it.

State flow:

    SEARCH -- (saw cup)          -> SEEK
    SEEK   -- (ahead blocked)    -> AVOID
    SEEK   -- (close to cup)     -> CENTRE
    AVOID  -- (ahead clear)      -> SEEK
    CENTRE -- (under cup)        -> DONE
    DONE   stays forever
"""

from __future__ import annotations
import math
import time
from typing import Optional, Tuple

import numpy as np
import cv2

from ..config import Config
from ..world_state import WorldState, FSMState
from ..sensors.interface import SensorSource
from ..perception.aruco import ArucoCupDetector, CupPose
from ..perception.walls import near_field_clearances, NearField
from ..perception.wall_lines import extract_segments
from ..perception.lidar_walls import ingest_lidar, WALL_VOTE_MAX
from ..perception.scan_match import match_scan
from ..planning.grid import (world_to_grid, world_to_grid_checked,
                              inflate_grid, clearance_cost_field)
from ..planning.astar import a_star, smooth_path, string_pull
from ..planning.clamp import nearest_free
from ..motion.differential import wrap_angle, clamp_wheel_pair, heading_pd
from ..motion import dwa as dwa_mod
from .. import telemetry


# Tuning. Algorithm-specific, kept at module scope for easy tweaking.
GOTO_REACH_M      = 0.12   # drive-to-goal: stop + hold within this of the goal
SEARCH_SPIN_RAD_S = 1.2    # wheel speed during SEARCH spin
# Reactive near-field guard (used by the drive-to-goal stack): a wall this
# close dead-ahead that the inflated map may not carry yet -> turn in place
# toward the more open side instead of driving into it.
AVOID_TRIGGER_M   = 0.10
PLAN_EVERY_TICKS  = 20     # how often to re-run A* (every ~0.3 s)
# DWA local planner config: defaults from `motion.dwa.DWAConfig` (see
# that module for the knob explanations). `path_idx` is repurposed by
# the DWA path-follower: it's now the index of the path cell currently
# CLOSEST to the bot, advancing monotonically; used as the start of
# the next tick's carrot search.
DWA_CFG = dwa_mod.DWAConfig()


# 360° LiDAR (DFRobot DFR0315)
# Maximum usable range; must match the `maxRange` field in the world
# files' Lidar nodes. Rays returning +inf (no return inside this range)
# are treated as "free space all the way out to this distance" for the
# negative-evidence carving in `ingest_lidar`.
LIDAR_MAX_RANGE_M    = 8.0
# Throttle for the "LiDAR ingest produced N hits / M carves" log so the
# console isn't flooded at 5.5 Hz. ~1 line per second at the perception
# cadence.
LIDAR_INGEST_LOG_EVERY_TICKS = 60

# LiDAR scan-to-map drift correction (ICP)
# Each fresh scan is matched against the accumulated wall map to correct
# odometry drift BEFORE stamping (see perception.scan_match, a distance-
# transform Gauss-Newton matcher, point-to-line in spirit). Only a
# FRACTION of the solved correction is applied per scan, so one noisy
# match can't yank the pose; over a few 5.5 Hz scans the correction
# converges. Implausibly large solutions are rejected by the matcher's
# own caps and leave the pose untouched. This is now the bot's sole drift
# correction (the ArUco anchor was removed), so the gain is a touch higher
# than the old anchor-backed 0.5. Set ENABLED = False for pure odometry.
LIDAR_ICP_ENABLED          = True
LIDAR_ICP_GAIN             = 0.7
LIDAR_ICP_MIN_WALL_CELLS   = 60     # need enough map to match against
LIDAR_ICP_LOG_EVERY_TICKS  = 120
# ICP corrects *odometry drift*, and drift only accrues while the bot
# moves. If odometry reports the bot essentially stationary since the last
# scan, there is no drift to correct; applying the matcher's residual
# (which is never exactly zero on a real, noisy scan) would just make a
# parked bot wander the map. So skip the correction below these per-scan
# motion thresholds.
LIDAR_ICP_MIN_TRANS_M      = 0.005          # 5 mm
LIDAR_ICP_MIN_ROT_RAD      = math.radians(0.5)

# LiDAR motion de-skew (spinning-sensor only; hardware opts in via
# WaiterFSM(lidar_deskew=True), the Webots fixed-instant LiDAR does NOT).
# A spinning LiDAR samples each ray at a different instant across the ~1/scan_hz
# sweep, so while the bot turns, every ray carries a slightly different true
# pose; stamping them all at the one snapshot pose smears walls into phantoms
# (the failure mode behind "false LiDAR data" while driving to a goal). We feed
# the per-sweep motion to ingest_lidar so it interpolates the pose per ray.
#   DIR   — spin-sense selector; the ONE term that can make de-skew WORSE if
#           wrong. If turning visibly degrades the map on hardware, flip to -1.
#   PHASE — aligns the revolution's wrap; forgiving (a phase error is just a
#           small constant scan rotation the voting map/ICP absorb).
# Guards: skip de-skew when the per-sweep motion is implausibly large (a pose
# teleport from move-bot/reset, or a multi-sweep gap after RECALIBRATE) so a
# bogus delta can't fling the scan.
LIDAR_DESKEW_DIR           = +1.0
LIDAR_DESKEW_PHASE         = 0.0
LIDAR_DESKEW_MAX_ROT_RAD   = 1.2            # > this in one sweep -> don't de-skew
LIDAR_DESKEW_MAX_TRANS_M   = 0.40           # > this in one sweep -> don't de-skew

# Body-presence wall clearing
# The cheapest form of SLAM evidence: the bot itself. If the robot is
# physically sitting at (bot_x, bot_y) and hasn't crashed, then no
# real wall can exist within its body radius; near-field AVOID would
# have triggered (PIVOT/PANIC) long before the bot actually entered a
# wall cell. So any wall mark inside the body
# disk is a projection phantom and gets wiped.
#
# Fixes the failure mode where A* can't plot a route because the bot's
# start cell sits inside a phantom wall (which then inflates outward,
# closing off every neighbouring cell too).
#
# Two radii because the rule applies at different layers:
#   - RAW: clear wall_status within bot_radius + one cell. This is
#     the strict physical claim; a wall here means the bot is inside
#     a wall, which is impossible.
#   - INFLATED: in the inflated grid A* actually reads, clear cells
#     within bot_radius + inflate_m. Body-only wasn't enough: when the
#     bot sits inside a legit wall's inflation halo (bot center 4-7 cm
#     from a real wall), the halo extends past a body-sized hole and
#     re-closes A*'s exit moves laterally. A disk wider by the full
#     inflate_m guarantees the entire halo of any wall the bot is
#     "inside the buffer of" is cleared, so A* always has at least one
#     exit direction. Raw wall_status is still untouched (real walls
#     remain; AVOID/DWA handle them at the controller level).
BOT_BODY_RADIUS_M       = 0.037   # e-puck body (matches config.py comment)
BODY_CLEAR_RAW_M        = BOT_BODY_RADIUS_M + 0.025   # body + 1 cell
# Inflated-grid clear radius is derived per-replan as
#   BOT_BODY_RADIUS_M + self.cfg.grid.inflate_m
# (read inside `_replan` because `inflate_m` lives in the runtime
#  GridConfig; keeping it out of a module constant means the two
#  values can never drift apart).

# Wall map
# The wall map (`wall_status`) is now populated by:
#   - LiDAR  (perception/lidar_walls.py): every revolution stamps hits
#     and carves free space along each ray. Mm-accurate; one full
#     revolution at scan-capture pose-snapshot ≈ 182 ms.
# The old camera-projection pipeline (windowed-consensus voting,
# per-scan in-range mask, negative-evidence clearance) was removed
# along with the `_roll_scan` method and `near_field_walls` import.
# Phantom-wall clearance is now LiDAR-driven and effectively
# instantaneous: a single revolution carves through any cell that the
# camera-pipeline used to leave stuck for ≥5 seconds. The
# `wall_status_tof` flag is reused as the "non-camera, geometrically
# confident" marker, set by LiDAR hits (the name is historical).

# ArUco false-positive rejection
# Require this many CONSECUTIVE frames detecting the marker before we
# accept the first cup sighting. Pings the smoothed estimate only after
# the first detection passes verification.
ARUCO_VERIFY_FRAMES   = 3
# All N frames must agree on cup world position within this radius to
# count as "the same detection" (i.e. not a transient mis-decode).
ARUCO_VERIFY_RADIUS_M = 0.20

# Cup smoothing alpha (rolling EMA on glass_xy_acc). Higher = faster
# convergence to new sightings, lower = more noise rejection. With the
# anchor gone, the LiDAR scan-match keeps the pose accurate, so the cup is
# smoothed against the live LiDAR-grounded pose on every sighting.
CUP_ALPHA = 0.3


class WaiterFSM:
    """One tick, one call to `step()`. Owns no platform code."""

    def __init__(self, sensors: SensorSource, config: Config,
                 lidar_deskew: bool = False):
        self.sensors = sensors
        self.cfg = config
        # Spinning-LiDAR motion de-skew: ON only for real hardware (run_brain
        # passes True). Webots' LiDAR samples a whole scan at one instant, so
        # the sim leaves this False and stamps every ray at the snapshot pose.
        self._lidar_deskew = lidar_deskew
        self._prev_scan_pose: Optional[Tuple[float, float, float]] = None
        self.world = WorldState.fresh(grid_dim=config.grid.dim)

        # Latest near-field perception, refreshed each tick.
        self._near: NearField = NearField(math.inf, math.inf, math.inf,
                                          math.inf, math.inf)

        # Perception: build expensive detector once, reuse every tick.
        # `marker_offset_m` is the cup hex apothem; the detector uses it
        # to back-project each visible face to the cup central axis, so
        # multi-face fusion averages estimates of the SAME point rather
        # than pulling toward whichever face is most visible.
        self._aruco = ArucoCupDetector(
            intrinsics=config.camera,
            marker_size_m=config.tuning.marker_size_m,
            aruco_dict_name=config.tuning.aruco_dict_name,
            marker_offset_m=config.tuning.marker_offset_m,
        )

        # Most recent camera frame + ArUco quad, kept for telemetry
        # overlay to annotate the same frame the FSM saw on this tick.
        self._last_rgb:           Optional[np.ndarray] = None
        self._last_pose:          Optional[CupPose]    = None
        # All ArUco quads detected this frame, as a list of
        # (corners4 ndarray, id int) tuples. Empty when nothing is in
        # view. The telemetry overlay iterates over the list and draws
        # each quad (used to be a single "biggest" pair, which hid the
        # fact that the cup detector + anchor fuser look at ALL faces).
        self._last_aruco_detections: list = []

        # Throttle for the A* FAILED diagnostic. _replan fires on every
        # SEEK tick while `path is None`, so without a throttle the
        # diagnostic floods at ~60 lines/sec and the older entries scroll
        # off the Webots console buffer before the user can see them.
        # Print at most once per 60 ticks (~1 s).
        self._last_astar_fail_log_tick: int = -10_000
        self._last_astar_fail_summary: str = ""

        # Pending detection accumulator. ArUco occasionally false-positives
        # (a quad-shaped wall corner + low contrast can momentarily decode
        # as a valid marker). We require N consecutive frames with a
        # mutually-consistent pose before promoting the detection to
        # `glass_xy_acc`. Resets if we miss a frame.
        self._pending_detections: list = []     # list of (world_x, world_y)

        # Return-home: where the bot set off from when it last started driving
        # to a target (cup / hand / map goal). The UI "return home" button drives
        # back here and parks. None = no trip in progress (nothing to return to).
        self._home_xy:     Optional[Tuple[float, float]] = None
        self._returning:   bool = False
        self._was_driving: bool = False   # drove to a target last tick? (edge detect)

        # Cached inflated occupancy grid. Recomputed by `_replan` (which
        # already runs cv2.dilate for A*); DWA reuses the SAME grid for
        # its rollout collision check so the two stay consistent; A*
        # never plans through a cell DWA would reject, and vice versa.
        self._inflated_grid:          Optional[np.ndarray] = None

        # LiDAR debug state: one-shot log on first scan to confirm
        # the device is wired up. The FSM doesn't read `world.lidar_scan`
        # yet; future ICP/SLAM code can pull it from there.
        self._lidar_first_read_logged: bool = False
        # Tick of the last LiDAR ingest log line; throttles printing
        # so the "N hits / M carves" message fires ~once a second
        # rather than at the device's 5.5 Hz scan rate.
        self._lidar_last_ingest_log_tick: int = -10_000
        self._lidar_icp_last_log_tick: int = -10_000
        # Odometry motion accumulated since the last LiDAR scan; gates
        # the ICP drift correction so a stationary bot doesn't wander.
        self._odom_trans_since_scan: float = 0.0
        self._odom_rot_since_scan:   float = 0.0
        # Counter for the per-scan LIDAR DEBUG line. We throttle by
        # scan count (every 4th = ~0.7 s at 5.5 Hz) rather than by
        # tick number, because the lidar's fresh-scan ticks (every
        # ~12 ticks) don't align with simple `tick % N == 0` gates.
        self._lidar_debug_scan_count: int = 0

        # Dashboard A* goal-probe result, refreshed on the imagery cadence
        # and reused on intervening ticks so the readout doesn't flicker.
        # None = no goal set. See `_probe_plan`.
        self._last_probe: Optional[dict] = None

        # Dashboard performance + pipeline telemetry
        # Computed FSM-side (works in sim and over the real ZMQ link with no
        # transport-layer changes). Timings are EMA-smoothed for a stable
        # readout; rates count fresh sensor returns over a ~1 s wall window.
        self._perf = {
            "tick_hz": 0.0, "perception_ms": 0.0, "plan_ms": 0.0,
            "render_ms": 0.0, "lidar_hz": 0.0, "cam_fps": 0.0,
        }
        self._t_prev_step  = None      # perf_counter at previous step()
        self._rate_t0      = None      # rate-window start (perf_counter)
        self._lidar_frames = 0         # fresh LiDAR scans seen this window
        self._cam_frames   = 0         # fresh camera frames seen this window
        # `extract_segments` (line-fit for the dashboard overlay) is ~8 ms and
        # dashboard-only (A*/nav use wall_status, not the segments), so run it
        # every Nth scan, not every scan; keeps the tick rate up.
        self._scan_count     = 0
        self._segments_every = 3
        self._last_cmd_wheels = (0.0, 0.0)   # last (v_l, v_r) commanded

        print(f"Waiter bot started: hybrid A* + reactive controller.")
        print(f"  grid {config.grid.dim}x{config.grid.dim} @ "
              f"{config.grid.cell_m*100:.0f} cm")
        if config.tuning.telemetry_port > 0:
            telemetry.start(port=config.tuning.telemetry_port)

    # Public
    def step(self) -> None:
        """Advance one tick of the FSM."""
        now = time.perf_counter()
        if self._t_prev_step is not None:
            dt = now - self._t_prev_step
            if dt > 0:
                self._perf["tick_hz"] = self._ema(self._perf["tick_hz"], 1.0 / dt)
        self._t_prev_step = now

        self.world.tick += 1
        self._update_odometry()
        t0 = time.perf_counter()
        self._run_perception()
        self._perf["perception_ms"] = self._ema(
            self._perf["perception_ms"], (time.perf_counter() - t0) * 1000.0)
        self._run_state()
        if self.cfg.tuning.telemetry_port > 0:
            self._publish_telemetry()

    @staticmethod
    def _ema(old: float, new: float, alpha: float = 0.2) -> float:
        """Exponential moving average for a stable dashboard readout."""
        return new if old == 0.0 else (1.0 - alpha) * old + alpha * new

    # Odometry from wheel encoders + IMU
    # Wheel encoders alone are sufficient for x/y position but suffer
    # from slip-induced yaw drift; every in-place rotation loses heading
    # information. When an IMU is present, we OVERRIDE the integrated yaw
    # with the absolute-yaw reading (zero-offset against the first sample
    # so the bot's frame stays anchored at "starting heading = 0"). x/y
    # are still encoder-derived, but the integration midpoint uses the
    # IMU-corrected yaw, so heading errors no longer compound into
    # position errors during rotation.
    def _update_odometry(self) -> None:
        enc = self.sensors.get_encoders()
        imu = self.sensors.get_imu()

        # Initialise IMU yaw offset on the first reading.
        if imu is not None and self.world.imu_yaw_offset is None:
            self.world.imu_yaw_offset = imu.yaw

        if self.world.prev_left_rad is not None:
            d_l = (enc.left_rad - self.world.prev_left_rad) * self.cfg.robot.wheel_radius_m
            d_r = (enc.right_rad - self.world.prev_right_rad) * self.cfg.robot.wheel_radius_m
            d_s   = (d_l + d_r) / 2.0

            if imu is not None and self.world.imu_yaw_offset is not None:
                # Absolute, drift-free yaw from the IMU.
                new_yaw = wrap_angle(imu.yaw - self.world.imu_yaw_offset)
                # Wrap-safe angle midpoint: half-step from old toward new.
                # `(old + new)/2` fails near +-pi (e.g. +179° & -179° -> 0°
                # instead of 180°); shortest-path average avoids that.
                d_yaw_imu = wrap_angle(new_yaw - self.world.bot_yaw)
                mid       = wrap_angle(self.world.bot_yaw + d_yaw_imu / 2.0)
                self.world.bot_x  += d_s * math.cos(mid)
                self.world.bot_y  += d_s * math.sin(mid)
                self.world.bot_yaw = new_yaw
                d_yaw = d_yaw_imu
            else:
                # Encoder-only fallback (drifts on slip; see Build Log 17).
                d_yaw = (d_r - d_l) / self.cfg.robot.wheel_base_m
                mid   = self.world.bot_yaw + d_yaw / 2.0
                self.world.bot_x   += d_s * math.cos(mid)
                self.world.bot_y   += d_s * math.sin(mid)
                self.world.bot_yaw  = wrap_angle(self.world.bot_yaw + d_yaw)

            # Track odometry motion so the ICP correction can tell a moving
            # bot (drift worth correcting) from a parked one (nothing to fix).
            self._odom_trans_since_scan += abs(d_s)
            self._odom_rot_since_scan   += abs(d_yaw)

        self.world.prev_left_rad  = enc.left_rad
        self.world.prev_right_rad = enc.right_rad

    # Perception: near-field walls + cup
    # Camera-based perception (ArUco, wall projection, near-field
    # clearances) is the most expensive part of the per-tick budget,
    # and the Webots camera only re-renders every CAMERA_INTERVAL_TICKS
    # ticks anyway (see WebotsSensorSource), so we only do that work
    # at the same cadence.
    PERCEPTION_INTERVAL_TICKS = 4

    def _run_perception(self) -> None:
        # LiDAR: pull at the device's native cadence. The adapter
        # returns None on ticks where no fresh scan has come in, so this
        # is essentially free between scans. Stored on WorldState for
        # future consumers (ICP / SLAM); FSM logic doesn't read it yet.
        lidar = self.sensors.get_lidar()
        if lidar is not None:
            self._lidar_frames += 1        # dashboard rate counter
            self.world.lidar_scan      = lidar
            # Snapshot pose at scan-capture time. Ray endpoints get
            # transformed using this pose, NOT the bot's current pose;
            # otherwise the dot cloud smears as the bot moves between
            # the 5.5 Hz scans.
            self.world.lidar_scan_pose = (self.world.bot_x,
                                           self.world.bot_y,
                                           self.world.bot_yaw)
            # Scan-to-map drift correction: align this scan to the
            # accumulated map and apply a damped fraction of the solved
            # correction to both the scan pose (so THIS scan stamps in
            # the right place) and the live odometry (so drift doesn't
            # re-accumulate). With the anchor gone this scan-match is the
            # bot's ONLY drift correction; runs once the map has enough
            # structure to register against.
            moved = (self._odom_trans_since_scan > LIDAR_ICP_MIN_TRANS_M
                     or self._odom_rot_since_scan > LIDAR_ICP_MIN_ROT_RAD)
            if (LIDAR_ICP_ENABLED and moved
                    and int(self.world.wall_status.sum()) >= LIDAR_ICP_MIN_WALL_CELLS):
                mr = match_scan(self.world.wall_status, lidar,
                                self.world.lidar_scan_pose, self.cfg.grid)
                if mr.accepted:
                    sx, sy, syaw = self.world.lidar_scan_pose
                    g    = LIDAR_ICP_GAIN
                    dx   = g * (mr.pose[0] - sx)
                    dy   = g * (mr.pose[1] - sy)
                    dyaw = g * wrap_angle(mr.pose[2] - syaw)
                    self.world.lidar_scan_pose = (sx + dx, sy + dy,
                                                  wrap_angle(syaw + dyaw))
                    self.world.bot_x  += dx
                    self.world.bot_y  += dy
                    self.world.bot_yaw = wrap_angle(self.world.bot_yaw + dyaw)
                    if self.world.tick - self._lidar_icp_last_log_tick \
                            >= LIDAR_ICP_LOG_EVERY_TICKS:
                        self._lidar_icp_last_log_tick = self.world.tick
                        print(f"  [t={self.world.tick}] LiDAR ICP: corrected "
                              f"{mr.corr_trans_m*100:.1f}cm/"
                              f"{math.degrees(mr.corr_rot_rad):.1f}° drift "
                              f"(applied {g:.0%}, resid "
                              f"{mr.mean_resid_m*1000:.1f}mm)")
            # Motion is measured between consecutive scans; reset now that
            # this scan has been consumed (whether or not ICP ran).
            self._odom_trans_since_scan = 0.0
            self._odom_rot_since_scan   = 0.0
            if not self._lidar_first_read_logged:
                self._lidar_first_read_logged = True
                finite = lidar.ranges[np.isfinite(lidar.ranges)]
                closest = float(finite.min()) if finite.size else math.inf
                print(f"  [t={self.world.tick}] LiDAR online: "
                      f"{lidar.ranges.size} rays, closest={closest:.2f}m")
                print(f"    angle_min={lidar.angle_min:+.3f} rad "
                      f"({math.degrees(lidar.angle_min):+.1f}°)  "
                      f"angle_inc={lidar.angle_increment:.5f} rad "
                      f"({math.degrees(lidar.angle_increment):.3f}°/ray)")
            # Per-scan diagnostic. Three things to verify in one line:
            #   1. raw IMU yaw vs derived bot_yaw vs the offset; if
            #      bot_yaw lags imu.yaw, the offset is wrong / drifting.
            #   2. world hit position of ray 180 (body-forward): should
            #      sweep around the walls as the bot rotates if the
            #      transform is correct.
            #   3. world hit position of ray 0 (body-BACKWARD): should
            #      also sweep, but always 180° opposite of ray 180. If
            #      ray 0 and ray 180 disagree (e.g., both point at the
            #      same wall), the local-angle convention is wrong.
            self._lidar_debug_scan_count += 1
            if self._lidar_debug_scan_count % 4 == 0:
                imu_now = self.sensors.get_imu()
                imu_yaw_now = imu_now.yaw if imu_now is not None else float('nan')
                offset = self.world.imu_yaw_offset
                n_rays = lidar.ranges.size
                def _ray_hit(i):
                    r = float(lidar.ranges[i])
                    if not math.isfinite(r) or r <= 0.05:
                        return None, None, r
                    la = lidar.angle_min + i * lidar.angle_increment
                    wa = self.world.bot_yaw + la
                    return (self.world.bot_x + r * math.cos(wa),
                            self.world.bot_y + r * math.sin(wa), r)
                fx, fy, fr = _ray_hit(n_rays // 2)              # ray 180 (body +X)
                bx, by, br = _ray_hit(0)                          # ray 0   (body -X)
                lx, ly, lr = _ray_hit(int(n_rays * 0.75))        # ray 270 (body +Y)
                rx, ry, rr = _ray_hit(int(n_rays * 0.25))        # ray 90  (body -Y)
                _f = lambda v: f"{v:+.2f}" if v is not None else "  inf"
                print(f"  [t={self.world.tick}] LIDAR DEBUG  "
                      f"imu={math.degrees(imu_yaw_now):+.1f}° "
                      f"offset={math.degrees(offset) if offset is not None else 0.0:+.1f}° "
                      f"bot_yaw={math.degrees(self.world.bot_yaw):+.1f}°")
                print(f"    body+X ray180  r={fr:.2f}  world=({_f(fx)},{_f(fy)})")
                print(f"    body-X ray  0  r={br:.2f}  world=({_f(bx)},{_f(by)})")
                print(f"    body+Y ray270  r={lr:.2f}  world=({_f(lx)},{_f(ly)})")
                print(f"    body-Y ray 90  r={rr:.2f}  world=({_f(rx)},{_f(ry)})")
            # Spinning-LiDAR motion de-skew (hardware only). Estimate the bot's
            # motion across THIS sweep as the signed delta between consecutive
            # scan-snapshot poses (~one revolution apart) and hand it to
            # ingest_lidar so it interpolates the pose per ray. Updated every
            # scan (even when ingest is skipped) so the delta always spans one
            # sweep; a delta too big for a single sweep (pose teleport from
            # move-bot/reset, or a multi-sweep gap after RECALIBRATE) is
            # rejected so it can't fling the scan. In Webots the LiDAR samples a
            # whole scan at one instant, so _lidar_deskew is False and every ray
            # uses the single snapshot pose.
            sweep_motion = None
            cur_pose = self.world.lidar_scan_pose
            if self._lidar_deskew and self._prev_scan_pose is not None:
                px, py, pyaw = self._prev_scan_pose
                m = (cur_pose[0] - px, cur_pose[1] - py,
                     wrap_angle(cur_pose[2] - pyaw))
                if (abs(m[2]) <= LIDAR_DESKEW_MAX_ROT_RAD
                        and math.hypot(m[0], m[1]) <= LIDAR_DESKEW_MAX_TRANS_M):
                    sweep_motion = m
            self._prev_scan_pose = cur_pose

            # Apply this scan to the wall map: stamp hits as walls,
            # carve free space along every ray. See
            # `perception.lidar_walls` for the algorithm. On the Webots Lidar
            # (`type "fixed"`) every ray in a scan is sampled at the same
            # instant, so the single pose-at-scan-time snapshot is exact; on the
            # spinning D500, sweep_motion de-skews it (above).
            # Total wall cells before ingest, so we can see whether
            # each scan grows the map (still discovering) or churns
            # it (re-stamping carved cells in a steady-state loop).
            cells_before = int(self.world.wall_status.sum())
            # Collect a handful of carved cells per scan to
            # investigate exactly what's getting cleared (and by
            # which ray). The list is filled in-place by
            # ingest_lidar and capped at 6 entries.
            carved_samples: list = []
            n_hits, n_carved = ingest_lidar(
                self.world, lidar, self.world.lidar_scan_pose,
                self.cfg.grid, max_range_m=LIDAR_MAX_RANGE_M,
                sample_carved=carved_samples,
                sweep_motion=sweep_motion,
                deskew_dir=LIDAR_DESKEW_DIR,
                deskew_phase=LIDAR_DESKEW_PHASE,
            )
            cells_after = int(self.world.wall_status.sum())
            # Post-ingest housekeeping: these used to live at the
            # end of `_roll_scan` (deleted along with the camera
            # pipeline). They run after every wall-map update so
            # the bot's footprint stays clear of phantoms and
            # `wall_segments` (dashboard + future ICP) stays in
            # sync with `wall_status`.
            self._clear_walls_under_body()
            # Dashboard-only line overlay: throttled to every Nth scan
            # (keeps the last result between updates) to save ~8 ms/scan.
            self._scan_count += 1
            if self._scan_count % self._segments_every == 0:
                self.world.wall_segments = extract_segments(
                    self.world.wall_status,
                    self.cfg.grid.cell_m,
                    self.cfg.grid.half_m,
                )
            if (n_hits or n_carved) and \
               self.world.tick - self._lidar_last_ingest_log_tick \
               >= LIDAR_INGEST_LOG_EVERY_TICKS:
                self._lidar_last_ingest_log_tick = self.world.tick
                delta = cells_after - cells_before
                print(f"  [t={self.world.tick}] LiDAR ingest: "
                      f"{n_hits} new hits, {n_carved} carved, "
                      f"map: {cells_before} -> {cells_after} "
                      f"(Δ {delta:+d})")
                cm = self.cfg.grid.cell_m
                hm = self.cfg.grid.half_m
                for c in carved_samples:
                    wx = (c["gx"] + 0.5) * cm - hm
                    wy = (c["gy"] + 0.5) * cm - hm
                    print(f"      cleared cell=({c['gx']},{c['gy']}) "
                          f"world=({wx:+.2f},{wy:+.2f})  "
                          f"vote dropped to {c['vote']}")

        # Camera-based work runs only every PERCEPTION_INTERVAL_TICKS;
        # the Webots Camera re-renders at the same cadence (enabled at
        # 4× basicTimeStep), so processing on every tick would mean
        # decoding the same buffer 4 times in a row.
        # Grab the freshest frame EVERY tick; cheap, since RemoteSensorSource
        # has already decoded it and this just hands over the cached array.
        # Keeping `_last_rgb` live at the link's full rate means the dashboard
        # /frame.jpg shows a current image instead of one gated to the slow
        # every-4-tick perception cadence (that gate is what pinned the
        # camera panel to ~4 fps). The heavy ArUco / cup detection below still
        # runs only every PERCEPTION_INTERVAL_TICKS; navigation doesn't need
        # it faster, and ArUco is the ~5 ms cost we don't want every tick.
        rgb = self.sensors.get_camera_frame()
        if rgb is not None:
            self._cam_frames += 1          # dashboard rate counter
            self._last_rgb = rgb

        if self.world.tick % self.PERCEPTION_INTERVAL_TICKS != 0:
            return
        if self._last_rgb is None:
            return
        rgb = self._last_rgb               # detect on the freshest frame

        # Near-field clearances: drives AVOID decisions (body frame, no
        # odometry dependency, so wheel slip doesn't degrade it).
        self._near = near_field_clearances(rgb, self.cfg.camera, near_range_m=0.40)

        # Camera-projection wall mapping was removed; LiDAR
        # (perception/lidar_walls.py) now owns wall_status updates,
        # called once per fresh scan at the top of this method.
        # Camera still drives near-field clearances (above), ArUco
        # detection (below), and the dashboard overlay.

        # Run the raw detector once so we have ALL visible quads to
        # draw on the telemetry frame; the wrapper detect() does the
        # same work inside (and re-uses the result through OpenCV's
        # internal caching). Storing every detection, not just the
        # biggest, lets the overlay show that multi-face fusion is
        # really seeing all faces.
        _gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _corners, _ids, _ = self._aruco._detector.detectMarkers(_gray)
        if _ids is not None and len(_ids) > 0:
            self._last_aruco_detections = [
                (_corners[i], int(_ids[i][0])) for i in range(len(_corners))
            ]
        else:
            self._last_aruco_detections = []

        # Cup detection via ArUco PnP, smoothed in world frame against the
        # live pose. With the anchor gone, the LiDAR scan-match keeps the pose
        # accurate, so the body->world projection is trustworthy on every frame
        # and the cup just rides the smoothed estimate (no freshness gate).
        pose = self._aruco.detect(rgb)
        self._last_pose = pose

        # If we lost the marker this frame, blow away the verification
        # buffer; we want N CONSECUTIVE detections, not N total ever.
        if pose is None:
            self._pending_detections = []
            return

        cy_ = math.cos(self.world.bot_yaw)
        sy_ = math.sin(self.world.bot_yaw)
        wx = self.world.bot_x + pose.forward * cy_ - pose.lateral * sy_
        wy = self.world.bot_y + pose.forward * sy_ + pose.lateral * cy_

        # Two-scan verification
        # Before the first cup sighting is accepted, require
        # ARUCO_VERIFY_FRAMES consecutive frames that AGREE on world
        # position (within ARUCO_VERIFY_RADIUS_M). One false-positive
        # decode is unlikely to repeat at the same world coordinates.
        if self.world.glass_xy_acc is None:
            self._pending_detections.append((wx, wy))
            # Trim to the last N entries
            if len(self._pending_detections) > ARUCO_VERIFY_FRAMES:
                self._pending_detections = \
                    self._pending_detections[-ARUCO_VERIFY_FRAMES:]
            # Not enough samples yet; keep collecting
            if len(self._pending_detections) < ARUCO_VERIFY_FRAMES:
                return
            # Check all N samples mutually agree
            mean_x = sum(p[0] for p in self._pending_detections) / ARUCO_VERIFY_FRAMES
            mean_y = sum(p[1] for p in self._pending_detections) / ARUCO_VERIFY_FRAMES
            max_dev = max(math.hypot(p[0] - mean_x, p[1] - mean_y)
                          for p in self._pending_detections)
            if max_dev > ARUCO_VERIFY_RADIUS_M:
                # Samples disagree; likely false positive. Drop the
                # OLDEST sample, keep collecting; effectively a sliding
                # window check.
                self._pending_detections.pop(0)
                return
            # Accept!
            self.world.glass_xy_acc = (mean_x, mean_y)
            print(f"  [t={self.world.tick}] first cup sighting VERIFIED "
                  f"({ARUCO_VERIFY_FRAMES} frames agree within "
                  f"{max_dev*100:.1f}cm)  world=({mean_x:+.2f},{mean_y:+.2f})  "
                  f"marker_id={pose.marker_id}")
            return

        # Subsequent updates: reject outliers, smooth the rest.
        ax, ay = self.world.glass_xy_acc
        if math.hypot(wx - ax, wy - ay) > 0.40:
            return
        self.world.glass_xy_acc = (
            (1 - CUP_ALPHA) * ax + CUP_ALPHA * wx,
            (1 - CUP_ALPHA) * ay + CUP_ALPHA * wy,
        )

    # State dispatch
    def _run_state(self) -> None:
        # Dashboard mode override (mode buttons). "pause" halts the FSM: zero
        # every drive command (so the SEARCH spin stops) while the ESP keeps the
        # bot balancing. Checked FIRST so it overrides search, map goals, and
        # return-home alike. "auto" (default) falls through to the normal mission.
        if telemetry.get_mode() == "pause":
            self._drive(0.0, 0.0)
            return

        # "Return home" (UI button): drive back to where the bot set off from
        # and park. Forget the current target so we don't immediately re-chase
        # it. Wins over everything below; one-shot arms it, _returning holds it.
        if telemetry.get_return_request() and self._home_xy is not None:
            self._returning = True
            self.world.glass_xy_acc = None
            telemetry.set_goal_state(None, None, active=False)
            print(f"[t={self.world.tick}] RETURN HOME requested -> "
                  f"({self._home_xy[0]:+.2f},{self._home_xy[1]:+.2f})")
        if self._returning:
            hx, hy = self._home_xy
            if math.hypot(hx - self.world.bot_x,
                          hy - self.world.bot_y) < GOTO_REACH_M:
                self._drive(0.0, 0.0)
                self.sensors.set_leds(True)
                self._returning = False
                self._home_xy = None
                self._was_driving = False
                self.world.state = FSMState.DONE
                print(f"[t={self.world.tick}] returned home; parked")
                return
            self.world.state = FSMState.SEEK          # display: en route
            self._state_goto(hx, hy)
            return

        # Map-goal override: when a goal is set (dashboard click or the palm-
        # follow carrot), drive straight to it with the goto stack, suspending
        # the cup hunt. The cup uses the very same stack (see _state_seek).
        goal = telemetry.get_goal_state()
        has_goal = bool(goal.get("active") and goal.get("x") is not None)

        # Remember where we set off from, captured the moment we START driving
        # to a target (goal or cup) from rest — that's the spot "return home"
        # brings us back to. Held until we actually get home again.
        driving = has_goal or (self.world.glass_xy_acc is not None)
        if driving and not self._was_driving and self._home_xy is None:
            self._home_xy = (self.world.bot_x, self.world.bot_y)
            print(f"[t={self.world.tick}] home set "
                  f"({self._home_xy[0]:+.2f},{self._home_xy[1]:+.2f})")
        self._was_driving = driving

        if has_goal:
            self._state_goto(float(goal["x"]), float(goal["y"]))
            return

        s = self.world.state
        if   s is FSMState.SEEK: self._state_seek()
        elif s is FSMState.DONE: self._state_done()
        else:                    self._state_search()   # SEARCH (or any legacy)

    # SEARCH
    def _state_search(self) -> None:
        """Spin in place until ArUco picks up the cup."""
        if self.world.glass_xy_acc is not None:
            print(f"[t={self.world.tick}] SEARCH -> SEEK  "
                  f"cup at ({self.world.glass_xy_acc[0]:+.2f},"
                  f"{self.world.glass_xy_acc[1]:+.2f})")
            self.world.state = FSMState.SEEK
            self.world.last_log = ""
            return
        self._drive(-SEARCH_SPIN_RAD_S, +SEARCH_SPIN_RAD_S)
        self.sensors.set_leds(False)
        if self.world.tick % 30 == 0:
            print(f"  [t={self.world.tick}] SEARCH  yaw={math.degrees(self.world.bot_yaw):+.0f}°  "
                  f"near={self._near}  cup=none")

    def _state_seek(self) -> None:
        """Drive to the detected cup, then stop. Uses the SAME goto stack the
        dashboard / palm-follow goals use (A* + DWA + near-field guard) — no
        fiducials, no anchor, no fine-docking phase. Lose the cup -> back to
        SEARCH; reach it -> DONE (stop + LEDs on)."""
        cup = self.world.glass_xy_acc
        if cup is None:
            self.world.state = FSMState.SEARCH
            self.world.last_log = ""
            return
        gx_w, gy_w = cup
        dist = math.hypot(gx_w - self.world.bot_x, gy_w - self.world.bot_y)
        if dist < GOTO_REACH_M:
            self._drive(0.0, 0.0)
            self.sensors.set_leds(True)
            self.world.state = FSMState.DONE
            self.world.last_log = ""
            print(f"[t={self.world.tick}] SEEK -> DONE  arrived at cup "
                  f"({gx_w:+.2f},{gy_w:+.2f})  d={dist*100:.1f}cm")
            return
        self._state_goto(gx_w, gy_w)

    def _body_disk_mask(self, radius_m: float) -> tuple:
        """Return (i_slice, j_slice, inside_bool) describing the grid
        cells whose centres lie within `radius_m` of the current bot
        pose. Used by both `_clear_walls_under_body` (on the raw map)
        and `_replan` (on the inflated grid). Cheap: only scans a
        small bounding box around the bot, not the full grid."""
        cell_m = self.cfg.grid.cell_m
        half_m = self.cfg.grid.half_m
        bx, by = self.world.bot_x, self.world.bot_y
        r_cells = int(math.ceil(radius_m / cell_m))
        gx0, gy0 = world_to_grid(bx, by, self.cfg.grid)

        dim_x, dim_y = self.world.wall_status.shape
        i0 = max(0,     gx0 - r_cells)
        i1 = min(dim_x, gx0 + r_cells + 1)
        j0 = max(0,     gy0 - r_cells)
        j1 = min(dim_y, gy0 + r_cells + 1)
        if i0 >= i1 or j0 >= j1:
            return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool)

        ii = np.arange(i0, i1).reshape(-1, 1)
        jj = np.arange(j0, j1).reshape(1, -1)
        wx = (ii + 0.5) * cell_m - half_m
        wy = (jj + 0.5) * cell_m - half_m
        inside = (wx - bx) ** 2 + (wy - by) ** 2 <= radius_m * radius_m
        return slice(i0, i1), slice(j0, j1), inside

    def _carve_body_footprint(self, grid: np.ndarray) -> None:
        """Clear the (body_radius + inflate_m) disk under the bot, in place,
        from an INFLATED grid. Guarantees the bot's start cell has an exit
        move even when it sits inside a wall's inflation halo. Shared by
        `_replan` and `_probe_plan` so both plan on an identically-carved
        grid."""
        i_slc, j_slc, inside = self._body_disk_mask(
            BOT_BODY_RADIUS_M + self.cfg.grid.inflate_m)
        if inside.any():
            sub = grid[i_slc, j_slc]
            sub[inside & sub] = False

    def _clear_wall_cells(self, mask: np.ndarray) -> None:
        """Reset every per-cell wall-map array for the True cells of `mask`.
        Single clearing point so a future array can't be silently skipped
        by one of the several clear sites (WIPE / SHIFT / body)."""
        self.world.wall_status[mask]          = False
        self.world.wall_status_tof[mask]      = False
        self.world.wall_discovered_tick[mask] = -1
        self.world.wall_absent_count[mask]    = 0
        self.world.wall_vote[mask]            = 0

    def _clear_walls_under_body(self) -> None:
        """Wipe wall_status cells inside the bot's physical footprint.

        Body-presence rule: the bot is sitting at (bot_x, bot_y) and
        hasn't crashed, so no real wall can exist within its body
        radius. Any wall mark there is a projection phantom; clear
        wall_status, the wall_status_tof flag, the discovery tick, and
        the absent-evidence counter so the cell is fully reset.

        Without this, phantom walls under the bot stay sticky:
        wall_status is one-way ADD via voting and only clears via
        negative-evidence accumulation, which can take seconds. In the
        meantime A* refuses to plan because the start cell sits inside a
        phantom wall whose inflation closes every exit.
        """
        ws = self.world.wall_status
        if ws is None:
            return
        i_slc, j_slc, inside = self._body_disk_mask(BODY_CLEAR_RAW_M)
        if not inside.any():
            return
        clear_mask = ws[i_slc, j_slc] & inside
        n = int(clear_mask.sum())
        if n == 0:
            return
        full = np.zeros_like(ws)
        full[i_slc, j_slc] = clear_mask
        self._clear_wall_cells(full)
        print(f"  [t={self.world.tick}] BODY CLEAR: wiped {n} phantom "
              f"wall cell(s) under bot at "
              f"({self.world.bot_x:+.2f},{self.world.bot_y:+.2f}) "
              f"(r={BODY_CLEAR_RAW_M*100:.1f}cm)")

    def _replan(self, gx_w: float, gy_w: float) -> None:
        """Rebuild the A* path from current bot position to the cup.

        On failure, prints a structured diagnostic explaining why the
        planner couldn't find a route. The raw wall map visible on the
        dashboard is what the bot has projected; the planner runs on
        the INFLATED grid (every wall dilated by inflate_m = 7 cm,
        ~3 cells). Gaps that look open in the raw view can be closed
        in the inflated grid, which is the most common reason for
        "I see a clear path, why can't A* find it?".
        """
        # Body-presence rule: scrub any phantom wall cells the bot is
        # currently sitting inside BEFORE we inflate. Otherwise the
        # phantom inflates outward by inflate_m and walls off every
        # neighbouring cell, leaving A* with no exit move from start.
        self._clear_walls_under_body()

        occ = self.world.wall_status
        grid = inflate_grid(occ, self.cfg.grid.inflate_m, self.cfg.grid)
        # Carve a (body_radius + inflate_m) disk out of the INFLATED
        # grid. The raw clear handled phantoms INSIDE the body; this
        # handles legitimate walls just OUTSIDE the body whose inflation
        # halo reaches into where the bot is physically sitting. The
        # radius has to span the FULL halo width, not just the body;
        # if the bot is 5 cm from a real wall (4 cm inside its 7 cm
        # halo), a body-only hole still leaves halo cells blocking
        # lateral exit moves (the original A*-can't-plan symptom one
        # level deeper). (body_radius + inflate_m) is the max distance
        # from bot center at which any inflated cell can belong to a
        # wall the bot is inside the buffer of, so clearing that disk
        # guarantees an exit direction exists. wall_status is untouched
        # (real walls remain; AVOID/DWA stop the bot from grinding in).
        self._carve_body_footprint(grid)
        # Cache for DWA's rollout collision check; same dilation as A*
        # so the local planner can never pick a (v, ω) leading into a
        # cell A* refuses to plan through.
        self._inflated_grid = grid
        start, start_clamped = world_to_grid_checked(
            self.world.bot_x, self.world.bot_y, self.cfg.grid)
        goal, goal_clamped = world_to_grid_checked(
            gx_w, gy_w, self.cfg.grid)
        # Flag clamping loudly; silently routing to the grid edge when
        # the cup is outside the planning area is the kind of bug that
        # eats days. If you see this, EITHER the grid is too small
        # (bump GridConfig.half_m) OR the cup detection has drifted
        # well outside the operational area.
        if goal_clamped:
            print(f"  [t={self.world.tick}] WARN: goal ({gx_w:+.2f},{gy_w:+.2f}) "
                  f"is OUTSIDE grid (half_m={self.cfg.grid.half_m:.1f}m); "
                  f"A* will route to clamped edge cell {goal} instead of "
                  f"actual cup. Bump GridConfig.half_m or re-examine cup "
                  f"detection.")
        if start_clamped:
            print(f"  [t={self.world.tick}] WARN: bot ({self.world.bot_x:+.2f},"
                  f"{self.world.bot_y:+.2f}) is OUTSIDE grid; odometry "
                  f"has wandered past the planning area.")
        start_was_blocked = bool(grid[start[0], start[1]])
        if start_was_blocked:
            grid[start[0], start[1]] = False
        goal_orig = goal
        goal      = nearest_free(grid, *goal)
        goal_was_moved = (goal != goal_orig)

        # Soft clearance cost: bias the route toward the middle of free
        # space rather than grazing the inflation boundary. None when the
        # feature is disabled (clearance_weight == 0): plain A*.
        cost_field = clearance_cost_field(grid, self.cfg.grid)
        _t_plan = time.perf_counter()
        path = a_star(grid, start, goal, cost_field=cost_field)
        self._perf["plan_ms"] = self._ema(
            self._perf["plan_ms"], (time.perf_counter() - _t_plan) * 1000.0)
        if path is None:
            # Diagnose. Most failures fall into one of these buckets:
            #   1. Start cell sat inside the INFLATED grid (bot too
            #      close to a wall after inflation). nearest_free is
            #      only applied to the goal, so the start is just
            #      forcibly unmarked; but if it's surrounded by
            #      inflated walls, there's no exit move.
            #   2. Goal cell was inside an obstacle and nearest_free
            #      bounced it to a cell that's still in an unreachable
            #      component.
            #   3. The map has confirmed walls that close off the
            #      whole start-to-goal corridor (often phantom walls
            #      inside the corridor itself).
            raw_walls  = int(occ.sum())
            infl_walls = int(grid.sum())
            # Flood-fill from start: which cells are reachable?
            reachable = self._flood_fill(grid, start)
            n_reach = int(reachable.sum())
            goal_reachable = bool(reachable[goal[0], goal[1]])
            # Find the nearest reachable cell to the original goal
            # ("the planner could have gotten you here instead, but
            # the goal you asked for is walled off").
            nearest_reach = nearest_free(~reachable, *goal_orig)
            dxg = (nearest_reach[0] - goal_orig[0]) * self.cfg.grid.cell_m
            dyg = (nearest_reach[1] - goal_orig[1]) * self.cfg.grid.cell_m
            d_to_reach = math.hypot(dxg, dyg)

            # Build a compact one-line summary so we can both throttle
            # repeated identical failures AND surface them inside the
            # survey log so the failure mode is visible to anyone
            # scanning the bottom of the console buffer.
            self._last_astar_fail_summary = (
                f"start={start} blocked={start_was_blocked} "
                f"goal={goal} (orig {goal_orig}"
                f"{'->moved' if goal_was_moved else ''}) "
                f"walls={raw_walls}(+{infl_walls - raw_walls} infl) "
                f"reach={n_reach} "
                f"goal_reach={'Y' if goal_reachable else 'N'}"
            )

            # Throttle the full multi-line diagnostic to ~1/sec.
            ticks_since_last = self.world.tick - self._last_astar_fail_log_tick
            if ticks_since_last >= 60:
                self._last_astar_fail_log_tick = self.world.tick
                print(f"  [t={self.world.tick}] A* FAILED: "
                      f"start={start} goal={goal} "
                      f"(orig {goal_orig}{'->moved' if goal_was_moved else ''}, "
                      f"start_blocked={start_was_blocked})")
                print(f"    raw walls: {raw_walls}  inflated walls: {infl_walls}  "
                      f"({infl_walls - raw_walls} cells added by inflation)")
                print(f"    reachable from start: {n_reach} cells; "
                      f"goal {'IS' if goal_reachable else 'is NOT'} in that set")
                if not goal_reachable:
                    print(f"    nearest reachable cell to original goal "
                          f"is {nearest_reach}, "
                          f"d={d_to_reach*100:.1f}cm away; try reducing "
                          f"inflate_m (current {self.cfg.grid.inflate_m*100:.0f}cm) "
                          f"or surveying to fill in unmapped corridor.")
                else:
                    # Reachable but A* still failed; shouldn't happen.
                    print(f"    (BUG) goal in reachable set but A* returned None")

            # No route: start blocked by inflation, or boxed in by inflated
            # walls. Drop the plan and wait; the goto driver holds position
            # until a later replan finds a way (the reverse-escape went with
            # the AVOID machinery).
            self.world.path = None
            return

        # Any-angle smoothing: greedy line-of-sight shortcutting over the
        # inflated grid turns A*'s 45°-quantised staircase into straight
        # diagonal runs. Safe because LOS is checked on the inflated grid,
        # so body clearance is preserved. Supersedes the old collinear-only
        # `smooth_path` approach.
        self.world.path = string_pull(grid, path)
        self.world.path_idx = 1 if len(self.world.path) > 1 else 0

    def _probe_plan(self, infl: np.ndarray) -> Optional[dict]:
        """Non-committal A* for the dashboard goal probe.

        Plans from the bot to the dashboard-clicked point on the same
        inflated grid + clearance cost the autonomous planner uses, and
        reports the route plus the cells A* explored. Purely for
        visualisation; never touches the motors or `self.world.path`.
        Returns None when no goal is set; otherwise a dict consumed by
        the telemetry snapshot + map render.
        """
        gs = telemetry.get_goal_state()
        if not gs.get("active") or gs.get("x") is None or gs.get("y") is None:
            return None
        gx_w, gy_w = float(gs["x"]), float(gs["y"])

        # Copy so carving the bot's footprint doesn't alter the halo the map
        # render shows, then carve the same disk the real planner does, so
        # the probe plans on an identically-prepared grid (see _replan).
        grid = infl.copy()
        self._carve_body_footprint(grid)

        start, _ = world_to_grid_checked(
            self.world.bot_x, self.world.bot_y, self.cfg.grid)
        goal, _ = world_to_grid_checked(gx_w, gy_w, self.cfg.grid)
        if grid[start[0], start[1]]:
            grid[start[0], start[1]] = False
        goal = nearest_free(grid, *goal)

        cost_field = clearance_cost_field(grid, self.cfg.grid)
        _t_plan = time.perf_counter()
        path, explored = a_star(grid, start, goal, cost_field=cost_field,
                                collect_explored=True)
        self._perf["plan_ms"] = self._ema(
            self._perf["plan_ms"], (time.perf_counter() - _t_plan) * 1000.0)

        explored_mask = np.zeros_like(grid)
        if explored:
            idx = np.asarray(explored, dtype=np.intp)      # (N, 2)
            explored_mask[idx[:, 0], idx[:, 1]] = True

        probe_path = string_pull(grid, path) if path else None
        return {
            "goal_world":    (gx_w, gy_w),
            "probe_path":    probe_path,
            "explored_mask": explored_mask,
            "reachable":     path is not None,
            "path_len":      len(probe_path) if probe_path else 0,
            "n_explored":    len(explored),
        }

    @staticmethod
    def _flood_fill(grid: np.ndarray, start: tuple) -> np.ndarray:
        """Boolean mask of all cells reachable from `start`.

        8-connected with the same diagonal-squeeze rule as A* (no
        cutting between two blocked orthogonal neighbours), so the
        reachable set this returns is exactly what A* can plan over.
        """
        dim_x, dim_y = grid.shape
        reachable = np.zeros_like(grid, dtype=bool)
        sx, sy = start
        if not (0 <= sx < dim_x and 0 <= sy < dim_y) or grid[sx, sy]:
            return reachable
        stack = [(sx, sy)]
        reachable[sx, sy] = True
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < dim_x and 0 <= ny < dim_y):
                        continue
                    if reachable[nx, ny] or grid[nx, ny]:
                        continue
                    if dx != 0 and dy != 0:
                        if grid[x + dx, y] or grid[x, y + dy]:
                            continue
                    reachable[nx, ny] = True
                    stack.append((nx, ny))
        return reachable

    def _state_done(self) -> None:
        self._drive(0, 0)
        self.sensors.set_leds(True)

    # GOTO — drive to a dashboard-set map goal (manual "go here")
    def _state_goto(self, gx_w: float, gy_w: float) -> None:
        """Route to a map goal with the SAME A* + DWA + drive stack SEEK uses,
        but self-contained: no CENTRE/RECAL/cup coupling, so it works in a bare
        room with no fiducials. On arrival it holds position until the goal is
        moved or cleared from the dashboard.
        """
        dist = math.hypot(gx_w - self.world.bot_x, gy_w - self.world.bot_y)

        # Arrived: stop and hold (don't auto-clear — let the user move/clear it).
        if dist < GOTO_REACH_M:
            self._drive(0.0, 0.0)
            self.world.path = None
            if self.world.tick % 30 == 0:
                print(f"  [t={self.world.tick}] GOTO arrived "
                      f"(d={dist*100:.0f}cm) - holding")
            return

        # Reactive near-field guard: a wall dead ahead the inflated map may not
        # carry yet -> turn in place toward the more open side (same sign
        # convention as the SEARCH spin: _drive(-,+) turns CCW/left).
        if self._near.ahead < AVOID_TRIGGER_M:
            turn_dir = +1 if self._near.left >= self._near.right else -1
            self._drive(-turn_dir * SEARCH_SPIN_RAD_S,
                        +turn_dir * SEARCH_SPIN_RAD_S)
            self.world.path = None
            return

        # Plan A* to the goal on the usual cadence, then follow with DWA.
        if self.world.tick % PLAN_EVERY_TICKS == 0:
            self._replan(gx_w, gy_w)
        if not self.world.path or self.world.path_idx >= len(self.world.path):
            # No route yet (map too sparse / goal unreachable): stop and wait
            # for the next replan rather than driving blind.
            self._drive(0.0, 0.0)
            if self.world.tick % 30 == 0:
                print(f"  [t={self.world.tick}] GOTO no route to "
                      f"({gx_w:+.2f},{gy_w:+.2f}) - waiting")
            return

        self._inflated_grid = inflate_grid(
            self.world.wall_status, self.cfg.grid.inflate_m, self.cfg.grid)
        self.world.path_idx = dwa_mod.advance_path_index(
            self.world.path, self.world.bot_x, self.world.bot_y,
            self.world.path_idx, self.cfg.grid,
        )
        sel = dwa_mod.select_velocity(
            self.world.bot_x, self.world.bot_y, self.world.bot_yaw,
            self.world.path, self.world.path_idx,
            self._inflated_grid, self.cfg.grid, self.cfg.robot,
            self.cfg.tuning.cruise_rad_s, DWA_CFG,
        )
        if sel is None:                      # all DWA trajectories collide
            self.world.path = None
            self._drive(0.0, 0.0)
            return
        v_l, v_r, _v_body, _w_body = sel
        self._drive(v_l, v_r)
        if self.world.tick % 30 == 0:
            print(f"  [t={self.world.tick}] GOTO d={dist*100:.0f}cm "
                  f"vL={v_l:+.2f} vR={v_r:+.2f}")

    # Helpers
    def _drive(self, v_l: float, v_r: float) -> None:
        # Dashboard "disconnect brain" switch: when off, zero EVERY command so
        # the bot just balances in place (the ESP keeps it upright) while the
        # brain keeps mapping / planning / publishing. Single choke point, so
        # no FSM state can move the motors while disconnected.
        if not telemetry.get_drive_enabled():
            v_l = v_r = 0.0
        v_l, v_r = clamp_wheel_pair(v_l, v_r, self.cfg.robot.max_wheel_rad_s)
        self._last_cmd_wheels = (v_l, v_r)    # dashboard balance strip
        self.sensors.set_wheel_velocities(v_l, v_r)

    # Telemetry
    def _apply_wall_clears(self) -> None:
        """Apply any dashboard wall-clear requests (`/clear`). Wipes
        wall_status + vote + flags for targeted cells; the anchor footprint
        (`wall_discovered_tick == -1`) is preserved as a known fixture.
        `all` clears every observed wall; otherwise a disk around the
        clicked point. Live LiDAR re-stamps any wall still in view.
        """
        ops = telemetry.get_wall_clear_ops()
        if not ops or self.world.wall_status is None:
            return
        ws = self.world.wall_status
        dim_x, dim_y = ws.shape
        cell_m, half_m = self.cfg.grid.cell_m, self.cfg.grid.half_m
        observed = self.world.wall_discovered_tick != -1   # -1 == anchor fixture
        total = 0
        for op in ops:
            if op.get("all"):
                mask = observed & ws
            else:
                gx = np.arange(dim_x).reshape(-1, 1)
                gy = np.arange(dim_y).reshape(1, -1)
                wx = (gx + 0.5) * cell_m - half_m
                wy = (gy + 0.5) * cell_m - half_m
                disk = (wx - op["x"]) ** 2 + (wy - op["y"]) ** 2 <= op["radius_m"] ** 2
                mask = disk & observed & ws
            n = int(mask.sum())
            if n:
                ws[mask] = False
                self.world.wall_status_tof[mask] = False
                self.world.wall_vote[mask] = 0
                self.world.wall_discovered_tick[mask] = -1
                total += n
        if total:
            self.world.path = None      # stale plan may route through cleared cells
            print(f"  [t={self.world.tick}] WALL CLEAR: wiped {total} cell(s) "
                  f"from {len(ops)} request(s)")

    def _apply_pose_set(self) -> None:
        """Apply a dashboard "move bot" request (`/pose`): teleport the
        odometry pose estimate to the clicked point (and yaw, if given),
        re-zeroing the IMU offset so the next odometry tick doesn't snap
        it back."""
        req = telemetry.get_pose_request()
        if req is None:
            return
        self.world.bot_x = req["x"]
        self.world.bot_y = req["y"]
        if req.get("yaw_deg") is not None:
            self.world.bot_yaw = wrap_angle(math.radians(req["yaw_deg"]))
            imu = self.sensors.get_imu()
            if imu is not None:
                self.world.imu_yaw_offset = wrap_angle(imu.yaw - self.world.bot_yaw)
        self.world.path = None      # the old plan started from the old pose
        print(f"  [t={self.world.tick}] MOVE BOT: pose set to "
              f"({self.world.bot_x:+.2f},{self.world.bot_y:+.2f},"
              f"{math.degrees(self.world.bot_yaw):+.0f}°)")

    def _apply_full_reset(self) -> None:
        """Apply a dashboard full-reset request (`/reset`, the map "reset"
        button): a clean slate. Wipe the ENTIRE wall map, drop the current
        plan, the detected cup and the A* goal probe, and re-zero the pose to
        the origin (re-zeroing the IMU offset so odometry doesn't snap back).
        Live LiDAR re-maps whatever is in view from the fresh origin."""
        if not telemetry.get_reset_request():
            return
        ws = self.world.wall_status
        if ws is not None:
            ws[:] = False
            self.world.wall_status_tof[:] = False
            self.world.wall_vote[:] = 0
            self.world.wall_discovered_tick[:] = -1
        self.world.path = None
        self.world.glass_xy_acc = None
        self._home_xy = None
        self._returning = False
        self._was_driving = False
        self.world.bot_x = 0.0
        self.world.bot_y = 0.0
        self.world.bot_yaw = 0.0
        imu = self.sensors.get_imu()
        if imu is not None:                          # re-zero so it won't drift back
            self.world.imu_yaw_offset = wrap_angle(imu.yaw - self.world.bot_yaw)
        telemetry.set_goal_state(None, None, active=False)
        print(f"  [t={self.world.tick}] FULL RESET: map wiped + pose re-zeroed "
              f"to origin")

    def _publish_telemetry(self) -> None:
        """Snapshot the current FSM + perception state for the dashboard.

        Images encoded every 4 ticks to keep encoding overhead bounded.
        """
        self._apply_full_reset()
        self._apply_wall_clears()
        self._apply_pose_set()
        cup = None
        if self.world.glass_xy_acc is not None:
            gx, gy = self.world.glass_xy_acc
            dist = math.hypot(gx - self.world.bot_x, gy - self.world.bot_y)
            cup = {"x": gx, "y": gy, "distance": dist}
        n = self._near
        def _finite(v): return None if math.isinf(v) else v
        n_walls = int(self.world.wall_status.sum()) if self.world.wall_status is not None else 0
        snap = {
            "tick": self.world.tick,
            "state": self.world.state.name,
            "mode": telemetry.get_mode(),
            "bot": {
                "x": self.world.bot_x,
                "y": self.world.bot_y,
                "yaw_deg": math.degrees(self.world.bot_yaw),
            },
            "cup": cup,
            "near": {
                "ahead":      _finite(n.ahead),
                "left":       _finite(n.left),
                "right":      _finite(n.right),
                "hard_left":  _finite(n.hard_left),
                "hard_right": _finite(n.hard_right),
            },
            "walls": n_walls,
            "path_len": len(self.world.path) if self.world.path else 0,
            "path_idx": self.world.path_idx,
            "home": ({"x": self._home_xy[0], "y": self._home_xy[1]}
                     if self._home_xy is not None else None),
            "returning": self._returning,
        }
        # Dashboard A* goal probe (reused from the last imagery tick so the
        # readout stays stable between refreshes). None when no goal set.
        snap["goal"] = None
        if self._last_probe is not None:
            gx_w, gy_w = self._last_probe["goal_world"]
            snap["goal"] = {
                "x":          gx_w,
                "y":          gy_w,
                "reachable":  self._last_probe["reachable"],
                "path_len":   self._last_probe["path_len"],
                "explored":   self._last_probe["n_explored"],
            }

        # Sensor rates over a ~1 s wall window (fresh-return counts).
        now = time.perf_counter()
        if self._rate_t0 is None:
            self._rate_t0 = now
        elapsed = now - self._rate_t0
        if elapsed >= 1.0:
            self._perf["lidar_hz"] = self._ema(self._perf["lidar_hz"], self._lidar_frames / elapsed)
            self._perf["cam_fps"]  = self._ema(self._perf["cam_fps"],  self._cam_frames / elapsed)
            self._lidar_frames = self._cam_frames = 0
            self._rate_t0 = now

        # Performance: real-time budget (quantitative metrics for the demo).
        snap["perf"] = {
            "tick_hz":       round(self._perf["tick_hz"], 1),
            "perception_ms": round(self._perf["perception_ms"], 1),
            "plan_ms":       round(self._perf["plan_ms"], 2),
            "render_ms":     round(self._perf["render_ms"], 1),
        }
        # Pipeline: sensor link health (proves the Mac<->Pi interface is live).
        snap["pipeline"] = {
            "lidar_hz": round(self._perf["lidar_hz"], 1),
            "cam_fps":  round(self._perf["cam_fps"], 1),
            "link_ok":  bool(self._perf["lidar_hz"] > 0.5 or self._perf["cam_fps"] > 0.5),
        }
        # Brain-connected switch (dashboard toggle) so the UI reflects it.
        snap["drive_enabled"] = telemetry.get_drive_enabled()
        # Balance: ESP32 subsystem channels. tilt is the balance angle the ESP
        # reports (via get_imu().pitch); yaw is the integrated gyro heading;
        # cmd_l/cmd_r are the wheel velocities the brain last commanded.
        imu = self.sensors.get_imu()
        v_l, v_r = self._last_cmd_wheels
        snap["balance"] = {
            "tilt_deg": None if imu is None else round(math.degrees(imu.pitch), 1),
            "yaw_deg":  round(math.degrees(self.world.bot_yaw), 1),
            "cmd_l":    round(v_l, 2),
            "cmd_r":    round(v_r, 2),
        }

        # Imagery: every 4 ticks
        frame_bgr = None
        map_bgr = None
        if self.world.tick % 4 == 0:
            # Dashboard imagery: prefer a sensor-provided DISPLAY frame (e.g. the
            # hardware adapter overlays a hand skeleton) so debug graphics never
            # land on the frame perception uses. Falls back to the raw frame; the
            # getattr keeps this a no-op for sources without the hook (sim).
            _disp = getattr(self.sensors, "get_display_frame", None)
            base_rgb = (_disp() if _disp is not None else None)
            if base_rgb is None:
                base_rgb = self._last_rgb
            if base_rgb is not None:
                frame_bgr = telemetry.annotate(
                    base_rgb, self.cfg.camera,
                    near=self._near,
                    pose=self._last_pose,
                    aruco_detections=self._last_aruco_detections,
                    fsm_state=self.world.state.name,
                )
            if self.world.wall_status is not None:
                # Compute the inflated grid here (cheaply) so the
                # dashboard can show the halo A* actually plans on.
                # Critical for the "why isn't A* finding a route?"
                # question: a corridor that looks open in the raw map
                # is often closed by inflation, and you can only see
                # that if you visualise the inflated grid directly.
                infl = inflate_grid(self.world.wall_status,
                                    self.cfg.grid.inflate_m,
                                    self.cfg.grid)
                # Refresh the dashboard A* goal probe on the imagery
                # cadence (cheap: one A* run only while a goal is set).
                self._last_probe = self._probe_plan(infl)
                # Dashboard zoom/pan: the browser POSTs to /view on
                # scroll/drag, the server stashes the values, and we
                # pull them here so the next image refresh uses the new
                # framing. Each None field falls back to the config
                # default, so a freshly-loaded dashboard renders with
                # the same starting framing as before.
                vs = telemetry.get_view_state()
                cfg_cx, cfg_cy = self.cfg.tuning.map_view_centre
                vhalf = (vs["half_m"]   if vs["half_m"]   is not None
                                          else self.cfg.tuning.map_view_half_m)
                vcx   = (vs["centre_x"] if vs["centre_x"] is not None
                                          else cfg_cx)
                vcy   = (vs["centre_y"] if vs["centre_y"] is not None
                                          else cfg_cy)
                _t_render = time.perf_counter()
                map_bgr = telemetry.render_map(
                    self.world.wall_status,
                    cell_m=self.cfg.grid.cell_m,
                    half_m=self.cfg.grid.half_m,
                    bot_xy=(self.world.bot_x, self.world.bot_y),
                    bot_yaw=self.world.bot_yaw,
                    cup_xy=self.world.glass_xy_acc,
                    path=self.world.path,
                    wall_status_tof=self.world.wall_status_tof,
                    view_half_m=vhalf,
                    view_centre=(vcx, vcy),
                    inflated_grid=infl,
                    wall_segments=self.world.wall_segments,
                    lidar_scan=self.world.lidar_scan,
                    lidar_scan_pose=self.world.lidar_scan_pose,
                    probe_goal=(self._last_probe["goal_world"]
                                if self._last_probe else None),
                    probe_path=(self._last_probe["probe_path"]
                                if self._last_probe else None),
                    explored_mask=(self._last_probe["explored_mask"]
                                   if self._last_probe else None),
                )
                self._perf["render_ms"] = self._ema(
                    self._perf["render_ms"], (time.perf_counter() - _t_render) * 1000.0)

        telemetry.publish(snap, frame_bgr, map_bgr)
