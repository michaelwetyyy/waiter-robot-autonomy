"""Mutable per-tick state shared between perception and the FSM.

Lives in the controller process for the lifetime of one run. Sensors
WRITE into it (odometry, perception); the FSM READS it to decide what
to command.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List
import numpy as np


class FSMState(Enum):
    SEARCH      = auto()  # haven't seen the cup yet, spin to find it
    SEEK        = auto()  # drive to the cup with the shared goto stack
    DONE        = auto()  # arrived, motors off, LEDs on


@dataclass
class WorldState:
    # Odometry pose (anchored at the bot's starting position)
    bot_x:   float = 0.0
    bot_y:   float = 0.0
    bot_yaw: float = 0.0

    # Encoder bookkeeping (None until first read)
    prev_left_rad:  Optional[float] = None
    prev_right_rad: Optional[float] = None

    # IMU bookkeeping
    # `imu_yaw_offset` is the world-frame yaw at the bot's starting
    # heading; we subtract it from every subsequent IMU reading so the
    # bot's odometry frame is anchored at "I started facing 0 rad". None
    # until the first IMU sample arrives. If get_imu() returns None
    # forever (older world without InertialUnit), this stays None and
    # odometry falls back to wheel-encoder-derived yaw.
    imu_yaw_offset: Optional[float] = None

    # Cup detection (smoothed across frames)
    glass_xy_acc: Optional[Tuple[float, float]] = None

    # Persistent wall map
    # `wall_status` is the FINAL "is this cell a wall" map A* uses.
    # Populated by LiDAR (perception/lidar_walls.py): every revolution
    # carves free space along each ray and stamps hits at the endpoint.
    # The camera-projection pipeline that used to feed this map (with
    # windowed-consensus voting + negative-evidence clearance) was removed.
    wall_status: Optional[np.ndarray] = None      # bool (dim_x, dim_y)

    # Sub-mask of `wall_status` flagging cells from a non-camera,
    # geometrically-confident source (LiDAR hits, anchor footprint). Used
    # purely for the dashboard (painted magenta). Name is historical.
    wall_status_tof: Optional[np.ndarray] = None  # bool (dim_x, dim_y)

    # Tick at which each wall cell was first confirmed. Used to roll back
    # cells discovered during a drifted-pose interval when the next
    # anchor fix arrives showing significant slip. Value -1 means "never
    # confirmed". Walls are sticky once confirmed UNLESS rolled back.
    wall_discovered_tick: Optional[np.ndarray] = None  # int32 (dim_x, dim_y)

    # Negative-evidence counter, per cell. Reset to 0 every time a
    # LiDAR ray carves through the cell. Currently no consumer reads
    # this (the camera-pipeline negative-evidence clearance that owned
    # it was removed); kept on WorldState as a hook for the slip
    # handler, which still adjusts it when rolling cells back across a
    # drifted-pose interval.
    wall_absent_count: Optional[np.ndarray] = None     # int16 (dim_x, dim_y)

    # Per-cell LiDAR occupancy confidence (0 .. WALL_VOTE_MAX). Hits add,
    # misses (rays passing through) subtract, weighted by range so distant
    # walls decay far slower than near ones. `wall_status` flips True/False
    # off this with hysteresis (see perception.lidar_walls), which is what
    # stops walls flickering and keeps occluded walls (never revisited by a
    # ray, so never decremented) standing until a ray actually clears them.
    wall_vote: Optional[np.ndarray] = None             # int16 (dim_x, dim_y)

    # Phase 1 SLAM: clean line-segment view of the wall map
    # Recomputed after every LiDAR scan ingest. Used by the dashboard
    # now; ICP (Phase 2) will consume the same list.
    # Each entry is a `perception.wall_lines.WallSegment`.
    wall_segments: list = field(default_factory=list)

    # Most recent A* path: list of (gx, gy) grid cells
    path: Optional[List[Tuple[int, int]]] = None
    path_idx: int = 0

    # Latest 360° LiDAR scan (DFRobot DFR0315 / RPLIDAR A1-class)
    # `LidarScan` (ranges, angle_min, angle_increment). None when no
    # LiDAR is connected in this world, or between fresh revolutions
    # (the device delivers a new scan at ~5.5 Hz; controller ticks at
    # ~62.5 Hz, so most ticks see the previous scan here).
    lidar_scan: Optional[object] = None
    # Bot pose (x, y, yaw) AT THE TIME the latest scan was captured.
    # The dashboard / wall-ingest / ICP code transforms ray endpoints
    # into world frame using THIS pose, not the bot's current pose;
    # the bot moves ~2 cm between scans at cruise, so using the
    # current pose would smear the ray cloud and make the
    # visualization look like the walls are drifting with the bot.
    lidar_scan_pose: Optional[Tuple[float, float, float]] = None

    # Reactive AVOID state
    avoid_dir: int = 0

    # FSM
    state:    FSMState = FSMState.SEARCH
    tick:     int = 0
    last_log: str = ""

    @classmethod
    def fresh(cls, grid_dim: int) -> "WorldState":
        return cls(
            wall_status          = np.zeros((grid_dim, grid_dim), dtype=bool),
            wall_status_tof      = np.zeros((grid_dim, grid_dim), dtype=bool),
            wall_discovered_tick = np.full((grid_dim, grid_dim), -1, dtype=np.int32),
            wall_absent_count    = np.zeros((grid_dim, grid_dim), dtype=np.int16),
            wall_vote            = np.zeros((grid_dim, grid_dim), dtype=np.int16),
        )
