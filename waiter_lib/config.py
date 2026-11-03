"""Runtime configuration for the waiter robot.

Constants that describe the WORLD, the BOT's hardware, and tuning
parameters for control + perception. All values are in SI units (metres,
radians, seconds) unless noted.

The same `CONFIG` instance is used on Webots simulation and on the real Pi
If a value needs to differ between them (e.g. real camera intrinsics
from calibration vs. sim's ideal pinhole) keep a per-platform override
inside the corresponding `SensorSource` rather than branching here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


# Camera intrinsics
@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera model for the bot's single mast-mounted camera."""

    width:    int           # image pixel width
    height:   int           # image pixel height
    fov_h:    float         # horizontal field of view (rad)
    mount_x:  float         # camera FORWARD offset in BODY frame (m).
                            # Negative = camera is BEHIND the body origin.
                            # The e-puck PROTO places its turretSlot at
                            # local (-0.0095, 0, 0.0486), so the camera
                            # is 9.5 mm aft of the bot's centre; the
                            # perception code uses this to convert
                            # camera-frame projections into body frame
                            # correctly.
    mount_z:  float         # camera height above the floor (m, body frame)
    tilt:     float         # pitch about body +Y axis (rad).
                            # positive = camera tilted DOWN.

    @property
    def f_x(self) -> float:
        return self.width / (2.0 * math.tan(self.fov_h / 2.0))

    @property
    def fov_v(self) -> float:
        return 2.0 * math.atan((self.height / self.width) * math.tan(self.fov_h / 2.0))

    @property
    def f_y(self) -> float:
        return self.height / (2.0 * math.tan(self.fov_v / 2.0))

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsics matrix for OpenCV PnP."""
        return np.array([[self.f_x, 0,        self.width / 2.0],
                          [0,       self.f_y, self.height / 2.0],
                          [0,       0,        1]], dtype=np.float32)

    @property
    def dist(self) -> np.ndarray:
        """Lens distortion coefficients. Webots is ideal pinhole; the Pi
        camera would override this with calibration data."""
        return np.zeros(5, dtype=np.float32)


# Robot mechanics
@dataclass(frozen=True)
class RobotMechanics:
    wheel_radius_m:   float
    wheel_base_m:     float    # left-right distance between wheels
    max_wheel_rad_s:  float    # speed limit per motor


# World / grid
@dataclass(frozen=True)
class GridConfig:
    half_m:           float    # grid covers ±half_m in odometry frame
    cell_m:           float
    inflate_m:        float    # safety margin around projected walls
    # Soft clearance preference (A* path quality)
    # On top of the hard `inflate_m` margin, A* adds a soft penalty for
    # routing within `clearance_pref_m` of a wall, so paths bias toward
    # the middle of free space instead of grazing the inflation boundary.
    # `clearance_weight` is the max extra cost per cell (≈ fraction of a
    # one-cell step) charged right at the boundary, tapering to 0 at the
    # preferred clearance. Set weight to 0 to disable and get plain A*.
    clearance_weight:  float = 0.35
    clearance_pref_m:  float = 0.10

    @property
    def dim(self) -> int:
        return int(2 * self.half_m / self.cell_m)


# Behaviour tuning
@dataclass(frozen=True)
class Tuning:
    cruise_rad_s:        float
    search_spin_rad_s:   float
    scan_revolution:     float        # rad to rotate before SCAN can exit
    turn_gain:           float
    waypoint_tol_m:      float
    arrived_tol_m:       float
    wall_hit_threshold:  int          # cell hit count to classify as wall
    aruco_dict_name:     str
    marker_size_m:       float        # ArUco black border edge length (m)
    marker_offset_m:     float        # radial distance from cup/hex axis to
                                      # marker centre (the hex apothem
                                      # (= flat-to-flat width / 2). Used by
                                      # the detector to back-project from
                                      # marker-pose to cup-axis position
                                      # along the marker normal).
    cup_z_m:             float        # world height of cup centre
    telemetry_port:      int          # 0 disables; otherwise HTTP server port

    # Anchor block for absolute pose calibration
    # A 4-face cube fiducial mounted at a KNOWN position in odometry
    # frame. Each face carries a different ArUco ID; the controller can
    # solve PnP against whichever face is currently visible (sometimes
    # two at once at oblique viewing angles). Same fiducial concept will
    # run on the real robot against a printed cube glued to the docking
    # station.
    #
    # LEGACY (anchor cube): the brain no longer uses the ArUco anchor — it
    # localises from the LiDAR walls via scan-matching (see state_machine's
    # LiDAR ICP). These fields are kept ONLY for tools/generate_aruco_markers.py,
    # which still prints the cube faces. Safe to delete once that tool is retired.
    #
    # `anchor_faces` is a tuple of dicts, one per face. Each dict has:
    #   id:         ArUco marker ID for that face
    #   xy:         (x, y) of marker centre in ODOMETRY frame (m)
    #   z:          height of marker centre above floor (m)
    #   normal_yaw: yaw of marker face NORMAL (rad, ODOM frame).
    #               0 = +X, π/2 = +Y, π = -X, -π/2 = -Y.
    anchor_faces:        tuple
    anchor_center_xy:    tuple

    # Dashboard map view
    # World extent the dashboard renders. Set to the arena size so the
    # map fills the frame instead of wasting space showing the empty
    # grid storage margin.
    map_view_half_m:     float        # half-side length of the rendered window (m)
    map_view_centre:     tuple        # (x, y) of view centre in ODOMETRY frame


# Top-level config (default values match generate_world.py)
@dataclass(frozen=True)
class Config:
    camera:     CameraIntrinsics
    robot:      RobotMechanics
    grid:       GridConfig
    tuning:     Tuning


CONFIG = Config(
    camera=CameraIntrinsics(
        # Pi Camera v2 (Sony IMX219), MOUNTED ON ITS SIDE so the sensor's
        # long axis is vertical:
        #   - Native lens FOV is 62.2°H × 48.8°V on a landscape sensor
        #   - With the chip rotated 90° about the optical axis, the
        #     image becomes portrait: 48.8° horizontal × 62.2° vertical
        #   - That extra vertical reach is what we want; it covers the
        #     overhead cup AND the floor right in front of the bot in
        #     the same frame, even with the camera level (tilt=0).
        #
        # Working resolution is 1848×2460 portrait. There's no native
        # IMX219 sensor mode at this size with the full FOV preserved
        # (mode 3 binned is 1640×1232, mode 0 is 3280×2464; crop modes
        # in between lose FOV). So on the REAL bot, libcamera reads
        # mode 0 (3280×2464, full FOV) and the ISP / driver downsamples
        # 4:3 to this resolution, i.e. mode 0 x 3/4 = 2460x1848 (which
        # we rotate 90° to portrait, giving 1848x2460). In WEBOTS we just
        # render at the target resolution directly; no software
        # downsample needed since the simulator has no sensor model.
        #
        # Why this size:
        #   f_x = 1848 / (2·tan(fov_h/2)) ≈ 2024 px
        #   30 mm marker @ 3 m -> 30*f_x/3000 ~= 20.2 px  (comfortable)
        #   30 mm marker @ 2 m -> 30*f_x/2000 ~= 30.4 px  (abundant)
        #   30 mm marker @ 4 m -> 30*f_x/4000 ~= 15.2 px  (marginal)
        # Goal was 3 m detection range with margin for noise / lighting,
        # which this hits exactly at the comfort threshold. The previous
        # 1232×1640 mode 3 binned resolution only reached ~2 m.
        #
        # Cost on the real bot: libcamera bandwidth still capped at
        # 21 fps (mode 0 sensor limit), ArUco ~40 ms/frame on Pi 5
        # (~half mode 0's native CPU cost; same sensor read, smaller
        # working image). The downsample itself adds ~5 ms with the
        # libcamera ISP. Total perception budget ~50 ms = 20 Hz max.
        width=1848,
        height=2460,
        fov_h=0.852,     # 48.8° in radians (the SHORT-side FOV of the
                         # rotated Pi-cam-v2 sensor)
        # Camera origin in BODY frame, composed from:
        #   E-puck PROTO TURRET_SLOT Pose : (-0.0095, 0, 0.0486)
        #   World Camera node             : ( 0    , 0, 0.12  )
        # giving (-0.0095, 0, 0.1686). Real-bot port: re-measure on hardware.
        mount_x = -0.0095,
        mount_z = 0.0486 + 0.12,
        # ~10° below horizontal; compromise between near-wall bottom-
        # edge projection (which wants more tilt down so rays exit the
        # lens at a steeper angle to the floor) and overhead cup
        # detection (which wants the camera near level so the cup stays
        # in the upper FOV). The portrait sensor's 62.2° vertical FOV
        # plus this 10° tilt comfortably covers both at the same time.
        tilt=0.1745,    # 10° in radians, positive = DOWN
    ),
    robot=RobotMechanics(
        wheel_radius_m=0.020,
        wheel_base_m=0.052,
        max_wheel_rad_s=6.0,
    ),
    grid=GridConfig(
        # Spans ±half_m around bot's odometry origin. Must cover the
        # full operational range (bot, cup, anchor) plus drift margin.
        # `world_to_grid` clamps silently, so undersizing here makes
        # A* route to the grid edge instead of the actual goal.
        half_m=2.50,
        cell_m=0.025,
        # 7 cm planner clearance. The e-puck has a 3.7 cm body radius
        # and the grid is quantised in 2.5 cm cells, so the bot's
        # CENTRE can sit anywhere up to half a cell inside its planning
        # cell; effective wall-to-body distance after planning is
        # (inflate_m - radius - cell/2). With the previous 4 cm
        # inflation that came out to -0.9 cm worst-case (i.e. the bot
        # body could end up inside a wall while A* and DWA both think
        # the trajectory is clear), which is what showed up as "A*
        # route looks fine but the bot drives into walls". 7 cm
        # leaves +2.1 cm worst-case body clearance, enough that
        # ordinary odometry drift, wheel slip, and the (v, ω) rollout
        # discretisation can't push the body into a wall surface.
        # Near-field AVOID is still the immediate-collision backstop.
        inflate_m=0.07,
    ),
    tuning=Tuning(
        # 0.5 rad/s below max_wheel_rad_s leaves headroom for the
        # heading-PD differential; the diff saturates before overshoot
        # at this gap.
        cruise_rad_s=5.5,
        # Full body revolution in ~4 s. ArUco PnP stays robust at this
        # spin rate because pixel angular velocity is no faster than
        # the slower-spin baseline once camera resolution scaled up.
        search_spin_rad_s=2.0,
        scan_revolution=2.0 * math.pi,
        turn_gain=4.0,
        waypoint_tol_m=0.04,
        arrived_tol_m=0.015,
        # Cell becomes a wall only after this many pixel-projection hits.
        # Tuned against full SCAN at 1.2 rad/s (~400 frames). Real walls
        # accumulate hundreds of hits; horizon-noise projections only a
        # handful, so 20 is comfortably above the noise floor.
        wall_hit_threshold=20,
        aruco_dict_name="DICT_4X4_100",
        # Cup ArUco markers are now 30 mm (matches the Fusion design of
        # the physical cup (65 mm flat-to-flat hex with one 30 mm
        # square marker engraved on each face, low on the body). PnP
        # uses the BLACK-BORDER edge length, not the textured plane
        # size. Down from 10.5 cm (16 cm cube -> 14 cm plane -> 10.5 cm
        # ArUco border) before the physical redesign.
        marker_size_m=0.030,
        # Apothem of the cup hex (flat-to-flat / 2 = 65 mm / 2 = 32.5
        # mm). The detector uses this to step from the marker centre
        # back along the marker normal to the cup's central axis,
        # giving a consistent "where is the cup centre" estimate
        # regardless of which face is currently visible.
        marker_offset_m=0.0325,
        cup_z_m=0.30,
        telemetry_port=8080,          # http://localhost:8080 for live dashboard

        # Anchor block: HEXAGONAL prism, same body as the cup. Centred
        # at world (-1.0, 0), body height 0.10 m sitting on the floor
        # (z = 0..0.10). TWO stacked markers per face, twelve total:
        #   lower row  (ids 94-99)  local z = -0.033 -> world z = 0.017
        #   upper row  (ids 88-93)  local z = +0.033 -> world z = 0.083
        # Each row gives 360° coverage; doubling them up means there is
        # almost always a near-head-on marker in the camera frame even
        # when one row sits at a steep view angle relative to the bot's
        # current pose. PnP fuses across whichever row + face combination
        # passes the gates each tick.
        #
        # Hex geometry:
        #   apothem      = 0.0325 m   (= flat-to-flat / 2 = 65 mm / 2)
        #   circumradius = 0.0375 m   (= side length = Cylinder.radius)
        #   marker plane = 0.030 × 0.030 m   (30 mm ArUco)
        #
        # In ODOMETRY frame (bot start at world (-0.8, 0)), hex centre is
        # at (-0.20, 0). Face marker centres = hex_centre + apothem *
        # (cos θ, sin θ); apothem · sin 60° ≈ 0.0281, · cos 60° = 0.01625.
        # Upper and lower entries share xy + normal_yaw, differ only in z.
        anchor_faces=(
            # Lower row, world z = 0.017 m
            {"id": 99, "xy": (-0.1675,  0.0000), "z": 0.017, "normal_yaw":  0.0},
            {"id": 96, "xy": (-0.1838, +0.0281), "z": 0.017, "normal_yaw":  math.pi / 3},
            {"id": 94, "xy": (-0.2163, +0.0281), "z": 0.017, "normal_yaw":  2 * math.pi / 3},
            {"id": 97, "xy": (-0.2325,  0.0000), "z": 0.017, "normal_yaw":  math.pi},
            {"id": 95, "xy": (-0.2163, -0.0281), "z": 0.017, "normal_yaw": -2 * math.pi / 3},
            {"id": 98, "xy": (-0.1838, -0.0281), "z": 0.017, "normal_yaw": -math.pi / 3},
            # Upper row, world z = 0.083 m
            {"id": 93, "xy": (-0.1675,  0.0000), "z": 0.083, "normal_yaw":  0.0},
            {"id": 92, "xy": (-0.1838, +0.0281), "z": 0.083, "normal_yaw":  math.pi / 3},
            {"id": 91, "xy": (-0.2163, +0.0281), "z": 0.083, "normal_yaw":  2 * math.pi / 3},
            {"id": 90, "xy": (-0.2325,  0.0000), "z": 0.083, "normal_yaw":  math.pi},
            {"id": 89, "xy": (-0.2163, -0.0281), "z": 0.083, "normal_yaw": -2 * math.pi / 3},
            {"id": 88, "xy": (-0.1838, -0.0281), "z": 0.083, "normal_yaw": -math.pi / 3},
        ),
        anchor_center_xy=(-0.20, 0.0),

        # Arena 2.4 × 2.4 m. Bot starts at world (-0.8, 0); odometry
        # frame is anchored there. Maze entrance at odom (+0.8, 0), maze
        # extends to odom (+1.7, ±0.5), cup at odom (+1.4, 0). View
        # centred at (+0.55, 0) with 1.30 m half-extent fits bot, maze,
        # and the anchor (odom (-0.20, 0)) with breathing room.
        # Arena is 2.4×2.4 m centred at world (0,0); odometry frame is
        # offset so the arena spans x ∈ [-0.4, +2.0], y ∈ [-1.2, +1.2]
        # in odom. View centred at odom (+0.80, 0) with half-extent
        # 1.40 m fits the full arena (±1.2 m from centre) with a 20 cm
        # margin on every side, both perimeter walls visible. Was
        # (0.55, 0)/1.30 m which cut off the right-hand perimeter wall.
        map_view_half_m=1.40,
        map_view_centre=(0.80, 0.0),
    ),
)
