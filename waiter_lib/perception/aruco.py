"""ArUco marker detection: cup pose and anchor-based pose fix.

Pure CV. No platform dependencies. Same code runs on Webots and on the Pi.

Two roles for ArUco markers in this build:
  - Cup markers (IDs 1-6): one on each vertical face of the moving
    HEXAGONAL cup. Drive SEEK / CENTRE. Pose is reported in the bot's
    BODY frame.
  - Anchor markers (IDs 94-99): the six vertical faces of the fixed
    hexagonal docking block at a KNOWN position in the odometry frame.
    When the camera sees ANY anchor face, we recover the bot's absolute
    pose in odometry frame via PnP and reset the drifting encoder + IMU
    estimate. Result is an `AnchorFix`. Two adjacent faces are usually
    visible together; `detect_anchor` picks the best candidate.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from ..config import CameraIntrinsics


# Marker-ID conventions. The detector skips anchor markers when scanning
# for cups (otherwise an anchor in view could be reported as a cup and
# yank the planner toward the docking station).
CUP_IDS = frozenset({1, 2, 3, 4, 5, 6})


@dataclass
class CupPose:
    forward:  float    # body +X (m)
    lateral:  float    # body +Y (m); positive = LEFT of bot
    distance: float    # horizontal distance to cup centre (m)
    marker_id: int     # which face we saw


class ArucoCupDetector:
    """Detects any single marker from a known dictionary and projects it
    into the bot's body frame.

    Reused: an instance is constructed once (the OpenCV detector is
    expensive to build) and `detect()` is called every frame.
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 marker_size_m: float,
                 aruco_dict_name: str = "DICT_4X4_100",
                 marker_offset_m: float = 0.0):
        self.intrinsics      = intrinsics
        self.marker_size_m   = marker_size_m
        # Radial distance from the cup central axis to a face marker
        # centre (= hex apothem). Used by `detect()` to back-project
        # each visible face along its outward normal to the cup axis,
        # then average across faces. 0.0 = legacy behaviour (report the
        # marker's own position).
        self.marker_offset_m = marker_offset_m

        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
        params = cv2.aruco.DetectorParameters()
        # Detector tuning to stretch usable range out to ~2 m.
        # With 30 mm markers and the 1232-wide camera, a marker at 2 m
        # projects to ~20 px, close to the decode threshold.
        #
        # `minMarkerPerimeterRate` is the min marker perimeter as a
        # fraction of max(width, height). Default 0.03 -> ~12 px per
        # side. We halve it so a marker at 2 m (≈ 80 px perimeter / 1640
        # ≈ 0.049) clears with margin. Lowering further risks noise.
        params.minMarkerPerimeterRate = 0.015
        # Sub-pixel corner refinement: adds ~1 ms per detected marker
        # but tightens PnP enough to be worth it at the edge of range.
        params.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy   = 0.05
        self._detector = cv2.aruco.ArucoDetector(d, params)

    def detect(self, rgb: np.ndarray) -> Optional[CupPose]:
        """Returns the cup's position in body frame, fused across ALL
        visible cup faces.

        Pipeline:
          1. Detect every visible cup marker (IDs in CUP_IDS).
          2. For each, IPPE_SQUARE PnP -> marker pose in camera frame.
          3. Convert position and normal into body frame.
          4. Step back along the marker's outward normal by
             `marker_offset_m` (the cup hex apothem), giving a
             per-face estimate of where the CUP AXIS is.
          5. Average those per-face axis estimates, weighted by marker
             pixel area (larger marker = lower-noise PnP).

        Why the back-projection: a single face only tells you where
        THAT FACE is, which sits 32.5 mm off the cup centre. With six
        faces 60° apart, mixing them naively biases the cup-position
        estimate toward whichever face is most visible. Back-projecting
        each face to the central axis first makes the per-face
        estimates AGREE in expectation, so averaging tightens them
        instead of pulling them apart.

        Anchor markers (non-cup IDs) are skipped so they can't
        masquerade as the cup.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        # Collect every visible cup marker with its pixel area.
        cup_detections = []   # list of (corners4, area, id)
        for i, c in enumerate(corners):
            mid = int(ids[i][0])
            if mid not in CUP_IDS:
                continue
            area = float(cv2.contourArea(c[0]))
            cup_detections.append((c[0], area, mid))
        if not cup_detections:
            return None

        # Pre-build constants used by every face.
        s = self.marker_size_m / 2.0
        obj_pts = np.array([
            [-s,  s, 0.0],   # top-left
            [ s,  s, 0.0],   # top-right
            [ s, -s, 0.0],   # bottom-right
            [-s, -s, 0.0],   # bottom-left
        ], dtype=np.float32)

        # Camera-frame -> body-frame rotation. See Build Log section 16:
        #   cam_x_in_body = (0, -1, 0)
        #   cam_y_in_body = (-sin(tilt), 0, -cos(tilt))
        #   cam_z_in_body = ( cos(tilt), 0, -sin(tilt))
        # Each column of R_cb is one camera axis expressed in body
        # frame, so v_body = R_cb @ v_cam.
        tilt = self.intrinsics.tilt
        ct, st = math.cos(tilt), math.sin(tilt)
        R_cb = np.array([
            [ 0.0,  -st,    ct],
            [-1.0,  0.0,  0.0],
            [ 0.0,  -ct,   -st],
        ])

        # Per-face: PnP -> marker pos in body frame -> step back along
        # marker normal by `marker_offset_m` -> cup-axis estimate.
        axis_estimates = []   # list of (axis_pos_body, weight=area, id)
        best_area = 0.0
        best_id   = cup_detections[0][2]
        for c4, area, mid in cup_detections:
            img_pts = c4.astype(np.float32)
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_pts, img_pts,
                self.intrinsics.K, self.intrinsics.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or len(tvecs) == 0:
                continue
            # IPPE planar-pose ambiguity: lower-reprojection-error
            # branch wins per face.
            best = int(np.argmin(np.asarray(errs).ravel()))
            tv = tvecs[best].reshape(3)
            R_mc, _ = cv2.Rodrigues(rvecs[best])

            # Marker centre and outward normal in body frame.
            # `R_cb @ tv` gives the marker offset from CAMERA in body
            # frame; add the camera's own body-frame position so the
            # result is the marker's position relative to the BOT
            # origin (which is what callers expect for navigation).
            camera_in_body = np.array([self.intrinsics.mount_x,
                                        0.0,
                                        self.intrinsics.mount_z])
            marker_pos_body = camera_in_body + R_cb @ tv
            marker_normal_cam = R_mc[:, 2]                     # marker +Z = outward
            marker_normal_body = R_cb @ marker_normal_cam

            # Cup axis = step from marker centre INTO the cup along
            # -normal by the apothem. If marker_offset_m == 0 we just
            # report the marker centre (legacy behaviour).
            axis_pos_body = (marker_pos_body
                             - self.marker_offset_m * marker_normal_body)
            axis_estimates.append((axis_pos_body, area, mid))
            if area > best_area:
                best_area = area
                best_id   = mid

        if not axis_estimates:
            return None

        # Weighted mean across faces (weight = marker pixel area). One
        # single face: returns that face's estimate directly.
        total_w = sum(w for _, w, _ in axis_estimates)
        fused = np.zeros(3)
        for pos, w, _ in axis_estimates:
            fused += pos * (w / total_w)

        forward = float(fused[0])    # body +X
        lateral = float(fused[1])    # body +Y (LEFT)
        dist_h  = math.hypot(forward, lateral)

        return CupPose(
            forward=forward,
            lateral=lateral,
            distance=dist_h,
            marker_id=best_id,
        )
