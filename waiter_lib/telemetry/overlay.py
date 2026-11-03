"""Draw debug overlays on the camera frame for the telemetry UI.

What gets drawn:
  - orange-mask outline (yellow contour)
  - ArUco quad with marker id + tvec
  - near-field sector arcs at the bottom of the image, colour-coded
    by clearance: green = clear, yellow = caution, red = panic
  - cup-direction crosshair if the cup is currently being detected

Pure CV. Same code runs on Webots and on the Pi.
"""

from __future__ import annotations
import math
from typing import Optional
import cv2
import numpy as np

from ..config import CameraIntrinsics
from ..perception.walls import orange_mask, NearField
from ..perception.aruco import CupPose


def annotate(
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    near: Optional[NearField] = None,
    pose: Optional[CupPose] = None,
    aruco_detections=None,    # iterable of (corners4 ndarray, id int)
    fsm_state: str = "",
) -> np.ndarray:
    """Return an annotated BGR copy of `rgb` ready for JPEG encoding.

    `aruco_detections` is the list of EVERY visible quad; the overlay
    draws all of them, not just the biggest. This makes it obvious when
    multi-face cup fusion is using more than one marker face
    simultaneously; previously only the largest quad was rendered which
    looked like single-marker detection.
    """
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # orange-mask outline
    mask = orange_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 1)   # yellow

    # ArUco quads
    # Draw every detected quad. The largest gets the distance annotation
    # appended (so the user sees the same headline number from the
    # fused cup pose).
    if aruco_detections:
        biggest_i = max(
            range(len(aruco_detections)),
            key=lambda i: cv2.contourArea(aruco_detections[i][0][0]),
        )
        for i, (corners, mid) in enumerate(aruco_detections):
            pts = corners.reshape(-1, 2).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)        # green
            cx, cy = pts.mean(axis=0).astype(int)
            label = f"id={mid}"
            if i == biggest_i and pose is not None:
                label += f" d={pose.distance*100:.0f}cm"
            cv2.putText(img, label, (cx - 30, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                        cv2.LINE_AA)

    # near-field sector indicator (small arc at bottom-centre)
    if near is not None:
        _draw_sector_strip(img, near)

    # FSM state badge
    if fsm_state:
        cv2.putText(img, fsm_state, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img, fsm_state, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)

    return img


def _draw_sector_strip(img: np.ndarray, near: NearField) -> None:
    """Bottom-centre row of 5 colour-coded sector squares.

    Layout:  [hL] [L] [A] [R] [hR]   with distance numbers underneath.
    Green = clear (>30 cm), yellow = caution (10-30 cm), red = panic (<10 cm).
    """
    h, w = img.shape[:2]
    pad = 4
    sw = 30      # square side
    gap = 4
    total_w = 5 * sw + 4 * gap
    x0 = (w - total_w) // 2
    y0 = h - sw - 22

    sectors = [
        ("hL", near.hard_left),
        ("L",  near.left),
        ("A",  near.ahead),
        ("R",  near.right),
        ("hR", near.hard_right),
    ]
    for i, (label, d) in enumerate(sectors):
        x = x0 + i * (sw + gap)
        if math.isinf(d):           col = (60, 200, 60)        # green
        elif d > 0.30:              col = (60, 200, 60)
        elif d > 0.10:              col = (60, 200, 200)       # yellow
        else:                       col = (60, 60, 220)        # red
        cv2.rectangle(img, (x, y0), (x + sw, y0 + sw), col, -1)
        cv2.rectangle(img, (x, y0), (x + sw, y0 + sw), (0, 0, 0), 1)
        cv2.putText(img, label, (x + 4, y0 + sw - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        dist_text = "inf" if math.isinf(d) else f"{d*100:.0f}"
        cv2.putText(img, dist_text, (x + 2, y0 + sw + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA)


def render_map(
    wall_status: np.ndarray,
    cell_m: float,
    half_m: float,
    bot_xy: tuple,
    bot_yaw: float,
    cup_xy: Optional[tuple] = None,
    path: Optional[list] = None,
    pixels: int = 900,
    wall_status_tof: Optional[np.ndarray] = None,
    view_half_m: Optional[float] = None,
    view_centre: Optional[tuple] = None,
    inflated_grid: Optional[np.ndarray] = None,
    wall_segments: Optional[list] = None,
    lidar_scan: Optional[object] = None,
    lidar_scan_pose: Optional[tuple] = None,
    probe_goal: Optional[tuple] = None,
    probe_path: Optional[list] = None,
    explored_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a top-down BGR view of the partial wall map.

    Storage extent is ±`half_m` (used for cell-index <-> world conversion).
    Rendered view extent is ±`view_half_m` centred at `view_centre`
    (both default to the storage extent + origin). Pass smaller values
    to zoom in on the maze rather than the full grid.

      - Inflated-grid halo: faint blue tint on cells that A* treats as
        BLOCKED but aren't raw walls (purely from inflation dilation).
        Critical for understanding why A* can't see a corridor that
        looks open in the raw map.
      - Camera-discovered walls: blue-ish
      - ToF-discovered walls   : magenta (when `wall_status_tof` given)
      - Wall line segments     : orange overlay (when `wall_segments`)
      - LiDAR hits             : small yellow dots at each ray's finite
                                  return endpoint, transformed via the
                                  pose-at-scan-time (`lidar_scan_pose`)
                                  so they don't smear as the bot moves
                                  between revolutions.
      - Bot: red circle + yaw arrow
      - Cup: pinkish cross
      - A* path: green polyline
      - Anchor marker: purple square + face-direction arrow
      - A* goal probe (dashboard click): pale-orange cloud of the cells
        A* explored (`explored_mask`), an orange route to the clicked
        point (`probe_path`), and an orange goal flag (`probe_goal`).
        Purely a visualisation; the bot is NOT driving here.
    """
    vhm = half_m if view_half_m is None else view_half_m
    vcx, vcy = view_centre if view_centre is not None else (0.0, 0.0)
    canvas = np.full((pixels, pixels, 3), 255, dtype=np.uint8)
    scale = pixels / (2.0 * vhm)               # px per metre

    def w2px(wx: float, wy: float) -> tuple:
        # World (x->right, y->up) to image (col->right, row->down), with
        # the view re-centred at (vcx, vcy).
        col = int((wx - vcx + vhm) * scale)
        row = int(pixels - (wy - vcy + vhm) * scale)
        return col, row

    def draw_cell_path(cells, line_col, dot_col, aa=False):
        # Polyline + corner dots through a list of (gx, gy) grid cells.
        # Shared by the autonomous A* path and the dashboard probe route.
        if not cells:
            return
        pts = [w2px((gx + 0.5) * cell_m - half_m, (gy + 0.5) * cell_m - half_m)
               for gx, gy in cells]
        line = (line_col, 2, cv2.LINE_AA) if aa else (line_col, 2)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], *line)
        for p in pts:
            cv2.circle(canvas, p, 3, dot_col, -1)

    # Painted in z-order: halo, then camera walls, then non-camera-sensor walls
    # (so the latter wins the overlap, and walls cover their inflation halo).
    half_px = int(cell_m * scale / 2)
    COL_CAM      = (180, 100,   0)   # blue-ish: camera-projected
    COL_TOF      = (180,   0, 180)   # magenta: non-camera sensor (LiDAR)
    COL_INFLATE  = (235, 215, 195)   # faint pale-blue: A* halo only
    COL_EXPLORED = (205, 228, 250)   # faint pale-orange: A* search cloud

    h_px = max(half_px, 0)

    def _paint_cells(mask: np.ndarray, colour: tuple) -> None:
        # Fully vectorised: one numpy fancy-index assignment per pixel offset
        # in the cell's (2·h_px+1)² square, NOT one cv2.rectangle per cell.
        # The old per-cell loop did thousands of Python->C calls as the wall
        # map + inflation halo grew (~47 ms/render, dragging the tick rate);
        # this is flat in cell count: a few dozen vectorised ops regardless.
        gxs, gys = np.where(mask)
        if gxs.size == 0:
            return
        cols = (((gxs + 0.5) * cell_m - half_m - vcx + vhm) * scale).astype(np.int32)
        rows = (pixels - ((gys + 0.5) * cell_m - half_m - vcy + vhm) * scale).astype(np.int32)
        col_arr = np.array(colour, dtype=np.uint8)
        for dy in range(-h_px, h_px + 1):
            rr = rows + dy
            in_r = (rr >= 0) & (rr < pixels)
            for dx in range(-h_px, h_px + 1):
                cc = cols + dx
                ok = in_r & (cc >= 0) & (cc < pixels)
                canvas[rr[ok], cc[ok]] = col_arr

    # A* search cloud first, so walls/halo/path all draw on top of it.
    if explored_mask is not None and explored_mask.any():
        _paint_cells(explored_mask, COL_EXPLORED)

    if inflated_grid is not None:
        _paint_cells(inflated_grid & ~wall_status, COL_INFLATE)
    if wall_status_tof is not None:
        _paint_cells(wall_status & ~wall_status_tof, COL_CAM)
        _paint_cells(wall_status_tof, COL_TOF)
    else:
        _paint_cells(wall_status, COL_CAM)

    # Phase 1 SLAM: clean line-segment overlay drawn on top of the
    # voted cells. Orange to read as "wall" (matches the real-world
    # wall colour) and to stay distinct from the green A* path. Wide
    # dark underlay then a thinner bright stroke for legibility on
    # both the white background and the blue/magenta cell layer.
    if wall_segments:
        for seg in wall_segments:
            c0 = w2px(seg.p0[0], seg.p0[1])
            c1 = w2px(seg.p1[0], seg.p1[1])
            cv2.line(canvas, c0, c1, (10,  40,  90),  4, cv2.LINE_AA)
            cv2.line(canvas, c0, c1, (40, 140, 230),  2, cv2.LINE_AA)

    # LiDAR hit cloud: one yellow dot per finite-return ray at the
    # ray's world-frame endpoint. Misses (inf) get no dot; the
    # absence of dots in a sector IS the visualisation of free space.
    # Pose-at-scan-time is preferred over current bot pose so the cloud
    # doesn't drift with the bot between revolutions. The fallback to
    # (bot_xy, bot_yaw) is the first-tick case where the FSM hasn't
    # received a scan yet; in practice the `lidar_scan` check is None
    # at that point so this branch never runs without a pose anyway.
    if lidar_scan is not None and getattr(lidar_scan, "ranges", None) is not None \
            and lidar_scan.ranges.size > 0:
        if lidar_scan_pose is not None:
            sx, sy, syaw = lidar_scan_pose
        else:
            sx, sy = bot_xy
            syaw   = bot_yaw
        ranges = np.asarray(lidar_scan.ranges, dtype=np.float64)
        # Webots Lidar.getRangeImage() returns 0.0 for "no return at
        # all" in the first uninitialised revolution AND for any ray
        # inside `minRange`; neither is a real hit, so filter both
        # alongside the +inf far-misses.
        finite = np.isfinite(ranges) & (ranges > 0.05)
        if finite.any():
            angles = (lidar_scan.angle_min
                      + np.arange(ranges.size, dtype=np.float64)
                        * lidar_scan.angle_increment)
            world_yaws = syaw + angles[finite]
            r_hit      = ranges[finite]
            hit_xs     = sx + r_hit * np.cos(world_yaws)
            hit_ys     = sy + r_hit * np.sin(world_yaws)
            # Pre-compute pixel coords vectorised, then circle in a
            # tight loop (cv2 has no batch circle primitive).
            cols = ((hit_xs - vcx + vhm) * scale).astype(np.int32)
            rows = (pixels - (hit_ys - vcy + vhm) * scale).astype(np.int32)
            for c, r in zip(cols, rows):
                cv2.circle(canvas, (int(c), int(r)), 2, (0, 220, 240), -1)

    # Grid lines (faint)
    for k in range(0, pixels + 1, int(scale * 0.25)):
        cv2.line(canvas, (k, 0), (k, pixels), (240, 240, 240), 1)
        cv2.line(canvas, (0, k), (pixels, k), (240, 240, 240), 1)

    # A* path (autonomous, green).
    draw_cell_path(path, (40, 180, 40), (40, 140, 40))

    # A* goal-probe route: orange, distinct from the green
    # autonomous path. Only present when the dashboard has a goal set.
    draw_cell_path(probe_path, (0, 140, 255), (0, 110, 220), aa=True)

    # A* goal flag at the clicked point (orange diamond + ring + label).
    if probe_goal is not None:
        col, row = w2px(*probe_goal)
        cv2.drawMarker(canvas, (col, row), (0, 140, 255),
                       cv2.MARKER_DIAMOND, 16, 2)
        cv2.circle(canvas, (col, row), 9, (0, 110, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "A* goal", (col + 11, row + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 110, 220), 1,
                    cv2.LINE_AA)

    # Cup estimate
    if cup_xy is not None:
        col, row = w2px(*cup_xy)
        cv2.drawMarker(canvas, (col, row), (220, 80, 80),
                       cv2.MARKER_CROSS, 16, 2)
        cv2.putText(canvas, "cup", (col + 8, row - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 80, 80), 1,
                    cv2.LINE_AA)

    # Bot
    bx, by = bot_xy
    bcol, brow = w2px(bx, by)
    cv2.circle(canvas, (bcol, brow), 8, (40, 40, 220), -1)
    # Yaw arrow
    ahead_len = 22
    bx2 = bcol + int(ahead_len * math.cos(bot_yaw))
    by2 = brow - int(ahead_len * math.sin(bot_yaw))
    cv2.arrowedLine(canvas, (bcol, brow), (bx2, by2),
                    (40, 40, 220), 2, tipLength=0.4)
    cv2.putText(canvas, "bot", (bcol + 10, brow + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 220), 1, cv2.LINE_AA)

    # Origin compass
    cv2.putText(canvas, "+x", (pixels - 30, pixels - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(canvas, "+y", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return canvas
