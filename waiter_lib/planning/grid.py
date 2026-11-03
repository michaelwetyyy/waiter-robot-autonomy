"""Occupancy grid math: world/grid coordinate conversion + inflation.

Pure numpy. Same code runs on Webots and Pi.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from ..config import GridConfig


def world_to_grid_checked(x: float, y: float, cfg: GridConfig
                          ) -> Tuple[Tuple[int, int], bool]:
    """Convert world coords (m) to a grid cell, also returning whether clamping occurred.

    `clamped` is True if either axis was outside the grid bounds and
    got pinned to an edge cell. Callers that care about OOB (e.g. the
    planner, where silently routing to the grid edge is the kind of bug
    that eats days) should use this; callers that don't can use the
    plain `world_to_grid` wrapper below.
    """
    raw_x = int((x + cfg.half_m) / cfg.cell_m)
    raw_y = int((y + cfg.half_m) / cfg.cell_m)
    gx = max(0, min(cfg.dim - 1, raw_x))
    gy = max(0, min(cfg.dim - 1, raw_y))
    return ((gx, gy), (gx != raw_x) or (gy != raw_y))


def world_to_grid(x: float, y: float, cfg: GridConfig) -> Tuple[int, int]:
    """World (m) to grid cell, clamped to grid bounds silently."""
    return world_to_grid_checked(x, y, cfg)[0]


def grid_to_world(gx: int, gy: int, cfg: GridConfig) -> Tuple[float, float]:
    """Grid cell to world (m) at cell centre."""
    return ((gx + 0.5) * cfg.cell_m - cfg.half_m,
            (gy + 0.5) * cfg.cell_m - cfg.half_m)


def occupied_bbox_world(occ: np.ndarray, cfg: GridConfig
                        ) -> Optional[Tuple[float, float, float, float]]:
    """World-frame bounding box of all occupied cells: (xmin, xmax, ymin, ymax).

    Returns None if nothing is occupied.
    """
    if not occ.any():
        return None
    xs, ys = np.where(occ)
    xmin = (xs.min() + 0.5) * cfg.cell_m - cfg.half_m
    xmax = (xs.max() + 0.5) * cfg.cell_m - cfg.half_m
    ymin = (ys.min() + 0.5) * cfg.cell_m - cfg.half_m
    ymax = (ys.max() + 0.5) * cfg.cell_m - cfg.half_m
    return float(xmin), float(xmax), float(ymin), float(ymax)


def render_ascii(occ: np.ndarray, cfg: GridConfig,
                  bot_world: Optional[Tuple[float, float]] = None,
                  goal_world: Optional[Tuple[float, float]] = None,
                  zoom_to_action: bool = True) -> str:
    """ASCII art of the occupancy grid.

    `#` = occupied, `.` = free, `B` = bot, `G` = goal.

    Cropped to the bounding box of (occupied cells + bot + goal) for
    readability; otherwise dumps the full grid (which is unreadable at
    60x60).
    """
    dim_x, dim_y = occ.shape
    # Compute crop window
    if zoom_to_action:
        bbox = occupied_bbox_world(occ, cfg)
        xs_lo, xs_hi = 0, dim_x - 1
        ys_lo, ys_hi = 0, dim_y - 1
        bb_xs, bb_ys = [], []
        if bbox is not None:
            bb_xs += [int((bbox[0] + cfg.half_m) / cfg.cell_m),
                      int((bbox[1] + cfg.half_m) / cfg.cell_m)]
            bb_ys += [int((bbox[2] + cfg.half_m) / cfg.cell_m),
                      int((bbox[3] + cfg.half_m) / cfg.cell_m)]
        for pt in (bot_world, goal_world):
            if pt is not None:
                bb_xs.append(int((pt[0] + cfg.half_m) / cfg.cell_m))
                bb_ys.append(int((pt[1] + cfg.half_m) / cfg.cell_m))
        if bb_xs:
            pad = 2
            xs_lo = max(0, min(bb_xs) - pad)
            xs_hi = min(dim_x - 1, max(bb_xs) + pad)
            ys_lo = max(0, min(bb_ys) - pad)
            ys_hi = min(dim_y - 1, max(bb_ys) + pad)
    else:
        xs_lo, xs_hi, ys_lo, ys_hi = 0, dim_x - 1, 0, dim_y - 1

    bot_cell  = (None if bot_world  is None
                 else world_to_grid(bot_world[0],  bot_world[1],  cfg))
    goal_cell = (None if goal_world is None
                 else world_to_grid(goal_world[0], goal_world[1], cfg))

    lines = []
    # Y axis runs north on top; iterate high to low for natural look
    for y in range(ys_hi, ys_lo - 1, -1):
        row = []
        for x in range(xs_lo, xs_hi + 1):
            if bot_cell is not None and (x, y) == bot_cell:    row.append('B')
            elif goal_cell is not None and (x, y) == goal_cell: row.append('G')
            elif occ[x, y]:                                    row.append('#')
            else:                                              row.append('·')
        lines.append(''.join(row))
    return '\n'.join(lines)


def clearance_field(grid: np.ndarray, cfg: GridConfig) -> np.ndarray:
    """Per-cell distance (metres) from each FREE cell to the nearest
    occupied cell, via an exact Euclidean distance transform.

    Pass the *inflated* grid: the result is then "clearance beyond the
    hard safety margin", which is what the planner's soft clearance cost
    wants. Occupied cells read 0. Used by A* (via a cost field) to prefer
    routes through the middle of free space rather than hugging the
    inflation boundary.

    cv2.distanceTransform measures, for every non-zero pixel, the distance
    to the nearest zero pixel, so we feed free=1, occupied=0.
    """
    free = (~grid).view(np.uint8)
    dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    return dist_cells.astype(np.float32) * cfg.cell_m


def clearance_cost_field(grid: np.ndarray, cfg: GridConfig) -> Optional[np.ndarray]:
    """Additive per-cell A* entry penalty derived from `clearance_field`.

    Cells closer than `cfg.clearance_pref_m` to an obstacle cost extra,
    scaling linearly from `cfg.clearance_weight` (right at the inflation
    boundary) down to 0 (at or beyond the preferred clearance). Returns
    None when the feature is disabled (`clearance_weight <= 0`), so A*
    falls back to pure shortest-path.
    """
    pref = getattr(cfg, "clearance_pref_m", 0.0)
    weight = getattr(cfg, "clearance_weight", 0.0)
    if weight <= 0.0 or pref <= 0.0:
        return None
    clear_m = clearance_field(grid, cfg)
    deficit = np.clip(pref - clear_m, 0.0, pref)           # metres missing
    return ((weight / pref) * deficit).astype(np.float32)   # 0 .. weight


def inflate_grid(occ: np.ndarray, inflate_m: float, cfg: GridConfig) -> np.ndarray:
    """Morphological dilation by `inflate_m` (radius in metres).

    Returns a new boolean grid with True cells expanded into nearby cells
    using a circular structuring element. Implemented with cv2.dilate
    for speed (a 200x200 dilation runs in microseconds vs ~5 ms for
    the equivalent shifted-OR loop in pure numpy).
    """
    cells_r = int(math.ceil(inflate_m / cfg.cell_m))
    if cells_r <= 0:
        return occ.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * cells_r + 1, 2 * cells_r + 1))
    dilated = cv2.dilate(occ.view(np.uint8), kernel)
    return dilated.astype(bool)
