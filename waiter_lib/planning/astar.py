"""8-connected A* pathfinder over a boolean occupancy grid.

Pure Python + numpy. Same code runs on Webots and Pi.
"""

from __future__ import annotations
import heapq
import math
from typing import List, Optional, Tuple
import numpy as np


Cell = Tuple[int, int]


def a_star(grid: np.ndarray, start: Cell, goal: Cell,
           cost_field: Optional[np.ndarray] = None,
           collect_explored: bool = False):
    """Classic A* with Euclidean heuristic. 8-connected neighbours.

    Returns the path as a list of cells from `start` to `goal` inclusive,
    or None if no path exists (or if either endpoint is blocked).

    `cost_field` (optional): a per-cell additive penalty (same shape as
    `grid`) charged on *entering* a cell, on top of the geometric move
    cost. Used to encode soft preferences (e.g. a clearance penalty that
    keeps the route away from walls; see planning.grid.clearance_cost_field).
    The heuristic stays the plain Euclidean distance, which remains
    admissible as long as the field is non-negative, so A* is still
    optimal for the combined cost.

    `collect_explored`: when True, return `(path, explored)` instead of
    just `path`, where `explored` is the list of cells A* settled (popped
    from the open set) in expansion order. Used by the dashboard to
    visualise the search frontier (the algorithm itself, not just its
    result). Off by default so the autonomous hot path is unchanged.
    """
    dim_x, dim_y = grid.shape
    use_field = cost_field is not None
    explored: list[Cell] = [] if collect_explored else None

    def _ret(path):
        return (path, explored) if collect_explored else path

    if grid[start[0], start[1]] or grid[goal[0], goal[1]]:
        return _ret(None)

    def h(a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_set: list[tuple[float, float, Cell]] = [(h(start, goal), 0.0, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}

    while open_set:
        _, gs, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return _ret(path[::-1])
        if gs > g_score.get(cur, float("inf")):
            continue
        if collect_explored:
            explored.append(cur)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < dim_x and 0 <= ny < dim_y):
                    continue
                if grid[nx, ny]:
                    continue
                # Forbid diagonal moves that "squeeze" between two
                # blocked orthogonal neighbours. Otherwise a path can
                # cut diagonally between a wall corner and a wall edge,
                # which the bot's bounding circle physically intersects.
                if dx != 0 and dy != 0:
                    if grid[cur[0] + dx, cur[1]] or grid[cur[0], cur[1] + dy]:
                        continue
                cost = math.hypot(dx, dy)
                if use_field:
                    cost += float(cost_field[nx, ny])
                tent = gs + cost
                if tent < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = cur
                    g_score[(nx, ny)] = tent
                    heapq.heappush(
                        open_set,
                        (tent + h((nx, ny), goal), tent, (nx, ny)),
                    )
    return _ret(None)


def smooth_path(path: Optional[List[Cell]]) -> Optional[List[Cell]]:
    """Drop collinear waypoints; keep only the corners."""
    if path is None or len(path) <= 2:
        return path
    out: List[Cell] = [path[0]]
    for i in range(1, len(path) - 1):
        a, b, c = out[-1], path[i], path[i + 1]
        if (b[0] - a[0]) * (c[1] - a[1]) != (c[0] - a[0]) * (b[1] - a[1]):
            out.append(b)
    out.append(path[-1])
    return out


def line_of_sight(grid: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if the straight segment a->b crosses no occupied cell.

    Samples the segment at sub-cell resolution (0.25 cell) so no cell
    along the line is skipped. The grid is the INFLATED grid, so a clear
    line of sight already preserves the bot's body clearance. That's
    what makes any-angle shortcutting safe here.
    """
    (x0, y0), (x1, y1) = a, b
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(dist / 0.25))
    for k in range(n + 1):
        t = k / n
        gx = int(round(x0 + t * (x1 - x0)))
        gy = int(round(y0 + t * (y1 - y0)))
        if grid[gx, gy]:
            return False
    return True


def string_pull(grid: np.ndarray, path: Optional[List[Cell]]) -> Optional[List[Cell]]:
    """Any-angle path smoothing via greedy line-of-sight shortcutting.

    Replaces A*'s 45°-quantised staircase with straight diagonal runs:
    starting from an anchor, keep the farthest waypoint still visible in
    a straight line, and only commit a corner where the line of sight
    finally breaks. Produces a sparser, shorter, straighter waypoint list
    than `smooth_path` (which only drops perfectly-collinear cells).

    Output is the same list-of-cells type as `smooth_path`, so it's a
    drop-in replacement for downstream consumers.
    """
    if path is None or len(path) <= 2:
        return path
    out: List[Cell] = [path[0]]
    anchor = path[0]
    last_visible = path[1]
    for nxt in path[2:]:
        if line_of_sight(grid, anchor, nxt):
            last_visible = nxt
        else:
            out.append(last_visible)
            anchor = last_visible
            last_visible = nxt
    out.append(path[-1])
    return out
