"""Helpers for snapping goal/start cells to reachable free cells.

When perception places the goal inside a wall (or just outside the
arena), we sidestep "no path" by snapping to the nearest free cell.
"""

from __future__ import annotations
import numpy as np


def nearest_free(grid: np.ndarray, gx: int, gy: int) -> tuple[int, int]:
    """Spiral outward from (gx, gy) and return the first free cell.

    If the seed cell is already free, returns it unchanged. If the whole
    grid is blocked, returns the seed (caller should handle that case).
    """
    dim_x, dim_y = grid.shape
    if not grid[gx, gy]:
        return (gx, gy)
    for r in range(1, max(dim_x, dim_y)):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                # Only test the perimeter of the r-radius square
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = gx + dx, gy + dy
                if not (0 <= nx < dim_x and 0 <= ny < dim_y):
                    continue
                if not grid[nx, ny]:
                    return (nx, ny)
    return (gx, gy)
