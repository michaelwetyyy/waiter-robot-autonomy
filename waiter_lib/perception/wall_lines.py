"""Line-segment extraction from the committed wall vote map.

Phase 1 of the wall-based SLAM upgrade. The cell grid in
`world.wall_status` is the right thing for A* (cheap point-in-set
collision check) but a poor geometric primitive: a single straight
wall is stored as a noisy row of voted cells with one-cell jitter,
which is messy to visualise and useless as a feature for ICP.

This module fits straight line segments to the connected wall blobs
in the grid so downstream code has clean geometry to work with:

  - the dashboard draws crisp lines instead of pixel clouds
  - the upcoming ICP step (Phase 2) gets stable point-to-line residuals
    to solve drift against

We do NOT touch the grid; A* and DWA still read `wall_status`
directly. Segments are an auxiliary representation derived from it.

Algorithm per cycle:
  1. Connected-components on the binary wall mask (8-connectivity).
  2. For each component, run progressive PCA line fitting:
     fit primary direction, peel off inliers within an
     INLIER_EPS_CELLS band, repeat on the residual until either too
     few points remain or no acceptable line is found.
  3. For each inlier set, project onto the line and split at gaps
     wider than MAX_SPLIT_GAP_CELLS so an L-shape blob becomes two
     segments rather than one line straddling the corner.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


# Cells in a component below this are dropped as noise rather than fit.
MIN_CELLS_PER_COMPONENT = 4
# Smallest accepted segment, in cells of inlier support.
MIN_CELLS_PER_SEGMENT   = 3
# Perpendicular inlier band, in cells. Has to be wider than the
# one-cell perpendicular spread a real wall gets at long range (σ_r
# grows quadratically with distance; see walls.py CELL_M_REF block),
# but small enough not to swallow parallel walls on the other side of
# a corridor (corridors in the maze are ≥4 cells wide). 1.2 cells =
# 3.0 cm comfortably covers a 1-cell perpendicular jitter without
# straddling parallel walls.
INLIER_EPS_CELLS        = 1.2
# Split a fitted line into multiple segments at gaps wider than this
# along the line direction. Tuned so a contiguous wall stays one
# segment even when one or two cells dropped out of the vote.
MAX_SPLIT_GAP_CELLS     = 2.5
# RANSAC passes per blob. Most components are one straight segment;
# an L-corner takes 2 passes; a T-junction or a + intersection takes
# 3. Beyond 4 we're fitting noise to noise.
MAX_PASSES_PER_COMPONENT = 4
# RANSAC sample iterations per pass. Capped so we don't burn cycles on
# tiny blobs (n*(n-1)/2 pairs total exist anyway). For a 50-point blob
# 50 random samples gives >99% chance of hitting two points on the
# dominant line if it has ≥40% inliers.
RANSAC_ITERS_PER_PASS   = 64


@dataclass(frozen=True)
class WallSegment:
    """One straight wall segment in world (odometry) frame, metres."""
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    # Unit normal perpendicular to the segment. Sign is arbitrary
    # (no walkable-side knowledge here); ICP only needs |normal|=1.
    normal: Tuple[float, float]
    # Number of grid cells contributing to this segment's inlier set.
    # Acts as a confidence weight for downstream consumers.
    support: int

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])

    @property
    def midpoint(self) -> Tuple[float, float]:
        return (0.5 * (self.p0[0] + self.p1[0]),
                0.5 * (self.p0[1] + self.p1[1]))


def extract_segments(
    wall_status: np.ndarray,
    cell_m: float,
    half_m: float,
) -> List[WallSegment]:
    """Fit line segments to every connected wall blob in the grid.

    `wall_status` is the (dim, dim) bool array of committed wall cells.
    Grid index to world coordinate matches the overlay's convention:
        wx = (gx + 0.5) * cell_m - half_m       (first dim -> x)
        wy = (gy + 0.5) * cell_m - half_m       (second dim -> y)
    """
    if wall_status is None or not wall_status.any():
        return []

    # 8-connectivity so diagonal cell chains stay in one component.
    n_labels, labels = cv2.connectedComponents(
        wall_status.astype(np.uint8), connectivity=8)

    segments: List[WallSegment] = []
    eps_m       = INLIER_EPS_CELLS    * cell_m
    max_gap_m   = MAX_SPLIT_GAP_CELLS * cell_m

    for label_id in range(1, n_labels):
        gx, gy = np.where(labels == label_id)
        if len(gx) < MIN_CELLS_PER_COMPONENT:
            continue
        wx = (gx + 0.5) * cell_m - half_m
        wy = (gy + 0.5) * cell_m - half_m
        pts = np.column_stack([wx, wy])
        segments.extend(_fit_blob(pts, eps_m, max_gap_m))

    return segments


def _fit_blob(
    pts: np.ndarray,
    eps_m: float,
    max_gap_m: float,
) -> List[WallSegment]:
    """RANSAC-then-PCA: propose a line by sampling random point pairs,
    pick the proposal with the most inliers, refine via PCA on that
    consensus set, peel its inliers off, repeat.

    Plain progressive PCA on the whole blob doesn't work for L/T
    corners: the centroid of an L-shape sits OFF both legs, so the
    principal axis perpendicular band catches neither. RANSAC's
    line-through-two-points proposal sidesteps that: any pair on the
    same leg defines a leg-aligned line, and the dominant leg wins by
    inlier count.
    """
    out: List[WallSegment] = []
    remaining = pts.copy()
    # Deterministic so dashboards are stable scan-to-scan and tests
    # are reproducible. A different seed every call would make the
    # segment list flicker between equally-valid orientations.
    rng = np.random.default_rng(0)

    for _ in range(MAX_PASSES_PER_COMPONENT):
        n = len(remaining)
        if n < MIN_CELLS_PER_SEGMENT:
            break

        # RANSAC proposal
        best_n     = 0
        best_mask  = None
        n_iters    = min(RANSAC_ITERS_PER_PASS, n * (n - 1) // 2)
        for _i in range(n_iters):
            i, j = rng.choice(n, size=2, replace=False)
            d    = remaining[j] - remaining[i]
            dn   = float(np.hypot(d[0], d[1]))
            if dn < 1e-9:
                continue
            d      = d / dn
            normal = np.array([-d[1], d[0]])
            perp   = (remaining - remaining[i]) @ normal
            mask   = np.abs(perp) <= eps_m
            k      = int(mask.sum())
            if k > best_n:
                best_n, best_mask = k, mask

        if best_n < MIN_CELLS_PER_SEGMENT or best_mask is None:
            break

        # PCA refinement on the consensus set, then re-cut inliers
        # using the refined line. The refinement matters because the
        # 2-point RANSAC proposal can be skewed by a single bad
        # endpoint; PCA on the whole consensus set finds the true
        # principal direction of the wall.
        direction, centroid = _pca_principal(remaining[best_mask])
        normal = np.array([-direction[1], direction[0]])
        perp   = (remaining - centroid) @ normal
        mask   = np.abs(perp) <= eps_m
        if int(mask.sum()) < MIN_CELLS_PER_SEGMENT:
            break

        inliers = remaining[mask]
        t       = (inliers - centroid) @ direction
        out.extend(_segments_from_projection(
            t, direction, centroid, normal, max_gap_m))
        remaining = remaining[~mask]

    return out


def _pca_principal(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unit_direction, centroid) of the (N,2) point set."""
    centroid = pts.mean(axis=0)
    centred  = pts - centroid
    cov      = centred.T @ centred           # 2x2
    # np.linalg.eigh returns eigvals ascending; last col = largest var.
    _, eigvecs = np.linalg.eigh(cov)
    direction  = eigvecs[:, -1]
    n = float(np.linalg.norm(direction))
    if n < 1e-12:
        # Degenerate (all points identical): pick an arbitrary axis;
        # downstream MIN_CELLS_PER_SEGMENT will reject it anyway.
        return np.array([1.0, 0.0]), centroid
    return direction / n, centroid


def _segments_from_projection(
    t: np.ndarray,
    direction: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    max_gap_m: float,
) -> List[WallSegment]:
    """Sort inliers along the line, split at any gap > max_gap_m."""
    order   = np.argsort(t)
    t_sort  = t[order]
    gaps    = np.diff(t_sort)
    cut_idx = np.where(gaps > max_gap_m)[0]
    starts  = [0, *(cut_idx + 1)]
    ends    = [*(cut_idx + 1), len(t_sort)]

    out: List[WallSegment] = []
    for s, e in zip(starts, ends):
        if (e - s) < MIN_CELLS_PER_SEGMENT:
            continue
        p0 = centroid + t_sort[s]      * direction
        p1 = centroid + t_sort[e - 1]  * direction
        out.append(WallSegment(
            p0      = (float(p0[0]), float(p0[1])),
            p1      = (float(p1[0]), float(p1[1])),
            normal  = (float(normal[0]), float(normal[1])),
            support = int(e - s),
        ))
    return out
