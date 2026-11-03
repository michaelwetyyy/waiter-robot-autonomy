"""Wall perception, near-field reactive flavour.

We deliberately do NOT build a global occupancy grid (we tried; see
Build Log section 17: the bottom-edge-of-mask technique can't disambiguate
near-wall-tops from far-wall-feet at a distance, so the accumulated map
fills with phantom walls). Instead we ask one question per frame:

    "Is there a wall within ~30 cm of me, and if so, in which sector?"

This range is where the bottom-edge projection is reliable, because
nothing can occlude a wall right in front of the bot. The output is fed
to a Bug0-style reactive controller that just steers around things as
it sees them, same way you'd navigate a cluttered table by feel.

Pure CV + math. No platform dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from ..config import CameraIntrinsics


# Orange wall HSV range, tightly bracketed around the maze-wall hue
# to keep the e-puck's own orange parts out of the mask.
#
# Maze walls are diffuseColor (1.0, 0.45, 0.0) -> HSV (13, 255, 255)
# rendered. The e-puck PROTO has an orange-coloured stalk on top of
# the body at (1.0, 0.647, 0) -> HSV (19, 255, 255). They're only 6
# hue points apart but distinguishable. Hue gate [10, 17] gives a
# 3-point margin on each side of the wall hue (13) and decisively
# rejects 19.
#
# V floor is 150 (was 80, but that was too generous). Walls under
# AMBIENT-only lighting render at V ≈ 217 (= 0.85 * 255 with our
# new directional-light setup); 150 sits well below that with margin
# but well above any non-orange object's red channel.
#
# S floor 200 (unchanged from previous tightening): all the orange
# diffuseColors in the scene render at S=255 since their B channel
# is zero, so the gate just excludes weakly-saturated reds/browns.
#
# Real robot will need to re-tune this against the lab's actual
# lighting + paint colour.
ORANGE_LO = np.array([ 10, 200, 150])
ORANGE_HI = np.array([ 17, 255, 255])

# Centre-weighted projection: pixels at the horizontal centre of the
# image contribute weight 1.0 to wall-cell votes; pixels at the
# horizontal edges contribute SIDE_WEIGHT_MIN. Smooth cosine roll-off
# between. This down-weights edge projections, which suffer most from
# lens distortion (real Pi cam) and grazing-angle / tilt-mismatch
# artefacts (any camera). Per-pixel weight:
#     dev = |u - W/2| / (W/2)              ∈ [0, 1]
#     w   = SIDE_WEIGHT_MIN + (1 - SIDE_WEIGHT_MIN) · cos(dev · π/2)
SIDE_WEIGHT_MIN = 0.20    # edge pixels still count 20% of a centre pixel

# Range-based confidence, derived from the projection geometry, not
# hand-tuned. A 1-pixel vertical jitter in the bottom-edge mask
# translates to a range error of
#
#     σ_r(r)  ≈  r² / (f_y · h)
#
# where f_y is the vertical focal length in pixels and h is the camera
# height above the floor. This is the dominant geometric noise source
# at our 10° tilt; lens distortion is negligible and the horizon-grazing
# term is already handled separately by `horizon_w` below.
#
# Weighting recipe: model each hit as a Gaussian σ-disk on the ground.
# While σ_r ≤ CELL_M_REF the hit lands reliably in the correct grid
# cell, so it deserves full weight. Beyond that the σ-disk spreads
# over ~(σ_r / cell_m)² cells and the share reaching the correct cell
# drops as the inverse square:
#
#     w(r)  =  min(1,  (cell_m / σ_r)² )
#            =  min(1,  (cell_m · f_y · h / r²)² )
#
# For the hi-res portrait Pi-Cam-v2 (f_y ≈ 2024 px, h ≈ 0.17 m) the
# unit-weight regime extends to r ≈ √(cell_m · f_y · h) ≈ 2.9 m, i.e.
# the curve is essentially flat all the way to the practical edge of
# the grid, and only starts decaying once the projection genuinely
# becomes cell-ambiguous. (The old linear 0.5 -> 2.5 m fade had walls
# at 1.5 m down at half-weight, which is why real walls were taking
# so many scans to commit.)
#
# We still discard hits whose σ_r grows past SIGMA_CAP_CELLS · cell_m
# -- at that point the σ-disk spans so many cells the vote is functionally
# noise. Phantom rejection beyond that is left to the consensus vote
# (real walls accumulate at the same world cell across poses; geometric
# phantoms scatter and get filtered by the wall-vote threshold).
CELL_M_REF      = 0.025    # keep in sync with CONFIG.grid.cell_m
SIGMA_CAP_CELLS = 3.0      # drop hits past ~3·cell_m of range stddev


def orange_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean (255/0) mask of orange pixels in the image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)


def bottom_edge_of_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each image column, return v of the BOTTOM-MOST True pixel.

    Returns (us, vs): pixel coordinates of the wall-meets-floor point in
    every column that has any True pixel. Columns with no True pixels
    are omitted.
    """
    H, _ = mask.shape
    has = mask.any(axis=0)
    if not has.any():
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    us = np.where(has)[0]
    cols = mask[:, us]
    # Last True row in each column = H - 1 - argmax(reversed)
    bottom_v = (H - 1) - np.argmax(cols[::-1, :], axis=0)
    return us, bottom_v


def pixels_to_ground(
    mask: np.ndarray,
    bot_x: float,
    bot_y: float,
    bot_yaw: float,
    intrinsics: CameraIntrinsics,
    target_z: float = 0.0,
) -> np.ndarray:
    """Inverse perspective: project bottom-edge mask pixels to ground.

    Each column's lowest orange pixel represents the WALL-FLOOR contact
    in that direction. We project it onto the plane z = `target_z`.

    Returns (N, 3) numpy array of (world_x, world_y, weight) tuples.
    `weight` is a cosine function of the source pixel's distance from
    the image's horizontal centre (central pixels weight = 1.0, far
    edges = SIDE_WEIGHT_MIN). Lets the caller down-weight noisier
    edge projections.
    """
    us, vs = bottom_edge_of_mask(mask)
    if len(us) == 0:
        return np.empty((0, 3))

    W = intrinsics.width
    H = intrinsics.height
    f_x = intrinsics.f_x
    f_y = intrinsics.f_y
    cam_x = intrinsics.mount_x    # camera X-offset in body frame (e-puck
                                  # PROTO puts the turretSlot 9.5 mm aft
                                  # of body centre, so this is negative)
    cam_z = intrinsics.mount_z
    tilt = intrinsics.tilt           # positive = camera tilted DOWN

    # Pixel angles relative to camera optical axis
    az_rel = np.arctan((us - W / 2.0) / f_x)              # +ve = image right
    el_rel = -np.arctan((vs - H / 2.0) / f_y)             # +ve = image up

    # World-ray elevation: camera tilted DOWN by `tilt`; optical axis
    # elevation = -tilt. The pixel's absolute elevation is then:
    elev_world = -tilt + el_rel
    # Body-frame azimuth: image right maps to bot's RIGHT (-y); body azim
    # (positive = LEFT = +Y body) is -az_rel.
    azim_body  = -az_rel

    sin_el = np.sin(elev_world)
    valid = sin_el < -0.02            # ray must actually point downward
    if not np.any(valid):
        return np.empty((0, 3))

    # Per-pixel centre weight, computed from the source u-coordinate.
    dev = np.abs(us - W / 2.0) / (W / 2.0)
    weights = SIDE_WEIGHT_MIN + (1.0 - SIDE_WEIGHT_MIN) * np.cos(dev * math.pi / 2.0)
    weights = weights[valid]

    # Down-weight near-horizon rays. range = dz / sin_el so a 1-pixel
    # bottom-edge jitter blows up to many cm of range error as
    # sin_el -> 0. Full weight at sin_el ≤ -0.10 (≈ ground within
    # 1.7 m at our 0.17 m camera height), linear fade to zero at the
    # -0.02 `valid` cutoff; grazing hits can reinforce a real wall
    # supported by closer full-weight hits, but can't confirm one
    # on their own.
    horizon_w = np.clip(-sin_el[valid] / 0.10, 0.0, 1.0)
    weights = weights * horizon_w

    dz = target_z - cam_z
    t  = dz / sin_el[valid]
    # (fwd, lat) are the wall point in CAMERA frame (camera origin =
    # camera optical centre). Body frame adds the camera's body-frame
    # position. Without `cam_x` this consistently over-reports forward
    # distance by ~9.5 mm regardless of range.
    fwd = cam_x + t * np.cos(elev_world[valid]) * np.cos(azim_body[valid])
    lat = t * np.cos(elev_world[valid]) * np.sin(azim_body[valid])

    # Range-based confidence: physics-derived inverse-square falloff
    # once σ_r exceeds one grid cell. See CELL_M_REF block above for
    # the derivation. `cam_z` is the camera height h that drops out of
    # dz = target_z - cam_z when target_z = 0.
    horiz    = np.hypot(fwd, lat)
    sigma_r  = horiz * horiz / (f_y * cam_z)
    range_w  = np.minimum(1.0, (CELL_M_REF / np.maximum(sigma_r, 1e-9)) ** 2)
    weights  = weights * range_w
    in_range = sigma_r < SIGMA_CAP_CELLS * CELL_M_REF
    fwd      = fwd[in_range]
    lat      = lat[in_range]
    horiz    = horiz[in_range]
    weights  = weights[in_range]

    # Frontal-incidence weight --------------------------------------------
    # Penalise pixels viewing a wall at grazing angle. The bottom
    # edge of the mask is hardest to extract cleanly when the
    # wall-floor contact line slants near-to-far across the image:
    # a one-pixel mask error displaces the projected hit along the
    # radial, which for a grazing wall pulls the vote well off the
    # actual wall position (the displacement is wall-perpendicular).
    # For a head-on wall the same radial error slides the hit along
    # the wall (harmless). We estimate the local wall direction from
    # the chord between neighbouring surviving columns' projections,
    # then weight by sin² of the angle between the view ray and that
    # chord:
    #     head-on wall  (chord perp ray)  -> weight 1
    #     grazing wall  (chord para ray)  -> weight 0
    # Stacks on top of σ_r; a frontal-but-distant hit still pays the
    # range penalty, a grazing-and-close hit also pays an incidence
    # penalty. Corners are mis-weighted on the one pixel that spans
    # the discontinuity (central difference straddles two walls), but
    # the surrounding pixels are fine and votes accumulate per cell.
    if len(fwd) >= 2:
        # np.gradient with explicit positions handles non-uniform pixel
        # spacing; columns drop out via `valid` and `in_range`, so
        # surviving us values aren't always +1 apart.
        us_in    = us[valid][in_range].astype(np.float64)
        chord_x  = np.gradient(fwd, us_in)
        chord_y  = np.gradient(lat, us_in)
        chord_sq = chord_x * chord_x + chord_y * chord_y
        horiz_s  = np.maximum(horiz, 1e-9)
        dot      = (fwd * chord_x + lat * chord_y) / horiz_s
        incid_w  = np.clip(
            1.0 - (dot * dot) / np.maximum(chord_sq, 1e-12),
            0.0, 1.0)
        weights  = weights * incid_w

    cy = math.cos(bot_yaw); sy = math.sin(bot_yaw)
    wx = bot_x + fwd * cy - lat * sy
    wy = bot_y + fwd * sy + lat * cy

    return np.column_stack((wx, wy, weights))


@dataclass(frozen=True)
class NearField:
    """Closest body-frame distance to an orange surface in five sectors.

    Five sectors total. SEEK / AVOID logic uses only ahead/left/right;
    hard_left & hard_right are reported for the telemetry overlay so
    we can SEE if walls are creeping in on the bot's hips.

    Sectors:
        hard_left  :  azim in (+60°, +90°)
        left       :  azim in (+15°, +60°)
        ahead      :  azim in (-15°, +15°)
        right      :  azim in (-60°, -15°)
        hard_right :  azim in (-90°, -60°)
    """
    ahead:      float
    left:       float
    right:      float
    hard_left:  float
    hard_right: float

    @property
    def ahead_clear(self) -> bool:
        return not math.isfinite(self.ahead)

    def __str__(self) -> str:
        def _fmt(x): return "inf" if math.isinf(x) else f"{x*100:.0f}cm"
        return (f"hL={_fmt(self.hard_left)} L={_fmt(self.left)} "
                f"A={_fmt(self.ahead)} R={_fmt(self.right)} hR={_fmt(self.hard_right)}")


def near_field_clearances(
    rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    near_range_m: float = 0.30,
) -> NearField:
    """Distance to nearest orange surface in each forward sector."""
    mask = orange_mask(rgb)
    pts = pixels_to_ground(mask, 0.0, 0.0, 0.0, intrinsics, target_z=0.0)
    if len(pts) == 0:
        return NearField(math.inf, math.inf, math.inf, math.inf, math.inf)

    fwd, lat = pts[:, 0], pts[:, 1]
    horiz = np.hypot(fwd, lat)
    azim  = np.arctan2(lat, fwd)

    near = (horiz < near_range_m) & (fwd > 0.0)
    if not near.any():
        return NearField(math.inf, math.inf, math.inf, math.inf, math.inf)

    def _sector_min(lo_deg: float, hi_deg: float) -> float:
        m = near & (azim >= math.radians(lo_deg)) & (azim < math.radians(hi_deg))
        return float(horiz[m].min()) if m.any() else math.inf

    return NearField(
        ahead      = _sector_min(-15, +15),
        left       = _sector_min(+15, +60),
        right      = _sector_min(-60, -15),
        hard_left  = _sector_min(+60, +90),
        hard_right = _sector_min(-90, -60),
    )


def near_field_walls(
    rgb: np.ndarray,
    bot_x: float, bot_y: float, bot_yaw: float,
    intrinsics: CameraIntrinsics,
    near_range_m: float = 0.25,
) -> np.ndarray:
    """High-confidence wall positions in world frame.

    Same algorithm as `pixels_to_ground` but returns only the projected
    points within `near_range_m` of the bot.

    Returns (N, 3) array of (world_x, world_y, weight). The third column
    is a centre-of-image weight (see `pixels_to_ground` for the curve)
    that callers can use to down-weight edge-of-image projections.
    """
    mask = orange_mask(rgb)
    pts = pixels_to_ground(mask, bot_x, bot_y, bot_yaw, intrinsics, target_z=0.0)
    if len(pts) == 0:
        return np.empty((0, 3))
    # Distance from bot in world frame
    dx = pts[:, 0] - bot_x
    dy = pts[:, 1] - bot_y
    near = np.hypot(dx, dy) < near_range_m
    return pts[near]
