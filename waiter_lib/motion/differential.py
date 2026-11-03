"""Differential-drive command synthesis: heading PD + wheel-speed clamp.

Pure math. The actual wheel-command write happens through
`SensorSource.set_wheel_velocities()`. This module just decides WHAT to
write. Same code runs on Webots and Pi.
"""

from __future__ import annotations
import math
from typing import Tuple

from ..config import RobotMechanics, Tuning


def wrap_angle(a: float) -> float:
    """Normalise an angle to (-pi, pi]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def clamp_wheel_pair(v_l: float, v_r: float, max_speed: float) -> Tuple[float, float]:
    """Cap wheel speeds while preserving the v_l / v_r ratio.

    This keeps turning behaviour correct when the controller would
    otherwise request impossible speeds. If only one wheel saturates,
    the other is scaled proportionally so the bot still pivots
    correctly, just slower.
    """
    peak = max(abs(v_l), abs(v_r))
    if peak > max_speed:
        scale = max_speed / peak
        v_l *= scale
        v_r *= scale
    return v_l, v_r


def heading_pd(
    dx: float,
    dy: float,
    bot_yaw: float,
    cruise_rad_s: float,
    tuning: Tuning,
    speed_cap_gain: float = 10.0,
) -> Tuple[float, float, float]:
    """Closed-loop heading + distance control to a (dx, dy) offset.

    Returns (v_left, v_right, heading_error). Caller can use
    heading_error for printing / replanning decisions.

    Speed slows as we approach the target (via `min(cruise, gain*dist)`)
    and as the heading error grows (via `cos(head_err)`).
    """
    target_yaw = math.atan2(dy, dx)
    head_err   = wrap_angle(target_yaw - bot_yaw)
    distance   = math.hypot(dx, dy)
    speed      = min(cruise_rad_s, speed_cap_gain * distance) * max(0.0, math.cos(head_err))
    diff       = tuning.turn_gain * head_err
    return speed - diff, speed + diff, head_err
