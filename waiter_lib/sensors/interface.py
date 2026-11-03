"""SensorSource Protocol: the only platform-specific surface in the
controller. Both Webots and Pi implementations conform to this.

Anything that touches a Webots API (`controller.Robot`, `device.*`, etc.)
lives ONLY inside `sensors/webots_source.py`. No other file in the
project may import from `controller`. This is enforceable with:

    grep -r 'from controller' waiter_lib/ | grep -v sensors/webots_source.py

returning empty.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass
class WheelEncoders:
    """Cumulative wheel angular position from each encoder (radians).

    The state machine differences these between ticks to derive distance
    travelled and heading change. Either implementation may reset its
    accumulators at construction, but should never reset them mid-run.
    """
    left_rad:  float
    right_rad: float


@dataclass
class ImuReading:
    """Optional IMU bundle. yaw is CCW-positive about world +Z. Pitch and
    roll are unused today but reserved for future balance integration on
    the real robot."""
    yaw:   float
    pitch: float
    roll:  float


@dataclass
class LidarScan:
    """One revolution from the 360° LiDAR (DFRobot DFR0315 / RPLIDAR
    A1-class). Distances in metres, in the sensor's local frame. Rays are
    swept CCW about local +Z. Returns outside [minRange, maxRange] are
    reported as `math.inf` (no return).

    Sample count and angular spacing are fixed by the device's
    `horizontalResolution` field in the world file (default 360 rays =
    1°/ray). The state machine doesn't consume this yet; exposed on
    WorldState so downstream code (ICP / SLAM stretch goal) can opt in.
    """
    ranges:          np.ndarray   # (N,) float32, metres; +inf = no return
    angle_min:       float        # radians (typically -pi)
    angle_increment: float        # radians per ray (CCW)


class SensorSource(Protocol):
    """The only platform-specific surface in the controller."""

    # Lifecycle
    def tick(self) -> bool:
        """Advance one timestep. Returns False when the simulation /
        process is signalled to stop. The FSM main loop calls `step()`
        for as long as `tick()` returns True."""
        ...

    def time(self) -> float:
        """Seconds since process start. Used for elapsed-time decisions."""
        ...

    def dt(self) -> float:
        """Seconds per tick; used by integrators and PD gains."""
        ...

    # Reads
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """HxWx3 RGB uint8, or None if no fresh frame is available."""
        ...

    def get_encoders(self) -> WheelEncoders: ...

    def get_imu(self) -> Optional[ImuReading]:
        """May return None when no IMU is connected; the FSM falls back
        to wheel-encoder-only odometry."""
        ...

    def get_lidar(self) -> Optional[LidarScan]:
        """Latest full 360° LiDAR scan, or None if not connected / no
        fresh scan since the last call. Implementations should return a
        fresh `LidarScan` no more often than the device's native rate
        (~5.5 Hz on DFR0315) and None in between."""
        ...

    # Writes
    def set_wheel_velocities(self, left_rad_s: float, right_rad_s: float) -> None:
        """Set commanded wheel angular velocity (rad/s).

        Both implementations are responsible for converting this into
        whatever their actuators want (Webots' RotationalMotor accepts
        rad/s directly; the Pi will send the value over serial to the
        ESP32 which has stepper-rate units)."""
        ...

    def set_leds(self, on: bool) -> None:
        """Visual feedback. Webots: all 8 e-puck LEDs on. Pi: a status
        LED on a GPIO pin (TBD)."""
        ...
