"""BrainSensorSource: adapts the live D500 Pi proxy to the waiter_lib
`SensorSource` Protocol, so the *unmodified* WaiterFSM brain (the Webots
controller's A* + DWA + RECAL + cup mission) can drive the real robot.

Two impedance mismatches sit between the brain and the D500 hardware, and
this class is the only place they're resolved:

  ODOMETRY.  The brain wants cumulative per-wheel encoder angle (rad) and an
  absolute IMU yaw; it differences the encoders into a forward step `d_s`
  (state_machine._update_odometry) and overrides heading with the IMU.
  The D500 ESP32 reports neither: it streams `O,<t_ms>,<v_mps>,<yaw_rad>`
  (forward speed + integrated gyro heading). We integrate v over the ESP32's
  own timestamps into a forward distance `s`, then expose SYNTHETIC equal
  encoders left=right=s/wheel_radius. Because the brain immediately multiplies
  back by wheel_radius, the radius cancels and `d_s` recovers the true metres
  regardless of which radius we use; heading comes straight from get_imu().

  ACTUATION.  The brain emits continuous per-wheel angular velocities
  (set_wheel_velocities, rad/s). The balancing D500 takes discrete drive
  chars on two INDEPENDENT firmware axes (forward w/s + turn a/d). We split
  the wheel pair into forward = (l+r)/2 and turn = (r-l)/2 and send BOTH axes
  (each above its own deadband) via remote.set_axes, so the proxy round-robins
  them and the bot can ARC — DWA's (v, w) becomes a real curve instead of the
  old turn-OR-straight bang-bang.

LEDs have no D500 channel, so set_leds is a no-op.

The network itself is already handled by mac.remote_source.RemoteSensorSource
(background drain thread, drop-oldest); this class only wraps it.
"""

from __future__ import annotations
import math
import time
from typing import Optional

import numpy as np
import cv2

# IMX219 on this rig is green-deficient under white light (a white target reads
# magenta; green ~half of red/blue). Boost the green channel so the dashboard
# feed looks natural AND MediaPipe / ArUco get ~natural skin tones. 1.0 = off.
CAM_GREEN_GAIN = 1.7

# waiter_lib is put on sys.path by the runner (run_brain.py) BEFORE this
# module is imported, so the brain and the adapter share ONE definition of
# these dataclasses rather than a local mirror that could drift.
from waiter_lib.sensors.interface import WheelEncoders, ImuReading, LidarScan

from mac.remote_source import RemoteSensorSource


class BrainSensorSource:
    """waiter_lib SensorSource backed by the live D500 proxy."""

    def __init__(self, pi_host: str, *, wheel_radius_m: float,
                 tick_hz: float = 30.0, recv_timeout_s: float = 2.0,
                 frame_rot_k: int = 0,
                 fwd_deadband_rad_s: float = 0.4,
                 turn_deadband_rad_s: float = 0.4,
                 turn_ccw_char: str = "a"):
        self._remote = RemoteSensorSource(pi_host, recv_timeout_s, frame_rot_k)
        self._r = wheel_radius_m
        self._tick_s = 1.0 / tick_hz
        self._t0 = time.monotonic()
        self._last_tick = self._t0

        # Odometry integration (from the ESP32's v_mps + yaw_rad stream).
        self._dist = 0.0                 # cumulative forward distance (m)
        self._yaw = 0.0                  # latest absolute heading (rad)
        self._have_yaw = False
        self._prev_t_ms: Optional[float] = None
        # "Parked": set while the dashboard brain-disconnect toggle is OFF. The
        # bot isn't being driven (and can't translate), so don't integrate the
        # reported velocity into the pose — otherwise the marker drifts even
        # though the toggle says "balances in place". Heading still tracks.
        self._parked = False

        # Wheel-velocity -> two-axis drive split. Each axis (forward, turn) is
        # emitted independently above its deadband, so DWA's (v, w) becomes a
        # real arc on the bot (see set_wheel_velocities).
        self._fwd_deadband = fwd_deadband_rad_s
        self._turn_deadband = turn_deadband_rad_s
        # 'a' / 'd' map to -/+ turn_cmd on the ESP32 (firmware main.cpp).
        # Which one is CCW (+yaw) depends on motor wiring + YAW_SIGN, so it's
        # a single knob here: if the bot turns the wrong way, flip this char.
        self._turn_ccw = turn_ccw_char
        self._turn_cw = "d" if turn_ccw_char == "a" else "a"

        # Green-deficiency correction LUT (precomputed once): the IMX219 reads
        # magenta under white light. A 256-entry lookup is a single C-level pass
        # over the green channel — no per-frame float temporaries. None = off.
        self._green_lut = None
        if CAM_GREEN_GAIN != 1.0:
            self._green_lut = np.clip(
                np.arange(256, dtype=np.float32) * CAM_GREEN_GAIN, 0, 255
            ).astype(np.uint8)

        # Latest DISPLAY frame (clean frame + hand-skeleton overlay), exposed via
        # get_display_frame() for the dashboard. Kept SEPARATE from the frame
        # get_camera_frame() returns so perception (near-field/ArUco) never sees
        # the overlay. None until the first frame.
        self._display_rgb = None

        # Open-palm gesture detection (MediaPipe). Drawn onto the dashboard
        # camera frame so the hand is visibly mapped. Lazily imported so a
        # missing/broken mediapipe can't take the brain down.
        self._hand = None
        try:
            from pi.gesture_detector import HandGestureDetector
            hd = HandGestureDetector()
            self._hand = hd if getattr(hd, "available", False) else None
            print(f"[brain_source] hand detector: "
                  f"{'ON' if self._hand else 'OFF (mediapipe unavailable)'}",
                  flush=True)
        except Exception as e:                       # pragma: no cover
            print(f"[brain_source] hand detector unavailable: {e}", flush=True)

    # -- lifecycle (runner loop drives these) ------------------------------
    def tick(self) -> bool:
        """Pace to tick_hz, fold the freshest odometry sample into the
        running distance/heading estimate. Returns False if the Pi link
        has gone silent (so the runner can hold instead of dying)."""
        now = time.monotonic()
        sleep = self._tick_s - (now - self._last_tick)
        if sleep > 0:
            time.sleep(sleep)
        self._last_tick = time.monotonic()

        o = self._remote.get_odometry()      # (t_ms, v, yaw, moving) or None
        if o is not None:
            t_ms, v, yaw, _moving = o
            # The ESP32 can emit a non-finite yaw (NaN/inf) when the IMU
            # hasn't initialised or the wrong firmware is flashed. Feeding
            # that into the brain poisons heading forever (wrap_angle(nan)
            # stays nan; SEARCH's one-revolution exit never fires). So only
            # accept FINITE samples: hold the last good yaw, skip integrating
            # a bad v. get_imu() reports None until a first finite yaw lands,
            # which cleanly drops the FSM to encoder-only odometry meanwhile.
            if self._prev_t_ms is not None and math.isfinite(v) and not self._parked:
                dt = (t_ms - self._prev_t_ms) / 1000.0
                # Guard against a repeated sample (dt==0) and against a
                # link-blip gap that would integrate one stale v over a
                # huge interval and lurch the pose.
                if 0.0 < dt < 1.0:
                    self._dist += v * dt
            self._prev_t_ms = t_ms
            if math.isfinite(yaw):
                self._yaw = yaw
                self._have_yaw = True

        return self._remote.link_ok()

    def time(self) -> float:
        return time.monotonic() - self._t0

    def dt(self) -> float:
        return self._tick_s

    # -- reads -------------------------------------------------------------
    def get_camera_frame(self):
        """CLEAN frame for perception (near-field clearances + ArUco). Colour-
        corrected but NOT annotated — the hand skeleton goes only on the display
        frame (get_display_frame) so it can't be misread as a near obstacle."""
        rgb = self._remote.get_camera_frame()       # RGB HxWx3 or None
        if rgb is None:
            return None                              # no new frame; hold display
        # Green-deficiency correction (LUT: one C pass, no float temporaries).
        # Neutral for ArUco's grayscale, better colour for MediaPipe.
        if self._green_lut is not None:
            rgb = rgb.copy()
            rgb[:, :, 1] = cv2.LUT(rgb[:, :, 1], self._green_lut)
        # Gesture detection runs on the clean RGB frame (what MediaPipe wants).
        # The skeleton is drawn onto a SEPARATE display copy — and only when a
        # hand is actually present, so we don't pay two colour conversions per
        # idle frame. draw_debug expects BGR, so round-trip the colour space
        # (otherwise its overlay colours are red/blue-swapped).
        self._display_rgb = rgb
        if self._hand is not None:
            try:
                self._hand.process(rgb)
                if self._hand.has_hands():
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgr = self._hand.draw_debug(bgr)
                    self._display_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        return rgb

    def get_display_frame(self):
        """Dashboard frame: the clean frame plus the hand skeleton when a hand
        is visible. Polled by the telemetry path; perception never sees it."""
        return self._display_rgb

    def get_open_palm(self):
        """(cx, cy, size) of the most prominent OPEN palm in the last camera
        frame, or None. cx,cy normalised [0,1] (cx: 0=left, 1=right); size is
        the normalised hand extent (rough proximity proxy — bigger = closer).
        Populated by the detector run in get_camera_frame(); read by
        run_brain's follow-palm behaviour. Open palm only, by construction."""
        if self._hand is None:
            return None
        best, best_size = None, 0.0
        for lm, open_, cx, cy in getattr(self._hand, "_last_detected_hands", []):
            if not open_:
                continue
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            size = max(max(xs) - min(xs), max(ys) - min(ys))
            if size > best_size:
                best, best_size = (float(cx), float(cy), float(size)), size
        return best

    def get_encoders(self) -> WheelEncoders:
        e = self._dist / self._r
        return WheelEncoders(e, e)                   # synthetic, equal

    def get_imu(self) -> Optional[ImuReading]:
        if not self._have_yaw:
            return None
        # pitch carries the ESP's balance tilt angle (rad) when available; the
        # brain only uses yaw for heading, but the dashboard shows tilt. roll
        # has no sensor on this single-axis balancer, so it stays 0.
        tilt = self._remote.get_tilt()
        pitch = tilt if (tilt is not None and math.isfinite(tilt)) else 0.0
        return ImuReading(self._yaw, pitch, 0.0)

    def get_lidar(self) -> Optional[LidarScan]:
        l = self._remote.get_lidar()                 # (ranges, amin, ainc) or None
        if l is None:
            return None
        ranges, amin, ainc = l
        return LidarScan(ranges, amin, ainc)

    # -- writes ------------------------------------------------------------
    def set_wheel_velocities(self, left_rad_s: float, right_rad_s: float) -> None:
        # Decompose DWA's wheel pair into the ESP's two INDEPENDENT firmware
        # axes and send BOTH at once: forward (w/s) AND turn (a/d). The proxy
        # round-robins them onto the serial wire, so the bot can ARC/diagonal
        # rather than the old turn-OR-straight bang-bang (which discarded the
        # simultaneity DWA computes). Each axis idles below its own deadband.
        fwd = 0.5 * (left_rad_s + right_rad_s)
        turn = 0.5 * (right_rad_s - left_rad_s)
        forward = ("w" if fwd >= self._fwd_deadband
                   else "s" if fwd <= -self._fwd_deadband else None)
        turn_c = (self._turn_ccw if turn >= self._turn_deadband
                  else self._turn_cw if turn <= -self._turn_deadband else None)
        self._remote.set_axes(forward, turn_c)

    def set_leds(self, on: bool) -> None:
        pass                                         # no LED channel on D500

    def calibrate_yaw(self) -> None:
        """Dashboard 'Calibrate yaw' button: tell the ESP to re-measure and
        persist its yaw-gyro zero-rate bias (robot should be held still)."""
        self._remote.calibrate_yaw()

    def set_parked(self, parked: bool) -> None:
        """Freeze/resume pose translation. Driven by the dashboard brain-
        disconnect toggle: while disconnected the bot isn't driving, so the
        dead-reckoned position must hold instead of drifting on reported speed."""
        self._parked = bool(parked)

    def drive_manual(self, fwd: int, turn: int) -> None:
        """Dashboard joystick teleop: drive the ESP's two discrete axes directly,
        bypassing the FSM/DWA. fwd in {-1,0,+1} (forward/back), turn in {-1,0,+1}
        (+1 = CCW/left, -1 = CW/right). Bang-bang only: the firmware has no speed
        channel, so the stick sets direction, not magnitude. Same two-axis
        set_axes path the autonomous drive uses (see set_wheel_velocities)."""
        forward = "w" if fwd > 0 else "s" if fwd < 0 else None
        turn_c = (self._turn_ccw if turn > 0
                  else self._turn_cw if turn < 0 else None)
        self._remote.set_axes(forward, turn_c)

    def close(self) -> None:
        try:
            self._remote.set_direction("-")
        finally:
            self._remote.close()
