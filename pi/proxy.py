"""Pi-side proxy: thin sensor/actuator bridge for the waiter bot.

Runs on the Pi 3A+. Does no mapping, vision, or planning (all of that moved to
the Mac, because the Pi 3A+ has ~512 MB RAM and can't run MediaPipe + mapping).
It only:
  - reads the D500 LiDAR (raw revolutions) and PUBs each scan to the Mac
  - reads the ESP32 odometry/power/tilt lines and PUBs them
  - captures the camera (hardware MJPEG, off-CPU) and PUBs JPEG frames
  - receives drive commands from the Mac (single-axis w/s/a/d, two-axis arcs,
    or a calibrate request) and relays them to the ESP32, refreshing the held
    drive char at ~25 Hz (under the firmware's key timeout)
  - watchdog: if no command arrives within CMD_TIMEOUT_S, force a STOP so the
    robot never keeps driving on a dead link

Per-subsystem flags so the rig is brought up one piece at a time and a missing
subsystem can't crash the proxy:
    python3 -m pi.proxy --lidar --serial /dev/ttyUSB0           # nav core
    python3 -m pi.proxy --lidar --camera --serial /dev/ttyUSB0  # everything
    python3 -m pi.proxy                                         # all stub

The ESP32 and D500 are two separate USB serial devices — point --serial and
--lidar-port at different /dev nodes.
"""

from __future__ import annotations
import argparse
import contextlib
import io
import os
import sys
import threading
import time

import numpy as np
import zmq

# Allow `python3 -m pi.proxy` (from repo root) and `python3 proxy.py` (from pi/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from transport import protocol as P          # noqa: E402
from lidar_d500 import LidarD500              # noqa: E402

# LiDAR self-clutter + mounting calibration. The three mast poles stand in the
# scan plane and return within ~14 cm of the sensor; drop anything closer so they
# don't smear arcs into the map. angle_offset_deg / ccw align ray 0 with the
# robot's forward axis (calibrate: object straight ahead should land at the top
# of the map). blank_sectors_deg drops fixed self-obstruction angles entirely
# (discover them with `lidar_test.py --poles`): [(centre_deg, half_width_deg), …].
# min_range is now just a small backstop for sensor-hub clutter — the poles are
# handled by the driver's angle-based auto-mask (pole_max_range), so we DON'T
# raise min_range high (that would blind the bot to genuinely close obstacles in
# every direction). Raise LIDAR_POLE_MAX_RANGE_M if a pole sits farther than this.
LIDAR_MIN_RANGE_M = 0.12
LIDAR_POLE_MAX_RANGE_M = 0.45
LIDAR_ANGLE_OFFSET_DEG = 90.0   # forward->top (was 0; foot-ahead showed at right)
LIDAR_CCW = True
LIDAR_BLANK_SECTORS_DEG: list = []

CMD_TIMEOUT_S = 0.5         # no command within this -> force STOP
TLM_HZ        = 50.0
FRAME_HZ      = 12.0
LIDAR_HZ      = 8.0         # publish cadence; the D500 reader fills faster
# Camera capture: landscape, hardware MJPEG. ArUco/gesture detection runs on
# the Mac; this resolution trades detection range against WiFi bandwidth.
CAM_CAP_W, CAM_CAP_H = 640, 480
CAM_HZ        = 10.0
# Hardware-MJPEG target bitrate. At 640x480 (0.3 MP) 3 Mbps gives ~30 KB/frame
# with plenty of bits per pixel, so the image is crisp (no blocky DCT mosaic)
# AND small enough to survive a flaky AP. Bump resolution + bitrate together on
# a strong link if you want more detail / longer ArUco range.
CAM_BITRATE   = 3_000_000
# Manual white balance (red_gain, blue_gain) or None for auto AWB. Auto AWB
# can't correct a green/magenta TINT — run pi/awb_cal.py and paste gains here.
CAM_COLOUR_GAINS = None
CAM_STALL_S   = 4.0          # no camera frame produced within this -> restart it
CAM_ROTATE_180 = True        # module is mounted upside-down (180° mount); flip
                             # at the source so EVERY consumer gets upright frames


# ── ESP32 serial bridge ──────────────────────────────────────────────────────
class ESP32Link:
    """Serial link to the ESP32. Holds the latest drive char and re-sends it at
    20 Hz (under the firmware's 200 ms key timeout); parses the "O,..." odometry
    and "P:..." power lines it sends back.

    Stub when serial_dev is None: drive chars are printed, odometry is a
    stationary fix (a stub ESP32 = a robot that isn't moving)."""

    # 25 Hz. Forward (keyState) and turn (turnState) are INDEPENDENT firmware
    # axes, each with a 150 ms timeout; round-robining the active chars keeps a
    # diagonal (e.g. w+d) alive — each axis refreshed every ~80 ms.
    SEND_INTERVAL_S = 0.04

    def __init__(self, serial_dev: str | None, baud: int = 115200):
        self._dev = serial_dev
        self._ser = None
        self._forward = None                # 'w' / 's' / None  (forward axis)
        self._turn = None                   # 'a' / 'd' / None  (turn axis)
        self._last_update = time.monotonic()
        self._last_sent = None              # last char written (drives `moving`)
        self._last_axes = (None, None)      # for change-only logging
        self._rr = 0                        # round-robin cursor across axes
        self._lock = threading.Lock()
        self._running = True
        self._rx = bytearray()
        self._got_tlm = False               # have we EVER heard from the ESP32?
        self._have_O = False                # have we seen a real O, odom line?
        # Latest parsed telemetry (None until the ESP32 reports).
        self._odom = None                   # (t_ms, v, yaw, moving)
        self._power = None                  # (voltage, current)
        self._tilt = None                   # balance tilt angle (rad, from `th=`)
        if serial_dev:
            import serial
            self._ser = serial.Serial(serial_dev, baud, timeout=0)
            print(f"[esp32] opened {serial_dev} @ {baud} — waiting for telemetry…",
                  flush=True)
            threading.Thread(target=self._writer, daemon=True).start()
            threading.Thread(target=self._reader, daemon=True).start()
        else:
            print("[esp32] STUB (no --serial): drive is a no-op, robot won't move",
                  flush=True)

    def set_axes(self, forward=None, turn=None) -> None:
        """Two-axis drive (4 cardinals + 4 diagonals + stop). Used by autonomy."""
        if forward not in ("w", "s", None) or turn not in ("a", "d", None):
            return
        with self._lock:
            self._forward = forward
            self._turn = turn
            self._last_update = time.monotonic()

    def set_direction(self, direction) -> None:
        """Legacy single-axis command (manual joystick): map onto one axis."""
        if direction == "-":
            direction = None
        if direction in ("w", "s"):
            self.set_axes(direction, None)
        elif direction in ("a", "d"):
            self.set_axes(None, direction)
        elif direction is None:
            self.set_axes(None, None)

    def stop(self) -> None:
        self.set_axes(None, None)

    def calibrate(self) -> None:
        """Send the yaw-recalibrate char ('z') to the ESP. One-shot; the
        firmware runs a 1 s averaging window and persists the bias to flash."""
        if self._ser is None:
            print("[esp32] STUB: calibrate ignored", flush=True)
            return
        with self._lock:
            with contextlib.suppress(Exception):
                self._ser.write(b"z")
        print("[esp32] -> calibrate (z)", flush=True)

    def _writer(self):
        while self._running:
            with self._lock:
                fwd, trn = self._forward, self._turn
                stale = (time.monotonic() - self._last_update) > CMD_TIMEOUT_S
            if stale:
                fwd = trn = None
            active = [c for c in (fwd, trn) if c is not None]
            if active:
                char = active[self._rr % len(active)]
                self._rr += 1
            else:
                char = P.STOP_CHAR  # clears BOTH firmware axes at once
            if (fwd, trn) != self._last_axes:    # change-only: see drive cmds arrive
                print(f"[esp32] -> fwd={fwd!r} turn={trn!r}", flush=True)
                self._last_axes = (fwd, trn)
            self._last_sent = char  # 'w'/'s'/'a'/'d' => moving; '-' => idle
            with contextlib.suppress(Exception):
                self._ser.write(char.encode())
            time.sleep(self.SEND_INTERVAL_S)

    def _reader(self):
        while self._running:
            try:
                data = self._ser.read(256)
            except Exception:
                time.sleep(0.2)
                continue
            if not data:
                time.sleep(0.01)
                continue
            self._rx.extend(data)
            while b"\n" in self._rx:
                line, _, rest = self._rx.partition(b"\n")
                self._rx = bytearray(rest)
                self._handle_line(line.decode("ascii", "ignore").strip())

    def _handle_line(self, line: str):
        if not self._got_tlm and (line.startswith("O,") or line.startswith("P")):
            self._got_tlm = True
            print(f"[esp32] telemetry OK (ESP32 is on this port): {line!r}", flush=True)
        if line.startswith("O,"):
            parts = line.split(",")
            try:                            # O,<t_ms>,<v_mps>,<yaw_rad>
                t_ms = float(parts[1]); v = float(parts[2]); yaw = float(parts[3])
            except (IndexError, ValueError):
                return
            moving = self._last_sent in P.DRIVE_KEYS
            self._odom = (t_ms, v, yaw, moving)
            self._have_O = True             # real odom (incl. yaw) is present
        elif line.startswith("P:") or line.startswith("P,"):
            try:                            # P:<voltage>,<current>
                v, i = line[2:].split(",")
                self._power = (float(v), float(i))
            except ValueError:
                return
        else:
            # The debug line ("t=148326 th=0.01 vel=0.0003 u=... ...") is a set
            # of space-separated key=value tokens that vary between builds
            # (th_d7, td_c, ...). Pull what we recognise, ignoring the rest so a
            # stray line can't crash us.
            kv = {}
            for tok in line.split():
                k, sep, val = tok.partition("=")
                if sep:
                    kv[k] = val
            # `th` is the balance tilt angle (rad). Current firmware emits this
            # debug line ALONGSIDE the real O, line, so capture tilt regardless
            # of _have_O — it's the only place the tilt is reported.
            try:
                self._tilt = float(kv["th"])
            except (KeyError, ValueError):
                pass
            # Odom fallback ONLY for OLD firmware with no O, line: dead-reckon
            # forward from t/vel (yaw unknown, 0.0). Once an O, has been seen
            # that path owns odom and we must NOT clobber its yaw with 0.0.
            if self._have_O:
                return
            try:
                t_ms = float(kv["t"]); vel = float(kv["vel"])
            except (KeyError, ValueError):
                return
            self._odom = (t_ms, vel, 0.0, abs(vel) > 0.01)   # m/s deadband
            if not self._got_tlm:
                self._got_tlm = True
                print(f"[esp32] telemetry OK (debug-format, no yaw): {line!r}",
                      flush=True)

    def telemetry(self, t: float) -> dict:
        if self._ser is None:               # stub: stationary robot
            return {"t": t, "odom": {"t_ms": t * 1000.0, "v": 0.0, "yaw": 0.0,
                                     "moving": False}, "power": None}
        odom = self._odom
        power = self._power
        out = {"t": t}
        if odom is not None:
            out["odom"] = {"t_ms": odom[0], "v": odom[1], "yaw": odom[2],
                           "moving": odom[3]}
        if power is not None:
            out["power"] = {"voltage": power[0], "current": power[1]}
        if self._tilt is not None:
            out["tilt"] = self._tilt          # balance tilt angle (rad)
        return out

    def close(self):
        self._running = False
        if self._ser is not None:
            with contextlib.suppress(Exception):
                self._ser.write(b"-")
                self._ser.close()


# ── Camera (hardware MJPEG) ──────────────────────────────────────────────────
class _LatestJpeg(io.BufferedIOBase):
    """FileOutput sink keeping only the most recent hardware-MJPEG frame.
    Subclasses io.BufferedIOBase because picamera2's FileOutput requires it.
    Stamps each write so the camera watchdog can detect a stalled encoder."""

    def __init__(self):
        super().__init__()
        self._frame = None
        self._lock = threading.Lock()
        self.last_write = 0.0

    def writable(self) -> bool:
        return True

    def write(self, buf) -> int:
        b = bytes(buf)
        with self._lock:
            self._frame = b
            self.last_write = time.monotonic()
        return len(b)

    def take(self):
        with self._lock:
            f, self._frame = self._frame, None
        return f


class CameraLink:
    """Pi Camera (IMX219) via the HARDWARE MJPEG encoder (off the CPU and off
    this proxy's hot loop). Self-healing: a background supervisor (re)starts the
    camera with retries if it isn't ready at launch (the previous proxy instance
    may not have released it yet on restart, or pipewire grabbed it), AND
    restarts the encoder if it stalls mid-run (no frame for CAM_STALL_S). A
    camera failure never crashes the proxy — LiDAR + ESP32 keep running."""

    def __init__(self, enabled: bool):
        self._cam = None
        self._sink = None
        self._want = enabled
        if enabled:
            threading.Thread(target=self._supervise, daemon=True).start()

    def _start(self) -> bool:
        cam = None
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import MJPEGEncoder
            from picamera2.outputs import FileOutput
            # Fail fast + quiet if libcamera sees no camera at all (CSI ribbon
            # unseated / module dead). Constructing Picamera2() with zero
            # cameras throws "list index out of range" every retry; instead say
            # it once and stop the supervisor so the log isn't flooded.
            if not Picamera2.global_camera_info():
                print("[camera] no camera detected (global_camera_info=[]); "
                      "disabling — check the CSI ribbon cable", flush=True)
                self._want = False
                return False
            cam = Picamera2()
            # Full-FOV fix ("zoom out"): at a small main size the IMX219 would
            # otherwise read out a cropped CENTRE window of the sensor (a
            # zoomed-in look). Pick the sensor mode that covers the most sensor
            # area (full field of view) for the raw stream and let the main
            # stream downscale from it, so the feed shows the whole scene.
            raw_size = None
            with contextlib.suppress(Exception):
                modes = cam.sensor_modes
                def _area(m):
                    cl = m.get("crop_limits")
                    return (cl[2] * cl[3]) if cl else 0
                mx = max((_area(m) for m in modes), default=0)
                if mx > 0:
                    full = [m for m in modes if _area(m) >= 0.95 * mx]
                    chosen = min(full, key=lambda m: m["size"][0] * m["size"][1])
                    raw_size = tuple(chosen["size"])
            ctrls = {"FrameRate": CAM_HZ}
            if CAM_COLOUR_GAINS is not None:
                ctrls["AwbEnable"] = False
                ctrls["ColourGains"] = tuple(CAM_COLOUR_GAINS)
            cfg_kw = {"main": {"size": (CAM_CAP_W, CAM_CAP_H)}, "controls": ctrls}
            if CAM_ROTATE_180:
                from libcamera import Transform
                cfg_kw["transform"] = Transform(hflip=1, vflip=1)   # 180° mount
            if raw_size is not None:
                cfg_kw["raw"] = {"size": raw_size}
            cfg = cam.create_video_configuration(**cfg_kw)
            cam.configure(cfg)
            # Reset the scaler crop to the whole sensor area (widest FOV).
            with contextlib.suppress(Exception):
                fov = cam.camera_properties.get("ScalerCropMaximum")
                if fov and fov[2] > 0 and fov[3] > 0:
                    cam.set_controls({"ScalerCrop": fov})
            sink = _LatestJpeg()
            sink.last_write = time.monotonic()          # grace before watchdog
            cam.start_recording(MJPEGEncoder(bitrate=CAM_BITRATE),
                                FileOutput(sink))
            self._cam, self._sink = cam, sink
            print("[camera] started", flush=True)
            return True
        except Exception as e:                          # noqa: BLE001
            print(f"[camera] start failed ({e}); retrying in 2s", flush=True)
            with contextlib.suppress(Exception):
                if cam is not None:
                    cam.close()
            return False

    def _stop(self):
        cam, self._cam, self._sink = self._cam, None, None
        if cam is not None:
            with contextlib.suppress(Exception):
                cam.stop_recording()
            with contextlib.suppress(Exception):
                cam.close()

    def _supervise(self):
        while self._want:
            if self._cam is None:
                if not self._start():
                    time.sleep(2.0)
                    continue
            time.sleep(1.0)
            # Watchdog: encoder produced no frame for CAM_STALL_S -> restart it.
            if (self._sink is not None
                    and time.monotonic() - self._sink.last_write > CAM_STALL_S):
                print(f"[camera] stall >{CAM_STALL_S:.0f}s -> restarting",
                      flush=True)
                self._stop()

    def frame_message(self, seq: int, t: float):
        sink = self._sink
        if sink is None:
            return None
        jpg = sink.take()
        if jpg is None:
            return None
        return P.encode_frame_jpeg(jpg, CAM_CAP_W, CAM_CAP_H, seq, t)

    def close(self) -> None:
        self._want = False
        self._stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar", action="store_true",
                    help="real D500 LiDAR (else stub box)")
    ap.add_argument("--lidar-port", default="/dev/ttyUSB1",
                    help="D500 serial device (default /dev/ttyUSB1)")
    ap.add_argument("--lidar-baud", type=int, default=230400)
    ap.add_argument("--camera", action="store_true",
                    help="real Pi camera via picamera2 hardware MJPEG")
    ap.add_argument("--serial", default=None,
                    help="ESP32 tty, e.g. /dev/ttyUSB0; omit to stub the ESP32")
    args = ap.parse_args()

    esp32 = ESP32Link(args.serial)
    lidar = LidarD500(args.lidar_port if args.lidar else None,
                      baud=args.lidar_baud,
                      angle_offset_deg=LIDAR_ANGLE_OFFSET_DEG, ccw=LIDAR_CCW,
                      blank_sectors_deg=LIDAR_BLANK_SECTORS_DEG,
                      min_range_m=LIDAR_MIN_RANGE_M,
                      auto_mask_poles=True,
                      pole_max_range_m=LIDAR_POLE_MAX_RANGE_M)
    camera = CameraLink(enabled=args.camera)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{P.SENSOR_PORT}")
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://*:{P.CMD_PORT}")
    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    print(f"[proxy] lidar={'REAL' if args.lidar else 'stub'} "
          f"camera={'REAL' if args.camera else 'stub'} "
          f"esp32={args.serial or 'stub'} | "
          f"PUB :{P.SENSOR_PORT}  PULL :{P.CMD_PORT}", flush=True)

    t0 = time.monotonic()
    last_tlm = last_frame = last_lidar = 0.0
    last_cmd_t = t0
    seq = 0
    try:
        while True:
            now = time.monotonic()
            t = now - t0

            # 1. Drain all pending commands; act on the latest.
            for sock, _ in poller.poll(timeout=0):
                while True:
                    try:
                        cmd = P.decode_cmd(sock.recv(flags=zmq.NOBLOCK))
                    except zmq.Again:
                        break
                    last_cmd_t = now
                    if cmd.get("type") == "calibrate":
                        esp32.calibrate()
                    elif cmd.get("type") == "drive":
                        if "forward" in cmd or "turn" in cmd:
                            esp32.set_axes(cmd.get("forward"), cmd.get("turn"))
                        else:
                            esp32.set_direction(cmd.get("direction"))

            # 2. Command link gone silent -> force STOP.
            if now - last_cmd_t > CMD_TIMEOUT_S:
                esp32.stop()

            # 3. Telemetry (odometry + power).
            if now - last_tlm >= 1.0 / TLM_HZ:
                last_tlm = now
                pub.send_multipart(P.encode_tlm(esp32.telemetry(t)))

            # 4. Camera frame (None = nothing fresh; skip).
            if now - last_frame >= 1.0 / FRAME_HZ:
                last_frame = now
                msg = camera.frame_message(seq, t)
                if msg is not None:
                    pub.send_multipart(msg)
                    seq += 1

            # 5. LiDAR scan.
            if now - last_lidar >= 1.0 / LIDAR_HZ:
                last_lidar = now
                scan = lidar.read_scan()
                if scan is not None:
                    ranges, amin, ainc = scan
                    pub.send_multipart(P.encode_lidar(ranges, amin, ainc, t))

            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n[proxy] shutting down; STOP sent", flush=True)
    finally:
        esp32.stop()
        esp32.close()
        lidar.close()
        camera.close()
        pub.close(0)
        pull.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
