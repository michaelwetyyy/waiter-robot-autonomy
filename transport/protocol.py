"""Wire protocol shared by the waiter-robot Pi proxy and the Mac server.

The Pi became a thin sensor/actuator bridge: it does NO mapping, vision, or
planning. All of that runs on the Mac. The two sides talk over ZeroMQ:

  Sensor channel   Pi PUB  -> Mac SUB   (port SENSOR_PORT)
      Pi -> Mac, fan-out, drop-oldest. Multipart, tagged by a topic frame:
        [TOPIC_TLM,   json]                  odometry + power + tilt @ ~50 Hz
        [TOPIC_FRAME, header_json, jpeg]     camera @ ~10 fps (JPEG bytes)
        [TOPIC_LIDAR, header_json, ranges]   D500 scan @ ~8 Hz (float32)

  Command channel  Mac PUSH -> Pi PULL  (port CMD_PORT)
      Mac -> Pi, single consumer. Single-part JSON, one of:
        {"type": "drive", "direction": "w"|"s"|"a"|"d"|"-"|null}   single-axis
        {"type": "drive", "forward": "w"|"s"|null,                 two-axis arc
                          "turn": "a"|"d"|null}
        {"type": "calibrate"}                            recal+persist ESP yaw

Telemetry / commands are JSON (stdlib). Camera frames travel as already-
encoded JPEG bytes (the Pi's hardware MJPEG encoder produces them, off-CPU),
wrapped with a tiny JSON header. LiDAR ranges travel as raw float32 (inf/nan
preserved for no-return rays), matching the OccupancyMap.integrate() scan
tuple (ranges, angle_min, angle_increment) the Mac consumes 1:1.

Stdlib + numpy on both sides; cv2 is imported lazily (Mac-only decode path)
so the Pi proxy never needs it.
"""

from __future__ import annotations
import json
from typing import Any

import numpy as np

# Ports (Pi binds, Mac connects)
SENSOR_PORT = 5555
CMD_PORT    = 5556

# Topics on the sensor channel
TOPIC_TLM   = b"tlm"
TOPIC_FRAME = b"frame"
TOPIC_LIDAR = b"lidar"

# Drive command alphabet — single source of truth for both sides.
DRIVE_KEYS = frozenset({"w", "s", "a", "d"})
STOP_CHAR  = "-"


def normalize_direction(direction):
    """Canonicalise a raw drive token: a valid key, or None for stop.
    Both STOP_CHAR ('-') and any unrecognised token map to None."""
    if direction == STOP_CHAR:
        return None
    return direction if direction in DRIVE_KEYS else None


# ── Telemetry (odometry + power) ─────────────────────────────────────────────
def encode_tlm(tlm: dict[str, Any]) -> list[bytes]:
    return [TOPIC_TLM, json.dumps(tlm).encode("utf-8")]


def decode_tlm(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


# ── Camera frame ─────────────────────────────────────────────────────────────
def encode_frame_jpeg(jpeg_bytes: bytes, w: int, h: int, seq: int,
                      t: float) -> list[bytes]:
    """Wrap ALREADY-encoded JPEG bytes (from the Pi's hardware MJPEG encoder)
    with a header. Keeps the heavy encode off the proxy's hot loop."""
    header = json.dumps({"w": int(w), "h": int(h), "fmt": "jpeg",
                         "seq": seq, "t": t}).encode("utf-8")
    return [TOPIC_FRAME, header, jpeg_bytes]


def decode_frame(header_bytes: bytes, pixel_bytes: bytes):
    """Reconstruct (HxWx3 uint8 RGB array, header dict) from a frame message.

    cv2 is imported here (not at module top) so the Pi proxy — which only ever
    encodes — doesn't need OpenCV imported on its hot path."""
    import cv2
    header = json.loads(header_bytes.decode("utf-8"))
    buf = np.frombuffer(pixel_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)      # JPEG -> BGR
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, header


# ── LiDAR scan ───────────────────────────────────────────────────────────────
def encode_lidar(ranges: np.ndarray, angle_min: float, angle_increment: float,
                 t: float) -> list[bytes]:
    """`ranges` is sent as float32 (NaN/inf preserved for no-return rays)."""
    r = np.ascontiguousarray(ranges, dtype=np.float32)
    header = json.dumps({"n": int(r.size), "angle_min": angle_min,
                         "angle_increment": angle_increment, "t": t}).encode("utf-8")
    return [TOPIC_LIDAR, header, r.tobytes()]


def decode_lidar(header_bytes: bytes, ranges_bytes: bytes):
    """Return (ranges float32 array, angle_min, angle_increment, t)."""
    h = json.loads(header_bytes.decode("utf-8"))
    ranges = np.frombuffer(ranges_bytes, dtype=np.float32)
    return ranges, h["angle_min"], h["angle_increment"], h["t"]


# ── Commands ─────────────────────────────────────────────────────────────────
def encode_cmd(cmd: dict[str, Any]) -> bytes:
    return json.dumps(cmd).encode("utf-8")


def decode_cmd(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def cmd_drive(direction) -> dict[str, Any]:
    """direction is a DRIVE_KEYS member, STOP_CHAR, or None (stop)."""
    return {"type": "drive", "direction": direction}


def cmd_calibrate() -> dict[str, Any]:
    """Ask the ESP32 to recalibrate + persist its yaw-gyro zero-rate bias."""
    return {"type": "calibrate"}


def cmd_axes(forward, turn) -> dict[str, Any]:
    """Two-axis drive: forward in {'w','s',None}, turn in {'a','d',None}.

    The ESP firmware runs forward and turn as INDEPENDENT axes, so sending
    both lets the proxy round-robin them onto the serial wire and keep both
    alive at once — i.e. arcs / diagonals (4 cardinals + 4 diagonals + stop),
    not the turn-OR-straight bang-bang a single drive char forces. The proxy's
    command receiver already routes a dict carrying 'forward'/'turn' to
    ESP32Link.set_axes."""
    return {"type": "drive", "forward": forward, "turn": turn}
