"""LDROBOT D500 LiDAR Kit driver.

The D500 is part of LDROBOT's LD-series (same wire protocol as the LD06/LD19):
a continuous stream of fixed 47-byte packets over UART, each carrying 12
measurement points. The kit ships with a USB-UART adapter, so on the Pi it
shows up as /dev/ttyUSB* at 230400 baud and needs no motor-control line (it
spins whenever powered).

This replaces the old RPLIDAR A1M8 path (adafruit_rplidar, /dev/serial0) used
in the now-removed waiter-hardware tree — the A1M8's protocol is completely
different, so that driver can't read a D500.

Packet layout (little-endian):
    [0]      0x54                header
    [1]      0x2C                ver/len: 0x2C => 12 points this packet
    [2:4]    speed               rotation speed, deg/s
    [4:6]    start_angle         0.01 deg
    [6:42]   12 x (dist u16 mm, intensity u8)
    [42:44]  end_angle           0.01 deg
    [44:46]  timestamp           ms (wraps at 30000)
    [46]     crc8                over bytes [0:46]

Per-point angle is linearly interpolated between start_angle and end_angle.
Points are binned into a fixed `rays`-wide uniform-angle array (inf = no
return), matching the (ranges, angle_min, angle_increment) shape the mapper
consumes — the same contract the old proxy.LidarLink used.

Usage:
    lidar = LidarD500("/dev/ttyUSB0")        # real
    lidar = LidarD500(None)                  # stub (static box, no hardware)
    scan = lidar.read_scan()                 # (ranges, amin, ainc) or None
"""

from __future__ import annotations

import contextlib
import math
import threading
import time

try:
    import numpy as np
except ImportError:  # mapper needs numpy anyway; fail clearly if it's missing
    np = None

HEADER = 0x54
VER_LEN = 0x2C            # 0x2C => 12 measurement points per packet
POINTS_PER_PACK = 12
PACKET_LEN = 47          # header..crc inclusive
DEFAULT_BAUD = 230400    # LD06/LD19/D500 family default

# CRC8 for the LD-series, poly 0x4D, MSB-first, init 0x00, over packet[0:46].
# Generate the table from the polynomial rather than hardcoding it — a
# transcription error in a pasted table silently fails ~94% of packets
# (verified against a real D500: a wrong table passed 37/692, this passes all).
_CRC_POLY = 0x4D


def _make_crc_table(poly: int = _CRC_POLY) -> tuple:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
        table.append(c)
    return tuple(table)


_CRC_TABLE = _make_crc_table()


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


class LidarD500:
    """Background-threaded reader for an LDROBOT D500 over USB serial.

    Self-healing like the old A1M8 path: a thrown error (unplug, garbage)
    or a silent stall (watchdog) both trigger a clean reconnect, so the
    dashboard never has to be restarted because the LiDAR hiccuped.

    Stub mode (``port=None``) returns a static 1 m box so the map/mapper
    pipeline can be exercised with no hardware attached.
    """

    STALL_S = 3.0            # no fresh revolution within this -> reconnect

    def __init__(self, port: str | None, baud: int = DEFAULT_BAUD,
                 rays: int = 450, angle_offset_deg: float = 0.0,
                 ccw: bool = True, blank_sectors_deg=None,
                 min_range_m: float = 0.0):
        if np is None:
            raise RuntimeError("numpy is required for the LiDAR driver")
        self._port = port
        self._baud = baud
        self._rays = rays
        # amin = 0, NOT -pi: _consume_packet bins raw angle 0deg into bin 0, so
        # the angle axis the mapper reconstructs (amin + i*ainc) must start at 0
        # too. With -pi every scan came out rotated 180deg (walls in front of the
        # LiDAR drawn behind the robot). angle_offset_deg still calibrates the
        # physical mounting on top of this.
        self._amin = 0.0
        self._ainc = 2 * math.pi / rays
        # angle_offset_deg + ccw line ray 0 up with the robot's forward axis
        # and match rotation sense. Calibrate once with the LiDAR spinning
        # (point a known object straight ahead, adjust until it lands at the
        # top of the map). Same role as proxy.LidarLink's offset/ccw.
        self._offset = angle_offset_deg
        self._ccw = ccw
        # Mask out fixed self-obstructions (the three mast poles standing in
        # the scan plane): any ray in a blanked sector, or closer than
        # min_range_m, is dropped to inf so it never pollutes the map. The
        # poles rotate with the robot, so unmasked they'd smear into arcs in
        # the world frame. Discover the sector angles with
        # `lidar_test.py --poles`. blank_sectors_deg = [(centre_deg, half_width_deg), ...]
        self._min_range = float(min_range_m)
        self._blank = np.zeros(rays, dtype=bool)
        for centre, half in (blank_sectors_deg or []):
            ray_deg = (np.arange(rays) + 0.5) / rays * 360.0
            d = np.abs((ray_deg - centre + 180.0) % 360.0 - 180.0)
            self._blank |= d <= half
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        self._ser = None
        self._last_data_t = 0.0
        if port is not None:
            threading.Thread(target=self._reader, daemon=True).start()
            threading.Thread(target=self._watchdog, daemon=True).start()

    # ── public API (matches proxy.LidarLink) ─────────────────────────────────

    def read_scan(self):
        """Latest finished revolution as (ranges, angle_min, angle_increment),
        or None if nothing new since the last call (drop-oldest)."""
        if self._port is None:
            return self._stub_scan()
        with self._lock:
            ranges, self._latest = self._latest, None
        if ranges is None:
            return None
        # Drop the masked mast-pole sectors and any too-close returns.
        if self._blank.any():
            ranges[self._blank] = np.inf
        if self._min_range > 0.0:
            ranges[ranges < self._min_range] = np.inf
        return ranges, self._amin, self._ainc

    def close(self):
        self._stop = True
        with contextlib.suppress(Exception):
            if self._ser is not None:
                self._ser.close()

    # ── reader thread ────────────────────────────────────────────────────────

    def _reader(self):
        import serial  # pyserial
        while not self._stop:
            self._last_data_t = time.monotonic()
            # Reset the parse buffer + partial-revolution accumulator on every
            # (re)connect, so a reconnect never blends the previous session's
            # leftover points into the first new revolution.
            buf = bytearray()
            acc = np.full(self._rays, np.inf, dtype=np.float32)
            last_start_deg = None
            try:
                self._ser = serial.Serial(self._port, self._baud, timeout=1.0)
                self._ser.reset_input_buffer()
                self._last_data_t = time.monotonic()
                while not self._stop:
                    chunk = self._ser.read(256)
                    if chunk:
                        buf.extend(chunk)
                    # Parse all complete packets currently in the buffer.
                    while len(buf) >= PACKET_LEN:
                        if buf[0] != HEADER or buf[1] != VER_LEN:
                            del buf[0]                 # resync to next header
                            continue
                        packet = bytes(buf[:PACKET_LEN])
                        if _crc8(packet[:PACKET_LEN - 1]) != packet[PACKET_LEN - 1]:
                            del buf[0]                 # bad packet, slide on
                            continue
                        del buf[:PACKET_LEN]
                        start_deg, finished = self._consume_packet(packet, acc,
                                                                   last_start_deg)
                        if finished is not None:        # a revolution completed
                            with self._lock:
                                self._latest = finished
                            acc = np.full(self._rays, np.inf, dtype=np.float32)
                            self._last_data_t = time.monotonic()
                        last_start_deg = start_deg
            except Exception as e:                      # keep the dashboard alive
                if not self._stop:                      # quiet on intentional close
                    print(f"[lidar] reader error: {e}; reconnecting", flush=True)
            finally:
                with contextlib.suppress(Exception):
                    if self._ser is not None:
                        self._ser.close()
            if not self._stop:
                time.sleep(1.0)

    def _consume_packet(self, packet: bytes, acc, last_start_deg):
        """Bin one packet's 12 points into `acc`. Returns
        (this packet's start angle in deg, finished_ranges_or_None).
        finished is non-None when this packet crossed the 360->0 wrap,
        i.e. a full revolution just completed."""
        start = int.from_bytes(packet[4:6], "little") * 0.01
        end = int.from_bytes(packet[42:44], "little") * 0.01
        span = (end - start) % 360.0
        step = span / (POINTS_PER_PACK - 1) if POINTS_PER_PACK > 1 else 0.0

        finished = None
        # Wrap detection: a new revolution starts when this packet's start
        # angle is "behind" the previous one (crossed 360 -> 0).
        if last_start_deg is not None and start < last_start_deg - 1.0:
            finished = acc.copy()

        for i in range(POINTS_PER_PACK):
            off = 6 + i * 3
            dist_mm = int.from_bytes(packet[off:off + 2], "little")
            if dist_mm <= 0:
                continue
            ang = (start + step * i) % 360.0
            a = (ang + self._offset) if self._ccw else (-ang + self._offset)
            b = int(((a % 360.0) / 360.0) * self._rays) % self._rays
            r = dist_mm / 1000.0
            if r < acc[b]:
                acc[b] = r
        return start, finished

    def _watchdog(self):
        """Force a reconnect if no fresh revolution arrives within STALL_S.
        Closing the port unblocks the reader's blocked read so its retry
        loop re-runs — rescues the silent-stall (desynced stream) case."""
        while not self._stop:
            time.sleep(1.0)
            if (self._ser is not None
                    and time.monotonic() - self._last_data_t > self.STALL_S):
                print(f"[lidar] stall >{self.STALL_S:.0f}s, forcing reconnect",
                      flush=True)
                self._last_data_t = time.monotonic()
                with contextlib.suppress(Exception):
                    self._ser.close()

    # ── stub ──────────────────────────────────────────────────────────────────

    def _stub_scan(self):
        """Static ~1 m box, so the mapper has geometry without hardware."""
        ranges = np.full(self._rays, np.inf, dtype=np.float32)
        for i in range(self._rays):
            a = self._amin + i * self._ainc
            ca, sa = math.cos(a), math.sin(a)
            best = np.inf
            for wall in (1.0, -1.0):
                if abs(ca) > 1e-6:
                    d = wall / ca
                    if 0 < d < best and abs(d * sa) <= 1.0:
                        best = d
                if abs(sa) > 1e-6:
                    d = wall / sa
                    if 0 < d < best and abs(d * ca) <= 1.0:
                        best = d
            ranges[i] = best
        return ranges, self._amin, self._ainc
