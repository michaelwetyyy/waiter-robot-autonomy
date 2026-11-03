"""RemoteSensorSource: lives on the Mac, talks to the Pi proxy over ZeroMQ.

Hides the network entirely. A background thread drains the Pi's PUB channel
into latest-value caches; drive commands go out over a PUSH socket.

  reads  (get_*)   <- SUB socket draining the Pi's sensor PUB channel
  writes (set_*)   -> PUSH socket feeding the Pi's command PULL channel

Drop-oldest semantics throughout: only the freshest frame / scan / telemetry
is ever kept, and the camera JPEG is decoded only ONCE per drain (the newest),
never the whole backlog — decoding a stale backlog would collapse the frame
rate into a death spiral after any link blip.
"""

from __future__ import annotations
import threading
import time
from typing import Optional

import zmq

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from transport import protocol as P          # noqa: E402


class RemoteSensorSource:
    # A receive stall longer than this means the Pi proxy went away (reboot,
    # power blip, or service restart). ZMQ auto-reconnects the TCP, but a
    # long-lived SUB that rides through several PUB rebinds can silently stop
    # redelivering a topic — observed: small topics (LiDAR/tlm) resumed but the
    # large multi-part camera FRAME never did. We force a clean reconnect by
    # rebuilding the sockets; see _rebuild_sockets.
    _STALL_REBUILD_S = 5.0

    def __init__(self, pi_host: str, recv_timeout_s: float = 2.0,
                 frame_rot_k: int = 0):
        self._pi_host = pi_host
        self._recv_timeout_s = recv_timeout_s
        self._frame_rot_k = frame_rot_k % 4
        self._t0 = time.monotonic()

        self._lock = threading.Lock()
        self._sub, self._push = self._open_sockets()
        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)

        self._frame = None
        self._frame_is_new = False
        self._tlm: Optional[dict] = None
        self._lidar = None               # (ranges, amin, ainc)
        self._lidar_is_new = False
        self._last_rx = time.monotonic()
        self._last_rebuild = time.monotonic()
        self._stop = False
        threading.Thread(target=self._drain_loop, daemon=True).start()

    def _open_sockets(self):
        """Create a fresh SUB (sensor) + PUSH (command) socket pair connected
        to the Pi. Used at startup and by _rebuild_sockets for self-healing."""
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{self._pi_host}:{P.SENSOR_PORT}")
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        push = ctx.socket(zmq.PUSH)
        push.connect(f"tcp://{self._pi_host}:{P.CMD_PORT}")
        return sub, push

    def _rebuild_sockets(self):
        """Tear down and reconnect after a sustained stall. Cheap, and the only
        reliable recovery from a wedged subscription once the Pi comes back."""
        print("[remote] receive stall -> rebuilding ZMQ sockets "
              "(Pi proxy likely restarted)", flush=True)
        self._poller.unregister(self._sub)
        self._sub.close(0)
        new_sub, new_push = self._open_sockets()
        self._sub = new_sub                      # _sub is owned by this thread
        self._poller.register(self._sub, zmq.POLLIN)
        with self._lock:                          # _push is shared with senders
            old_push, self._push = self._push, new_push
        old_push.close(0)

    # -- background drain --------------------------------------------------
    def _drain_loop(self):
        import numpy as np
        while not self._stop:
            socks = dict(self._poller.poll(timeout=200))
            now = time.monotonic()
            # Self-heal: a sustained stall means the Pi proxy restarted. Rebuild
            # the sockets (and keep rebuilding every _STALL_REBUILD_S while the
            # Pi is gone) so we reconnect cleanly the moment it returns, instead
            # of riding a wedged subscription that silently drops the camera.
            if (now - self._last_rx > self._STALL_REBUILD_S and
                    now - self._last_rebuild > self._STALL_REBUILD_S):
                self._rebuild_sockets()
                self._last_rebuild = time.monotonic()
                continue
            if self._sub not in socks:
                continue
            latest_frame_parts = None
            while True:
                try:
                    parts = self._sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                topic = parts[0]
                if topic == P.TOPIC_TLM:
                    with self._lock:
                        self._tlm = P.decode_tlm(parts[1])
                elif topic == P.TOPIC_FRAME:
                    latest_frame_parts = (parts[1], parts[2])   # decode last only
                elif topic == P.TOPIC_LIDAR:
                    ranges, amin, ainc, _ = P.decode_lidar(parts[1], parts[2])
                    with self._lock:
                        self._lidar = (ranges, amin, ainc)
                        self._lidar_is_new = True
                self._last_rx = time.monotonic()
            if latest_frame_parts is not None:
                frame, _ = P.decode_frame(*latest_frame_parts)
                if self._frame_rot_k:
                    frame = np.ascontiguousarray(np.rot90(frame, self._frame_rot_k))
                with self._lock:
                    self._frame = frame
                    self._frame_is_new = True
        self._sub.close(0)              # closed by the thread that polls it

    # -- health ------------------------------------------------------------
    def link_ok(self) -> bool:
        return (time.monotonic() - self._last_rx) < self._recv_timeout_s

    def time(self) -> float:
        return time.monotonic() - self._t0

    # -- reads (drop-oldest) ----------------------------------------------
    def get_camera_frame(self):
        with self._lock:
            if self._frame_is_new:
                self._frame_is_new = False
                return self._frame
            return None

    def get_lidar(self):
        with self._lock:
            if self._lidar_is_new:
                self._lidar_is_new = False
                return self._lidar
            return None

    def get_odometry(self):
        """Latest (t_ms, v, yaw, moving) or None."""
        with self._lock:
            o = (self._tlm or {}).get("odom")
        if not o:
            return None
        return o["t_ms"], o["v"], o["yaw"], o["moving"]

    def get_power(self):
        """Latest {'voltage','current'} or None."""
        with self._lock:
            return (self._tlm or {}).get("power")

    def get_tilt(self):
        """Latest balance tilt angle (rad) from the ESP, or None."""
        with self._lock:
            return (self._tlm or {}).get("tilt")

    # -- writes ------------------------------------------------------------
    def _push_cmd(self, cmd: dict) -> None:
        # Grab the current push socket under the lock (it may be swapped out by
        # _rebuild_sockets), then send NOBLOCK: a command dropped during a
        # rebuild is harmless (re-sent at the loop rate; the Pi watchdog stops
        # the robot on command loss) and must never block the brain loop.
        with self._lock:
            push = self._push
        try:
            push.send(P.encode_cmd(cmd), flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def set_direction(self, direction) -> None:
        """Single-axis drive char (one of w/s/a/d/-/None)."""
        self._push_cmd(P.cmd_drive(direction))

    def set_axes(self, forward, turn) -> None:
        """Two-axis drive: forward in {'w','s',None} AND turn in {'a','d',None}
        at once, so the bot can arc/diagonal (the proxy round-robins both)."""
        self._push_cmd(P.cmd_axes(forward, turn))

    def calibrate_yaw(self) -> None:
        """Trigger an ESP yaw-gyro recalibration + persist (dashboard button)."""
        self._push_cmd(P.cmd_calibrate())

    def close(self):
        self._stop = True               # drain thread closes _sub as it exits
        self._push.close(0)
