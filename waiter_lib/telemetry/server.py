"""Tiny zero-dependency telemetry HTTP server.

Runs in a daemon thread alongside the FSM. Serves:

    /              -> index.html
    /state         -> JSON dump of the latest snapshot
    /frame.jpg     -> JPEG of the latest annotated camera frame
    /map.png       -> PNG of the latest top-down occupancy + path render
    /static/*      -> static assets (CSS, JS)

The controller calls `publish(snapshot, frame_bgr, map_bgr)` once per
tick. The server thread holds onto the latest blob behind a lock and
serves whatever was last published when a request comes in.

Stdlib only; works on Webots, the Pi, and any vanilla Python with no
extra installs.
"""

from __future__ import annotations
import json
import os
import socket
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional

import cv2
import numpy as np


# shared state
_lock = threading.Lock()
_state_json: bytes = b"{}"
_frame_jpg:  bytes = b""
_map_png:    bytes = b""

# Dashboard-controlled map view. The browser POSTs to /view whenever
# the user pans or zooms; the FSM polls `get_view_state()` when
# rendering the next map frame so the change shows up at the very
# next image refresh (~100 ms latency). All in odometry-frame metres.
# None values mean "use the controller-side default from config";
# this lets a fresh dashboard start at the configured baseline.
_view_state = {
    "half_m":   None,   # half-side length of the rendered window
    "centre_x": None,   # view-centre x in odometry frame
    "centre_y": None,
}

# Dashboard A* goal probe. The browser POSTs an odometry-frame (x, y) to
# /goal when the user clicks the map; the FSM polls `get_goal_state()` on
# its telemetry-publish path and, when active, runs a non-committal A*
# from the bot to this point on the same inflated grid the real planner
# uses, purely to visualise the route + search. It NEVER drives the
# motors. POSTing {"clear": true} switches it off.
_goal_state = {
    "x":      None,
    "y":      None,
    "active": False,
}

# Dashboard wall-clear requests. The browser POSTs to /clear; the FSM drains
# this list on its telemetry path and wipes the matching wall cells. A list
# (not a single value) so rapid erase clicks all apply. Each op is either
# {"all": True} or {"x", "y", "radius_m"} (clear a disk around a point).
_wall_clear_ops: list = []

# Dashboard "move bot" request: teleport the odometry pose estimate. The
# browser POSTs an odometry-frame {x, y, yaw_deg?} to /pose; the FSM applies
# it once (latest wins) and re-zeros the IMU offset so it doesn't drift back.
_pose_request: Optional[dict] = None

# Dashboard "disconnect brain" switch. When False, the FSM zeroes every drive
# command so the bot just balances in place (the ESP keeps it upright); the
# brain still maps, plans, and publishes telemetry. Toggled via POST /drive.
_drive_enabled = True

# Dashboard "full reset" request (the map "reset" button). Unlike /clear (which
# wipes only observed walls), this is a clean-slate: wipe ALL walls incl. the
# anchor footprint, drop the breadcrumb trail + current plan + detected cup, and
# re-zero the pose to the origin. The FSM applies it once on its telemetry path.
_reset_request = False

# Dashboard "calibrate yaw" request (the "Calibrate yaw" button). One-shot:
# polled by run_brain, which asks the SensorSource (-> ESP) to re-measure and
# persist its yaw-gyro zero-rate bias. Robot should be held still during it.
_calibrate_request = False

# Dashboard "return home" request (the "Return home" button). One-shot: the FSM
# remembers where it set off from when it last drove to a target (cup / hand /
# map goal); this asks it to drive back there and park. Polled by the FSM.
_return_request = False

# Dashboard FSM mode override (the mode buttons). "auto" = run the normal
# mission (SEARCH spin -> SEEK -> DONE); "pause" = halt the FSM (zero drive, no
# spin) while the ESP keeps the bot balancing. Polled at the TOP of the FSM's
# _run_state, so "pause" overrides everything (search, map goal, return-home).
_MODES = ("auto", "pause")
_mode = "auto"

# Dashboard manual-drive joystick. The browser POSTs {fwd, turn, active} as the
# user drags the on-screen knob (or holds W/A/S/D); run_brain polls it every
# tick and, when active, drives the ESP's two discrete axes directly (w/s + a/d),
# overriding the autonomous FSM. fwd/turn each in {-1, 0, +1}; +turn = CCW/left.
# SAFETY (deadman): the command is timestamped and get_manual_drive() expires it
# after MANUAL_TIMEOUT_S, so if the dashboard is closed or the network drops
# mid-drive the bot stops rather than running away. The browser re-POSTs a
# heartbeat while the stick is held to keep the command alive.
_manual_drive = {"fwd": 0, "turn": 0, "active": False}
_manual_drive_ts = 0.0
MANUAL_TIMEOUT_S = 0.4

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_server_thread: Optional[threading.Thread] = None
_httpd: Optional[ThreadingHTTPServer] = None


_publish_errors = 0


def get_view_state() -> dict:
    """Return the dashboard's current map-view override, or all-None.

    Called by the FSM's telemetry-publish path; each field that's not
    None overrides the controller's compiled-in default. Returns a
    fresh dict so callers can safely mutate / read without the lock.
    """
    with _lock:
        return dict(_view_state)


def set_view_state(half_m: Optional[float] = None,
                   centre_x: Optional[float] = None,
                   centre_y: Optional[float] = None) -> None:
    """Programmatic counterpart to the POST /view endpoint."""
    with _lock:
        if half_m   is not None: _view_state["half_m"]   = float(half_m)
        if centre_x is not None: _view_state["centre_x"] = float(centre_x)
        if centre_y is not None: _view_state["centre_y"] = float(centre_y)


def get_goal_state() -> dict:
    """Return the dashboard's A* goal probe: {x, y, active}.

    Polled by the FSM telemetry-publish path. Returns a fresh dict so
    callers can read without holding the lock.
    """
    with _lock:
        return dict(_goal_state)


def set_goal_state(x: Optional[float], y: Optional[float],
                   active: bool) -> None:
    """Programmatic counterpart to the POST /goal endpoint."""
    with _lock:
        if x is not None: _goal_state["x"] = float(x)
        if y is not None: _goal_state["y"] = float(y)
        _goal_state["active"] = bool(active)


def get_drive_enabled() -> bool:
    """Whether the FSM may drive the motors. False = brain disconnected, bot
    just balances in place. Polled by the FSM's _drive choke point."""
    with _lock:
        return _drive_enabled


def set_drive_enabled(enabled: bool) -> None:
    """Programmatic counterpart to the POST /drive endpoint."""
    global _drive_enabled
    with _lock:
        _drive_enabled = bool(enabled)


def get_mode() -> str:
    """Current FSM mode override ("auto" | "pause"). Polled at the top of the
    FSM's _run_state. "pause" halts the mission (incl. the SEARCH spin)."""
    with _lock:
        return _mode


def set_mode(mode: str) -> None:
    """Programmatic counterpart to POST /mode. Unknown values fall back to
    'auto' so a bad request can never wedge the FSM into a non-mode."""
    global _mode
    with _lock:
        _mode = mode if mode in _MODES else "auto"


def get_wall_clear_ops() -> list:
    """Return AND clear the pending wall-clear ops (apply-once). Polled by
    the FSM telemetry-publish path."""
    global _wall_clear_ops
    with _lock:
        ops, _wall_clear_ops = _wall_clear_ops, []
        return ops


def add_wall_clear_op(op: dict) -> None:
    """Programmatic counterpart to the POST /clear endpoint."""
    with _lock:
        _wall_clear_ops.append(op)


def get_pose_request() -> Optional[dict]:
    """Return AND clear the pending "move bot" request (apply-once)."""
    global _pose_request
    with _lock:
        r, _pose_request = _pose_request, None
        return r


def set_pose_request(x: float, y: float, yaw_deg: Optional[float] = None) -> None:
    """Programmatic counterpart to the POST /pose endpoint."""
    global _pose_request
    with _lock:
        _pose_request = {"x": float(x), "y": float(y),
                         "yaw_deg": None if yaw_deg is None else float(yaw_deg)}


def get_reset_request() -> bool:
    """Return AND clear the pending full-reset request (apply-once). Polled by
    the FSM telemetry-publish path."""
    global _reset_request
    with _lock:
        r, _reset_request = _reset_request, False
        return r


def request_reset() -> None:
    """Programmatic counterpart to the POST /reset endpoint."""
    global _reset_request
    with _lock:
        _reset_request = True


def get_calibrate_request() -> bool:
    """Return AND clear the pending yaw-calibrate request (apply-once)."""
    global _calibrate_request
    with _lock:
        r, _calibrate_request = _calibrate_request, False
        return r


def request_calibrate() -> None:
    """Programmatic counterpart to the POST /calibrate endpoint."""
    global _calibrate_request
    with _lock:
        _calibrate_request = True


def get_return_request() -> bool:
    """Return AND clear the pending return-home request (apply-once). Polled by
    the FSM telemetry-publish path."""
    global _return_request
    with _lock:
        r, _return_request = _return_request, False
        return r


def request_return() -> None:
    """Programmatic counterpart to the POST /return endpoint."""
    global _return_request
    with _lock:
        _return_request = True


def get_manual_drive() -> dict:
    """Return the dashboard joystick command {fwd, turn, active}. Polled by
    run_brain every tick. Auto-expires (active -> False) once no fresh POST has
    arrived for MANUAL_TIMEOUT_S — a deadman so a closed/disconnected dashboard
    stops the bot instead of leaving the last command latched."""
    with _lock:
        d = dict(_manual_drive)
        expired = (d["active"]
                   and time.monotonic() - _manual_drive_ts > MANUAL_TIMEOUT_S)
    return {"fwd": 0, "turn": 0, "active": False} if expired else d


def set_manual_drive(fwd: int, turn: int, active: bool) -> None:
    """Programmatic counterpart to the POST /manual endpoint."""
    global _manual_drive_ts
    with _lock:
        _manual_drive["fwd"] = max(-1, min(1, int(fwd)))
        _manual_drive["turn"] = max(-1, min(1, int(turn)))
        _manual_drive["active"] = bool(active)
        _manual_drive_ts = time.monotonic()


def publish(state: dict,
            frame_bgr: Optional[np.ndarray] = None,
            map_bgr:   Optional[np.ndarray] = None) -> None:
    """Drop the latest telemetry blob; the server returns it on request.

    Cheap: JSON dump + JPEG encode per call. Skipping the frame/map
    args reuses the previously published blob, so callers can publish
    state more often than imagery.

    Swallows encode errors silently after the first; telemetry must
    never crash the controller.
    """
    global _state_json, _frame_jpg, _map_png, _publish_errors
    try:
        # Sanitise inf/NaN: json doesn't handle them by default, and
        # browsers can't parse "Infinity" in strict JSON anyway.
        state_bytes = json.dumps(state, default=_json_default,
                                  allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as e:
        if _publish_errors < 3:
            print(f"telemetry publish JSON error: {e}")
            _publish_errors += 1
        return
    fb = None
    mb = None
    try:
        if frame_bgr is not None:
            # Downscale the frame to "360p" (360 px on the short side)
            # before JPEG-encoding, ONLY for the dashboard stream. The
            # full-resolution image is still used by perception (ArUco
            # decode needs every pixel); this resize happens just
            # before the network hop, so it cuts WiFi bandwidth without
            # touching detection quality.
            #
            # Camera renders portrait (1848 wide × 2460 tall), so
            # 360 px short side = 360 wide x 480 tall = 0.17 MP, about
            # 25x fewer pixels to JPEG-compress and send. At quality 75
            # that's typically ~20 KB/frame, ~200 KB/s at 10 Hz,
            # comfortable even on bad WiFi.
            h, w = frame_bgr.shape[:2]
            short_side = min(w, h)
            if short_side > 360:
                scale = 360.0 / short_side
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                # INTER_AREA is the right resampler for downscaling:
                # avoids the moire patterns INTER_LINEAR produces on
                # high-frequency content like ArUco grids.
                stream_bgr = cv2.resize(frame_bgr, (new_w, new_h),
                                         interpolation=cv2.INTER_AREA)
            else:
                stream_bgr = frame_bgr
            ok, buf = cv2.imencode(".jpg", stream_bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                fb = buf.tobytes()
        if map_bgr is not None:
            ok, buf = cv2.imencode(".png", map_bgr)
            if ok:
                mb = buf.tobytes()
    except Exception as e:
        if _publish_errors < 3:
            print(f"telemetry publish encode error: {e}")
            _publish_errors += 1
    with _lock:
        _state_json = state_bytes
        if fb is not None: _frame_jpg = fb
        if mb is not None: _map_png  = mb


def _json_default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer, np.floating)): return o.item()
    if o == float("inf"): return None
    if o != o: return None             # NaN
    return str(o)


# handler
class _Handler(BaseHTTPRequestHandler):
    # Silence default request logging; would spam stdout at 10 Hz
    def log_message(self, format, *args): return

    def do_GET(self):
        # Strip any query string ("?t=12345" cache-buster) before
        # matching. Without this, "/frame.jpg?t=12345" falls through
        # to 404, which was exactly the dashboard-image bug.
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html; charset=utf-8",
                              cache=False)
        elif path == "/state":
            with _lock: body = _state_json
            self._serve_bytes(body, "application/json")
        elif path == "/frame.jpg":
            with _lock: body = _frame_jpg
            if not body:
                self.send_error(503, "no frame yet"); return
            self._serve_bytes(body, "image/jpeg", cache=False)
        elif path == "/map.png":
            with _lock: body = _map_png
            if not body:
                self.send_error(503, "no map yet"); return
            self._serve_bytes(body, "image/png", cache=False)
        elif path == "/view":
            # Latest view-state, useful for a dashboard reload to sync
            # back to whatever the previous session was looking at.
            with _lock:
                body = json.dumps(_view_state).encode("utf-8")
            self._serve_bytes(body, "application/json")
        elif path == "/goal":
            # Latest A* goal probe, lets a reloaded dashboard re-show
            # the marker the previous session dropped.
            with _lock:
                body = json.dumps(_goal_state).encode("utf-8")
            self._serve_bytes(body, "application/json")
        elif path == "/drive":
            # Brain-connected switch, so a reloaded dashboard syncs the toggle.
            with _lock:
                body = json.dumps({"enabled": _drive_enabled}).encode("utf-8")
            self._serve_bytes(body, "application/json")
        elif path.startswith("/static/"):
            name = path.replace("/static/", "", 1)
            ext  = os.path.splitext(name)[1].lower()
            mime = {".css": "text/css", ".js": "application/javascript",
                    ".png": "image/png"}.get(ext, "application/octet-stream")
            # No-cache for static assets during development; saves
            # constantly fighting browser caches while we iterate.
            self._serve_file(name, mime, cache=False)
        else:
            self.send_error(404)

    def _read_json_body(self):
        """Parse a small JSON request body. On any problem, send 400 and
        return None (caller should bail). Length-limited defensively;
        we expect a few dozen bytes and reject anything larger than 1 KiB."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 1024:
                self.send_error(400, "bad length"); return None
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as e:
            self.send_error(400, f"bad json: {e}"); return None

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/view", "/goal", "/clear", "/pose", "/drive", "/reset",
                        "/calibrate", "/manual", "/return", "/mode"):
            self.send_error(404); return
        body = self._read_json_body()
        if body is None:
            return

        if path == "/pose":
            try:
                px = float(body["x"]); py = float(body["y"])
            except (KeyError, TypeError, ValueError):
                self.send_error(400, "need x,y"); return
            set_pose_request(px, py, body.get("yaw_deg"))
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/clear":
            # {"all": true} wipes all observed walls; {x, y, radius_m}
            # clears a disk around a point. The FSM applies it next tick.
            if body.get("all"):
                add_wall_clear_op({"all": True})
            else:
                try:
                    cx = float(body["x"]); cy = float(body["y"])
                except (KeyError, TypeError, ValueError):
                    self.send_error(400, "need x,y or all"); return
                r = float(body.get("radius_m", 0.15))
                add_wall_clear_op({"x": cx, "y": cy,
                                   "radius_m": max(0.02, min(5.0, r))})
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/goal":
            # {"clear": true} disables the probe; {x, y} sets+enables it.
            if body.get("clear"):
                set_goal_state(None, None, active=False)
            else:
                try:
                    gx = float(body["x"]); gy = float(body["y"])
                except (KeyError, TypeError, ValueError):
                    self.send_error(400, "need x,y or clear"); return
                set_goal_state(gx, gy, active=True)
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/drive":
            # {"enabled": false} disconnects the brain (bot balances in place);
            # {"enabled": true} reconnects it.
            set_drive_enabled(bool(body.get("enabled", True)))
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/reset":
            # Full clean-slate reset (the map "reset" button). No body needed.
            request_reset()
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/calibrate":
            # Yaw-gyro recalibration (the "Calibrate yaw" button). No body
            # needed; run_brain forwards it to the ESP. Hold the robot still.
            request_calibrate()
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/return":
            # Drive back to where the bot set off from and park. No body needed.
            request_return()
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/mode":
            # {"mode": "auto"|"pause"}. "pause" halts the FSM (stops the SEARCH
            # spin); "auto" resumes the mission. Unknown -> "auto" (see set_mode).
            set_mode(str(body.get("mode", "auto")))
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        if path == "/manual":
            # Joystick / WASD teleop. {fwd, turn} each clamped to {-1,0,1};
            # {"active": false} (or release) stops. The browser heartbeats while
            # the stick is held; the server-side deadman expires it otherwise.
            try:
                fwd = int(body.get("fwd", 0)); turn = int(body.get("turn", 0))
            except (TypeError, ValueError):
                self.send_error(400, "need int fwd,turn"); return
            set_manual_drive(fwd, turn, bool(body.get("active", True)))
            self._serve_bytes(b"{\"ok\":true}", "application/json")
            return

        # /view: pull the three keys (any subset is fine). The setter
        # clamps to sensible bounds since a runaway client could
        # otherwise ask for a 0 m view and divide-by-zero downstream.
        half_m   = body.get("half_m")
        centre_x = body.get("centre_x")
        centre_y = body.get("centre_y")
        if half_m is not None:
            half_m = max(0.05, min(5.0, float(half_m)))
        set_view_state(half_m, centre_x, centre_y)
        self._serve_bytes(b"{\"ok\":true}", "application/json")

    # helpers
    def _serve_bytes(self, body: bytes, mime: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, name: str, mime: str, cache: bool = True):
        path = os.path.join(_STATIC_DIR, name)
        if not os.path.isfile(path):
            self.send_error(404); return
        with open(path, "rb") as f:
            self._serve_bytes(f.read(), mime, cache=cache)


# start / stop
def start(port: int = 8080) -> None:
    """Launch the server in a daemon thread. Idempotent."""
    global _server_thread, _httpd
    if _server_thread is not None:
        return
    _httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    _server_thread = threading.Thread(target=_httpd.serve_forever,
                                       name="telemetry-http",
                                       daemon=True)
    _server_thread.start()
    print(f"Telemetry server on http://{_lan_ip()}:{port}  "
          f"(also http://localhost:{port})")


def stop() -> None:
    global _server_thread, _httpd
    if _httpd is not None:
        _httpd.shutdown()
        _httpd.server_close()
        _httpd = None
    _server_thread = None


def _lan_ip() -> str:
    """Best-effort guess of the host's LAN IP (works on macOS + Linux)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "?"
