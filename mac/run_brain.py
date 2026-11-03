"""Run the real waiter_lib WaiterFSM brain on the live D500 robot.

This is the "replace the brain" entry point. It constructs the EXACT FSM
the Webots sim runs (perception, clearance-A*, any-angle smoothing, LiDAR
ICP, DWA, RECAL, the SEARCH->SEEK->CENTRE->DONE cup mission) and ticks it
against the real robot, with BrainSensorSource standing in for the sim's
WebotsSensorSource. Nothing in waiter_lib changes; only the SensorSource does.

The brain serves its own rich telemetry dashboard on http://localhost:8080
(started inside WaiterFSM.__init__ from CONFIG.tuning.telemetry_port): the
A*-goal click, route + search overlays, wall map, move-bot/erase tools, and
the FSM state badge. Only ONE brain should run at a time — two would both push
drive commands to the Pi and fight over the command channel.

Usage (on the Mac, in the venv):
    mac/.venv-local/bin/python -m mac.run_brain --host ROBOT_HOST

Notes / known gaps on real hardware (by design — see the scoping decision):
  - The mission hunts a sim-specific ArUco cup + a 12-marker anchor cube at
    hardcoded odometry coordinates. Those fiducials don't exist in the room,
    so the bot stays in SEARCH (spinning) until they're physically placed /
    the anchor config is re-measured. The stack still runs end-to-end:
    mapping, A*, the dashboard, and drive-char output all work live.
  - Camera intrinsics in CONFIG are the sim's 1848x2460; real frames are a
    different size, so ArUco PnP distances are off until calibrated. Detection
    still runs.
  - Wheels won't physically move until the fixed firmware is flashed; the
    quantised drive chars are sent and shown in the proxy log meanwhile.
"""

from __future__ import annotations
import argparse
import dataclasses
import math
import os
import sys
import time

# This preview places waiter_lib at the repository root. Resolve it relative
# to this file so the runner does not depend on a personal checkout path.
WAITER_LIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAITER_LIB_DIR)

# This repo's root, so `mac.*` and `transport.*` resolve when run as a module
# or as a bare script path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _ensure_venv() -> None:
    """Re-exec under mac/.venv-local if launched with any other interpreter.

    mediapipe / numpy / cv2 only load from .venv-local on this Mac — the .venv
    copy throws "library load disallowed by system policy". A wrong-venv launch
    used to silently disable MediaPipe (hand detector OFF) instead of failing.
    Self-correct here so ANY launch — `python -m mac.run_brain`, bare `python`,
    the broken .venv, a stray shell, another agent — lands on the one
    interpreter that works. The env-var guard makes the exec idempotent so it
    can't loop, and a missing .venv-local degrades to a loud warning."""
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".venv-local")
    on_target = os.path.realpath(sys.prefix) == os.path.realpath(venv_dir)
    if on_target or os.environ.get("WAITER_BRAIN_REEXEC") == "1":
        return
    target = os.path.join(venv_dir, "bin", "python")
    if not os.path.exists(target):
        print(f"[run_brain] WARNING: {target} missing; staying on "
              f"{sys.executable} (MediaPipe likely unavailable)", flush=True)
        return
    os.environ["WAITER_BRAIN_REEXEC"] = "1"
    print(f"[run_brain] re-exec under {target}", flush=True)
    os.execv(target, [target, "-u", os.path.abspath(__file__), *sys.argv[1:]])


# Must run BEFORE the numpy/mediapipe-touching imports below, so the re-exec
# (if any) happens on a clean interpreter.
_ensure_venv()

from waiter_lib.config import CONFIG                       # noqa: E402
from waiter_lib.decision.state_machine import WaiterFSM    # noqa: E402
from waiter_lib import telemetry                           # noqa: E402
from mac.brain_source import BrainSensorSource             # noqa: E402


# ── Follow-palm behaviour ─────────────────────────────────────────────────────
# An OPEN palm facing the camera becomes a moving A* goal placed a fixed
# "carrot" distance ahead along its bearing, so the brain drives toward it
# (reusing the GOTO path: set_goal_state -> _state_goto). The palm's horizontal
# position in the frame + the camera's horizontal FOV give the bearing; when the
# palm leaves the frame the bot stops (parks where it is) and releases the goal.
PALM_HFOV_RAD     = 0.852   # uprighted-frame horizontal span = IMX219 SHORT
                            # axis (~49°) after the 90° CW portrait remount; the
                            # wide ~62° axis is now VERTICAL (max vertical FOV).
PALM_CARROT_M     = 0.80    # goal distance ahead along the palm bearing
PALM_LOST_GRACE_S = 0.6     # palm gone this long -> park + release
PALM_BEARING_SIGN = 1.0     # flip to -1.0 if the bot turns AWAY from the hand
PALM_BEARING_EMA  = 0.5     # bearing smoothing (0..1; lower = smoother)

# Manual-drive joystick (dashboard). +turn = CCW/left to match the brain's turn
# sign; flip to -1 if the bot turns the wrong way for a given stick push.
MANUAL_TURN_SIGN  = 1


def _hw_camera_config(src, settle_s: float = 4.0):
    """Rebuild the camera intrinsics around the REAL live frame.

    CONFIG.camera describes the SIM frame (1848x2460). ArUco PnP maps a marker's
    pixel position to a bearing using that geometry, so on the real ~480x640 feed
    a marker held DEAD-CENTRE reads as wildly off-axis -> the cup's direction is
    wrong. Probe one live frame for its true (H, W) and rebuild the intrinsics:
    principal point becomes the real centre and the focal length follows from the
    unchanged fov_h (0.852 rad = the 48.8 deg short-side FOV, which is the
    HORIZONTAL axis of the uprighted frame after the 90 deg portrait remount).
    Sim is untouched -- it uses CONFIG directly via the Webots entry point."""
    t0 = time.monotonic()
    shape = None
    while time.monotonic() - t0 < settle_s:
        f = src.get_camera_frame()
        if f is not None:
            shape = f.shape
            break
        time.sleep(0.05)
    if shape is None:
        print("[run_brain] WARNING: no camera frame in "
              f"{settle_s:.0f}s; keeping SIM intrinsics "
              "(cup bearing will be wrong until the camera is up)", flush=True)
        return CONFIG
    h, w = int(shape[0]), int(shape[1])
    cam = dataclasses.replace(CONFIG.camera, width=w, height=h)
    print(f"[run_brain] camera intrinsics -> {w}x{h} "
          f"(fx={cam.f_x:.0f} fy={cam.f_y:.0f} cx={w/2:.0f} cy={h/2:.0f}); "
          f"sim was {CONFIG.camera.width}x{CONFIG.camera.height}", flush=True)
    return dataclasses.replace(CONFIG, camera=cam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True,
                    help="Pi proxy hostname or IP; no network default is supplied")
    ap.add_argument("--secs", type=float, default=0.0,
                    help="run duration; 0 = run until Ctrl-C")
    ap.add_argument("--frame-rot-k", type=int, default=3,
                    help="rotate camera frames k*90 CCW to upright the feed; "
                         "default 3 undoes the 90 deg CW portrait remount "
                         "(use 1 if the dashboard shows it upside-down)")
    args = ap.parse_args()

    src = BrainSensorSource(args.host,
                            wheel_radius_m=CONFIG.robot.wheel_radius_m,
                            frame_rot_k=args.frame_rot_k)
    # Override the sim's camera intrinsics with ones built for the REAL frame
    # size, so ArUco cup bearings are correct (a centred marker reads ~straight
    # ahead instead of off to one side).
    cfg = _hw_camera_config(src)
    # lidar_deskew=True: the D500 is a spinning sensor, so motion-de-skew
    # each scan (interpolate pose per ray) to stop turns smearing phantom walls.
    # HARDWARE CALIBRATION: the D500's spin sense ran de-skew the wrong way at
    # DIR=+1 (turns smeared the map); -1 matches this rig. Override here (not in
    # state_machine) to avoid touching the file the other session is editing.
    fsm = WaiterFSM(src, cfg, lidar_deskew=False)
    print(f"[run_brain] FSM constructed against {args.host}. "
          f"Dashboard: http://localhost:{CONFIG.tuning.telemetry_port}")
    if args.secs > 0:
        print(f"[run_brain] running for {args.secs:.0f}s")
    else:
        print("[run_brain] running until Ctrl-C")

    # Resilient loop: a WiFi blip makes tick() report the feed silent. Don't
    # exit (that drops the dashboard). Keep looping; the ZMQ SUB auto-
    # reconnects and tick() returns True again once the Pi resumes.
    ticks = 0
    link_up = True
    following = False        # currently steering toward a palm?
    last_palm_t = 0.0        # src.time() of the last open-palm sighting
    bearing_f = 0.0          # smoothed palm bearing (rad, robot frame)
    try:
        while True:
            if args.secs > 0 and src.time() > args.secs:
                break
            if src.tick():
                if not link_up:
                    print(f"[run_brain] link RESTORED at t={src.time():.0f}s",
                          flush=True)
                    link_up = True
                fsm.step()
                ticks += 1
                # Dashboard "Calibrate yaw" button -> tell the ESP to re-measure
                # + persist its yaw-gyro bias. Polled here (not in the shared
                # FSM) so waiter_lib stays untouched; one-shot, auto-cleared.
                if telemetry.get_calibrate_request():
                    src.calibrate_yaw()
                    print(f"[run_brain] yaw calibrate requested at "
                          f"t={src.time():.0f}s", flush=True)
                # Manual joystick (dashboard) OVERRIDES autonomy. fsm.step()
                # already ran this tick (so mapping/planning keep going); issuing
                # the human's command LAST is what actually reaches the ESP — the
                # proxy drives whatever axes were set most recently. The command
                # deadman-expires server-side, so a closed dashboard stops the bot.
                manual = telemetry.get_manual_drive()
                manual_active = bool(manual["active"])
                # Parked (freeze dead-reckoned translation) only when NOTHING is
                # driving: neither the connected brain nor a manual command. The
                # marker would otherwise drift on reported velocity while idle.
                src.set_parked(not (telemetry.get_drive_enabled()
                                    or manual_active))
                if manual_active:
                    src.drive_manual(int(manual["fwd"]),
                                     MANUAL_TURN_SIGN * int(manual["turn"]))
                elif fsm._returning:
                    # FSM is driving back to where it set off from (the UI
                    # "return home" button). Don't fight it with a hand goal;
                    # drop follow state so we don't re-park when it arrives.
                    following = False
                else:
                    # Follow-palm: an open palm steers a moving A* goal a carrot
                    # distance ahead along its bearing (reusing GOTO). Lose the
                    # palm -> park where we are, then release the goal so manual
                    # goals / the mission resume. Kept here (not in waiter_lib) so
                    # the brain stays generic; engages only while a palm's in view.
                    palm = src.get_open_palm()           # (cx, cy, size) or None
                    t_now = src.time()
                    if palm is not None:
                        cx = palm[0]
                        bearing = PALM_BEARING_SIGN * (0.5 - cx) * PALM_HFOV_RAD
                        bearing_f = (PALM_BEARING_EMA * bearing
                                     + (1.0 - PALM_BEARING_EMA) * bearing_f)
                        ang = fsm.world.bot_yaw + bearing_f
                        gx = fsm.world.bot_x + PALM_CARROT_M * math.cos(ang)
                        gy = fsm.world.bot_y + PALM_CARROT_M * math.sin(ang)
                        telemetry.set_goal_state(gx, gy, active=True)
                        last_palm_t = t_now
                        if not following:
                            print(f"[run_brain] palm acquired -> following "
                                  f"(t={t_now:.0f}s)", flush=True)
                            following = True
                    elif following and (t_now - last_palm_t) > PALM_LOST_GRACE_S:
                        telemetry.set_goal_state(fsm.world.bot_x,
                                                 fsm.world.bot_y, active=True)
                        following = False
                        print(f"[run_brain] palm lost -> parked "
                              f"(t={t_now:.0f}s)", flush=True)
            elif link_up:
                link_up = False
                print(f"[run_brain] link LOST (Pi feed silent) at "
                      f"t={src.time():.0f}s - holding; dashboard stays up",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[run_brain] interrupted")
    finally:
        src.close()
        state = getattr(fsm.world, "state", "?")
        walls = (int(fsm.world.wall_status.sum())
                 if getattr(fsm.world, "wall_status", None) is not None else -1)
        print(f"[run_brain] done: {ticks} ticks, final state={state}, "
              f"wall cells mapped={walls}")


if __name__ == "__main__":
    main()
