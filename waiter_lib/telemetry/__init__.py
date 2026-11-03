"""Live telemetry HTTP server for sim + Pi.

Usage:
    from waiter_lib import telemetry
    telemetry.start(port=8080)
    ...
    telemetry.publish(snapshot_dict, annotated_frame, map_image)

Then point a browser at http://<bot-ip>:8080.
"""

from .server import (start, stop, publish, get_view_state, set_view_state,
                     get_goal_state, set_goal_state,
                     get_drive_enabled, set_drive_enabled,
                     get_mode, set_mode,
                     get_wall_clear_ops, add_wall_clear_op,
                     get_pose_request, set_pose_request,
                     get_reset_request, request_reset,
                     get_calibrate_request, request_calibrate,
                     get_return_request, request_return,
                     get_manual_drive, set_manual_drive)
from .overlay import annotate, render_map

__all__ = [
    "start", "stop", "publish",
    "get_view_state", "set_view_state",
    "get_goal_state", "set_goal_state",
    "get_drive_enabled", "set_drive_enabled",
    "get_mode", "set_mode",
    "get_wall_clear_ops", "add_wall_clear_op",
    "get_pose_request", "set_pose_request",
    "get_reset_request", "request_reset",
    "get_calibrate_request", "request_calibrate",
    "get_return_request", "request_return",
    "get_manual_drive", "set_manual_drive",
    "annotate", "render_map",
]
