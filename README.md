# Waiter robot autonomy and perception

A standalone source snapshot of Michael Wettasinghe’s autonomy and perception
software for a six-person Imperial EE2 service-robot project. It contains
clearance-aware A* planning and path smoothing, a Dynamic Window local planner,
LiDAR occupancy mapping and scan matching, ArUco/PnP cup perception, a MediaPipe
hand-gesture detector, a mission state machine and telemetry UI. A Raspberry Pi
bridge streams sensors and relays drive commands over ZeroMQ to an off-board
Python brain.

This is the source subset approved by Michael for public publication.
The balance controller, mechanical design and motor/power electronics are outside
Michael’s stated contribution and are excluded. This snapshot does not include a
license or grant new reuse rights. Python dependencies are referenced rather than
vendored; their upstream licenses still apply.

## Layout

- `waiter_lib/`: shared planning, motion, perception, world state, configuration,
  sensor interface, decision state machine and telemetry server/static UI.
- `mac/`: off-board runner, transport client and hardware-to-brain adapter.
- `pi/`: D500 serial driver, sensor/actuator proxy and hand-gesture detector
  (the latter is imported by the off-board brain).
- `transport/`: shared message encoding and port definitions.

## Dependencies and integration

The preserved dependency declarations are `mac/requirements.txt` and
`pi/requirements.txt`. The off-board stack uses NumPy, OpenCV contrib, Pillow,
ZeroMQ and the legacy MediaPipe Solutions API. The Pi proxy uses NumPy, pyserial
and ZeroMQ; camera support additionally expects a working system Picamera2
installation. Dependencies were not installed or resolved for this snapshot.

For the original runner’s environment convention, create a local Python 3.12
virtual environment at `mac/.venv-local` and install `mac/requirements.txt` into
it. The Pi needs its own environment and platform-specific camera installation.
After reviewing hardware integration, the off-board entry point is
`python -m mac.run_brain --host ROBOT_HOST`; replace `ROBOT_HOST` explicitly with
your own Pi proxy address. The Pi entry point is `python -m pi.proxy`; real sensors
require its explicit camera/LiDAR/serial flags. These commands were not run.
Do not run two drive clients against the same robot.

The transport ports and wire messages are defined in `transport/protocol.py`.
The adapter expects external firmware telemetry and drive-character semantics;
that firmware is not included. Wheel geometry, camera geometry, LiDAR orientation,
colour correction and drive signs retain project-specific calibration and need
review for another robot. No Webots worlds, model assets or simulator entry points
are included, so this is not a complete simulator distribution.

The original telemetry HTTP server listens on all interfaces; the ZeroMQ proxy
also binds network interfaces. These interfaces have no authentication layer in
this subset. Review isolation and command access before connecting any actuator.

## Provenance and checks

Extracted from source commit `31f1b1dcadacc083c4532880f54c2e03b399dd17`.
Michael has identified the autonomy/perception/bridge software as his work;
selected-path Git history lists Michael as commit author. This does not replace
confirmation of any institutional or third-party disclosure obligations.

For this standalone snapshot, only the off-board runner was changed: its personal absolute import path became
repository-relative and its lab-network default became a required explicit host.
The algorithm and control logic are unchanged. Private access/deployment material,
team balance firmware, recordings, environments and Git history are omitted.

Validation is limited to parsing every included Python file without importing or
executing it, reviewing local import targets and scanning the text package for
private paths, network addresses and credential indicators. No hardware, service,
network connection, dependency install, simulation or end-to-end test was run.
