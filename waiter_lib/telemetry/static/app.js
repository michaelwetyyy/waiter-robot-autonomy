/* Telemetry dashboard: polls /state at 10 Hz, refreshes images at 10 Hz.
 *
 * Detects FSM-state transitions client-side and appends to a log so we
 * can see the bot's state history even when scrolling away. Reconnects
 * automatically if the server is restarted.
 */

const STATE_POLL_MS = 100;     // 10 Hz: text fields
const IMG_POLL_MS   = 100;     // 10 Hz: camera frame + map

let lastState = null;
let driveEnabled = true;            // brain-connected toggle (mirrors server)
const $ = (id) => document.getElementById(id);

const fmt = (v, dec = 2, unit = "") =>
  v === null || v === undefined ? "·" : v.toFixed(dec) + unit;
const fmtCm = (v) => v === null ? "∞" : (v * 100).toFixed(0);

function appendLog(msg) {
  const el = $("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
  // Cap log to 200 lines
  const lines = el.textContent.split("\n");
  if (lines.length > 200) el.textContent = lines.slice(-200).join("\n");
}

async function pollState() {
  try {
    const r = await fetch("/state", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    render(s);
  } catch (e) {
    $("state-pill").textContent = "OFFLINE";
    $("state-pill").className = "state-pill";
  }
}

function render(s) {
  $("tick").textContent = s.tick ?? "·";

  // FSM state pill
  const st = s.state || "?";
  const pill = $("state-pill");
  pill.textContent = st;
  pill.className = "state-pill " + st;

  // Log state transitions
  if (lastState && lastState.state !== s.state) {
    appendLog(`t=${s.tick}  ${lastState.state} -> ${s.state}`);
  }
  // Log explicit transition message if provided
  if (s.last_transition && (!lastState || lastState.last_transition !== s.last_transition)) {
    appendLog(`t=${s.tick}  ${s.last_transition}`);
  }
  lastState = s;

  // Pose
  $("bx").textContent   = fmt(s.bot?.x, 2, " m");
  $("by").textContent   = fmt(s.bot?.y, 2, " m");
  $("byaw").textContent = s.bot?.yaw_deg !== undefined ? s.bot.yaw_deg.toFixed(0) + "°" : "·";

  // Cup estimate
  if (s.cup) {
    $("cupx").textContent = fmt(s.cup.x, 2, " m");
    $("cupy").textContent = fmt(s.cup.y, 2, " m");
    $("cupd").textContent = fmt(s.cup.distance, 2, " m");
  } else {
    $("cupx").textContent = $("cupy").textContent = $("cupd").textContent = "·";
  }

  // Near-field
  const n = s.near || {};
  $("hL").textContent = fmtCm(n.hard_left);
  $("L").textContent  = fmtCm(n.left);
  $("A").textContent  = fmtCm(n.ahead);
  $("R").textContent  = fmtCm(n.right);
  $("hR").textContent = fmtCm(n.hard_right);

  // Plan
  $("walls").textContent = s.walls ?? "·";
  $("plen").textContent  = s.path_len ?? "·";
  $("pidx").textContent  = s.path_idx ?? "·";

  // A* goal probe
  const g = s.goal;
  if (g) {
    $("gx").textContent     = fmt(g.x, 2, " m");
    $("gy").textContent     = fmt(g.y, 2, " m");
    $("greach").textContent = g.reachable ? "yes" : "NO";
    $("greach").className   = g.reachable ? "ok" : "bad";
    $("gplen").textContent  = g.reachable ? g.path_len : "-";
    $("gexp").textContent   = g.explored ?? "·";
  } else {
    $("gx").textContent = $("gy").textContent = $("gplen").textContent =
      $("gexp").textContent = "·";
    $("greach").textContent = "·";
    $("greach").className = "";
  }

  // Brain-connected toggle: the server is authoritative, so sync the button
  // to whatever /state reports (covers a reload or a second dashboard).
  if (s.drive_enabled !== undefined) {
    driveEnabled = s.drive_enabled;
    updateBrainBtn();
  }

  // Mode buttons: highlight whichever mode the FSM reports (auto | pause).
  if (s.mode) {
    $("mode-auto").classList.toggle("active", s.mode === "auto");
    $("mode-pause").classList.toggle("active", s.mode === "pause");
  }

  // Return-home button: enabled only when the bot has a home to go back to
  // (i.e. it drove somewhere); shows progress while it's on its way back.
  const rh = $("return-home");
  if (rh) {
    if (s.returning) {
      rh.textContent = "returning…";
      rh.disabled = true;
    } else {
      rh.textContent = "return home";
      rh.disabled = !s.home;
    }
  }

  // Performance (real-time budget, all measured on the brain loop)
  const p = s.perf || {};
  $("tickHz").textContent   = fmt(p.tick_hz, 0, " Hz");
  $("percMs").textContent   = fmt(p.perception_ms, 1, " ms");
  $("planMs").textContent   = fmt(p.plan_ms, 2, " ms");
  $("renderMs").textContent = fmt(p.render_ms, 1, " ms");

  // Pipeline (sensor link health: proves the Mac<->Pi interface is live)
  const pl = s.pipeline || {};
  $("lidarHz").textContent = fmt(pl.lidar_hz, 1, " Hz");
  $("camFps").textContent  = fmt(pl.cam_fps, 1, " fps");
  const lb = $("linkBadge");
  lb.textContent = pl.link_ok ? "LINK OK" : "LINK DOWN";
  lb.className   = "badge " + (pl.link_ok ? "ok" : "bad");

  // Balance (ESP32): tilt = balance angle, yaw = integrated gyro heading,
  // cmd L/R = wheel velocities the brain last commanded.
  const b = s.balance || {};
  $("tilt").textContent   = fmt(b.tilt_deg, 1, "°");
  $("balYaw").textContent = fmt(b.yaw_deg, 0, "°");
  $("cmdLR").textContent  = (b.cmd_l === undefined || b.cmd_l === null)
    ? "·" : fmt(b.cmd_l, 1) + " / " + fmt(b.cmd_r, 1);
}

// Brain connect/disconnect: when disconnected the bot just balances in place
// (the FSM zeroes all drive commands server-side). Optimistic UI update, then
// the next /state poll confirms.
function updateBrainBtn() {
  const bt = $("brain-toggle");
  if (!bt) return;
  bt.textContent = driveEnabled ? "brain connected" : "brain DISCONNECTED";
  bt.classList.toggle("off", !driveEnabled);
}
$("brain-toggle").addEventListener("click", () => {
  driveEnabled = !driveEnabled;
  updateBrainBtn();
  postJSON("/drive", { enabled: driveEnabled });
});

function refreshImages() {
  const t = Date.now();
  $("camera").src = "/frame.jpg?t=" + t;
  $("map").src    = "/map.png?t=" + t;
}

$("clear-log").addEventListener("click", () => $("log").textContent = "");

/*  map zoom/pan
 * The view is described in WORLD METRES: a half-side `halfM` and a
 * centre `(cx, cy)` in the bot's odometry frame. The dashboard POSTs
 * these to /view; the controller picks them up on the next render and
 * re-rasterises the map with the new framing. So zooming in actually
 * shrinks halfM (= more grid cells per pixel) rather than just
 * enlarging an already-rasterised image.
 *
 * The default values mirror config.tuning's compiled-in baseline; we
 * GET /view once on load in case a previous session was looking at
 * something specific. We also push a CSS transform live during the
 * drag/wheel gesture so the user gets instant feedback before the
 * server's next frame arrives (~100 ms refresh).
 */
const VIEW_DEFAULT = { halfM: 1.40, cx: 0.80, cy: 0.0 };
const VIEW = { ...VIEW_DEFAULT };
const VIEW_MIN_HALF = 0.05;     // 5 cm half-side = tightest zoom (clamped server-side too)
const VIEW_MAX_HALF = 5.0;      // 5 m half-side = full grid extent
const POST_DEBOUNCE_MS = 60;    // coalesce rapid gestures into one POST

const mapViewport = $("map-viewport");
const mapImg = $("map");
const zoomLevel = $("zoom-level");

// CSS-transform companion state: applied to the <img> while a gesture
// is in flight so the user sees the new framing immediately, then
// reset to identity once a fresh server-rendered frame is loaded.
let liveTx = 0, liveTy = 0, liveScale = 1;
let postPending = null;
let pendingRefresh = false;

function applyLiveTransform() {
  mapImg.style.transform =
    `translate(${liveTx}px, ${liveTy}px) scale(${liveScale})`;
}

function clearLiveTransform() {
  liveTx = 0; liveTy = 0; liveScale = 1;
  mapImg.style.transform = "";
}

function updateZoomLabel() {
  // Show zoom as ratio of default-half / current-half. 100% = default
  // baseline view, 200% = zoomed to half the area, etc.
  const pct = Math.round((VIEW_DEFAULT.halfM / VIEW.halfM) * 100);
  zoomLevel.textContent = pct + "%";
}

// POST a JSON body, swallowing transport errors (server-restart races;
// the next gesture/click retries). Shared by the view and goal endpoints.
function postJSON(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => { /* ignore; next call retries */ });
}

function postView() {
  if (postPending !== null) clearTimeout(postPending);
  postPending = setTimeout(() => {
    postPending = null;
    postJSON("/view", { half_m: VIEW.halfM, centre_x: VIEW.cx, centre_y: VIEW.cy });
  }, POST_DEBOUNCE_MS);
}

// Map a viewport pixel (relative to viewport rect) to a world-frame
// (x, y) coordinate using the CURRENT view (halfM + centre).
function pxToWorld(px, py) {
  const w = mapViewport.clientWidth, h = mapViewport.clientHeight;
  // World x grows rightward, world y grows upward (rendered image
  // already inverts y). 2*halfM metres span the viewport's px width.
  const wx = VIEW.cx + (px / w - 0.5) * 2 * VIEW.halfM;
  const wy = VIEW.cy + (0.5 - py / h) * 2 * VIEW.halfM;
  return [wx, wy];
}

function resetMapView() {
  Object.assign(VIEW, VIEW_DEFAULT);
  clearLiveTransform();
  updateZoomLabel();
  postView();
}

// Wheel zoom: anchor on cursor position so the world point under
// the cursor stays under the cursor across the zoom.
mapViewport.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = mapViewport.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  // Negative deltaY = scroll up = zoom in. exp() smooths per-event
  // deltas (mouse wheel notches vs trackpad's smaller increments).
  const factor = Math.exp(-e.deltaY * 0.0015);
  const newHalf = Math.max(VIEW_MIN_HALF,
                   Math.min(VIEW_MAX_HALF, VIEW.halfM / factor));
  if (newHalf === VIEW.halfM) return;
  // Pin the cursor's world point: world_pt = centre + (px/w - 0.5)*2*halfM
  // is invariant => centre_new = world_pt - (px/w - 0.5)*2*halfM_new
  const [wx, wy] = pxToWorld(px, py);
  const w = mapViewport.clientWidth, h = mapViewport.clientHeight;
  VIEW.cx = wx - (px / w - 0.5) * 2 * newHalf;
  VIEW.cy = wy - (0.5 - py / h) * 2 * newHalf;
  VIEW.halfM = newHalf;
  // Instant visual feedback: scale up the rasterised image by the
  // same factor around the cursor, then clear once a fresh frame
  // arrives.
  liveScale *= factor;
  liveTx = px - (px - liveTx) * factor;
  liveTy = py - (py - liveTy) * factor;
  applyLiveTransform();
  updateZoomLabel();
  postView();
}, { passive: false });

// Click-drag pan. Pixel deltas convert to metres via the current
// halfM <-> viewport-width ratio. Drag right => view centre moves LEFT
// (you're sliding the world to the right under a fixed window).
// A pointer gesture whose start-to-end displacement is under this many pixels
// is treated as a click (set the A* goal) rather than a pan.
const CLICK_THRESH_PX = 6;
let dragging = false;
let dragLastX = 0, dragLastY = 0;
let dragStartX = 0, dragStartY = 0;   // press position, for click-vs-drag
mapViewport.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 && e.pointerType === "mouse") return;
  // Don't start a map gesture when the press lands on a toolbar control;
  // the viewport's setPointerCapture would otherwise swallow the button's
  // click (events get redirected to the viewport), so buttons never fire.
  if (e.target.closest(".zoom-toolbar")) return;
  dragging = true;
  dragLastX = dragStartX = e.clientX;
  dragLastY = dragStartY = e.clientY;
  mapViewport.classList.add("dragging");
  mapViewport.setPointerCapture(e.pointerId);
});
mapViewport.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dxPx = e.clientX - dragLastX;
  const dyPx = e.clientY - dragLastY;
  dragLastX = e.clientX;
  dragLastY = e.clientY;
  const w = mapViewport.clientWidth, h = mapViewport.clientHeight;
  const metresPerPx = (2 * VIEW.halfM) / w;
  VIEW.cx -= dxPx * metresPerPx;
  VIEW.cy += dyPx * metresPerPx;  // sign flip: dashboard +y is screen-up
  // Live feedback while the gesture is in flight: translate the
  // rasterised image with the drag, then clear when the next
  // server-rendered frame arrives.
  liveTx += dxPx;
  liveTy += dyPx;
  applyLiveTransform();
  postView();
});
const endDrag = (e) => {
  if (!dragging) return;
  dragging = false;
  mapViewport.classList.remove("dragging");
  if (e && e.pointerId !== undefined) {
    try { mapViewport.releasePointerCapture(e.pointerId); } catch (_) {}
  }
};
mapViewport.addEventListener("pointerup", (e) => {
  // A near-stationary press is a click: drop the A* goal at that world
  // point. Anything that moved more than the threshold was a pan.
  const moved = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
  const wasClick = dragging && moved < CLICK_THRESH_PX;
  endDrag(e);
  if (wasClick) {
    const rect = mapViewport.getBoundingClientRect();
    const [wx, wy] = pxToWorld(e.clientX - rect.left, e.clientY - rect.top);
    if (mapMode === "movebot")    postJSON("/pose",  { x: wx, y: wy });
    else if (mapMode === "erase") postJSON("/clear", { x: wx, y: wy, radius_m: ERASE_RADIUS_M });
    else                          postGoal(wx, wy);
  }
});
mapViewport.addEventListener("pointercancel", endDrag);
mapViewport.addEventListener("pointerleave", endDrag);

/*  A* goal probe
 * Clicking the map POSTs an odometry-frame (x, y) to /goal. The
 * controller runs a non-committal A* to that point and renders the
 * route + search cloud on the next map frame; the readout panel shows
 * reachability + route length. "clear" turns the probe off.
 */
function postGoal(wx, wy) {
  postJSON("/goal", { x: wx, y: wy });
}

function clearGoal() {
  postJSON("/goal", { clear: true });
}

$("clear-goal").addEventListener("click", clearGoal);

/*  Map click modes
 * One exclusive mode decides what a map click does:
 *   goal    (default) - set the A* probe goal
 *   erase             - clear walls within ERASE_RADIUS_M of the click
 *   movebot           - teleport the bot's pose to the click (POST /pose)
 * "clear all" is a one-shot button (no mode). The toggles are mutually
 * exclusive; clicking the active one returns to goal mode. The FSM applies
 * erase/move on its next tick; live LiDAR re-stamps any wall still in view.
 */
const ERASE_RADIUS_M = 0.15;
let mapMode = "goal";

const goalHint = $("goal-hint");
const HINT = {
  goal:    '<b>click</b> set A* goal · <b>drag</b> pan · <b>scroll</b> zoom',
  erase:   '<b>click</b> erase walls (' + (ERASE_RADIUS_M * 100).toFixed(0) + ' cm) · <b>drag</b> pan',
  movebot: '<b>click</b> set bot position · <b>drag</b> pan',
};

function setMapMode(mode) {
  mapMode = mode;
  $("erase-toggle").classList.toggle("active", mode === "erase");
  $("move-bot").classList.toggle("active",     mode === "movebot");
  mapViewport.classList.toggle("erasing",  mode === "erase");
  mapViewport.classList.toggle("placing",  mode === "movebot");
  if (goalHint) goalHint.innerHTML = HINT[mode] || HINT.goal;
}

$("calibrate-yaw").addEventListener("click", () => {
  // Recalibrate the ESP yaw-gyro bias (hold the robot still ~1s). The firmware
  // averages the gyro and persists the result to flash.
  const btn = $("calibrate-yaw");
  postJSON("/calibrate", {});
  const orig = btn.innerHTML;
  btn.innerHTML = "hold still…";
  btn.disabled = true;
  setTimeout(() => { btn.innerHTML = orig; btn.disabled = false; }, 1500);
});

$("return-home").addEventListener("click", () => postJSON("/return", {}));

// FSM mode buttons. Optimistically highlight on click; the next /state poll
// confirms (and corrects, if the POST was rejected). "pause" stops the spin.
$("mode-auto").addEventListener("click",  () => { postJSON("/mode", { mode: "auto"  }); });
$("mode-pause").addEventListener("click", () => { postJSON("/mode", { mode: "pause" }); });

$("clear-walls").addEventListener("click", () => postJSON("/clear", { all: true }));
$("erase-toggle").addEventListener("click", () => setMapMode(mapMode === "erase"   ? "goal" : "erase"));
$("move-bot").addEventListener("click",     () => setMapMode(mapMode === "movebot" ? "goal" : "movebot"));

mapViewport.addEventListener("dblclick", (e) => {
  e.preventDefault();
  resetMapView();
});

document.querySelectorAll("[data-zoom-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const action = btn.dataset.zoomAction;
    if (action === "reset") {
      // Full reset: wipe the map + re-zero the pose on the brain, then reset
      // the view. Confirm first (destructive). Double-click the map for a
      // view-only reset.
      if (!confirm("Full reset: wipe the entire map (walls, trail, plan) "
                   + "and re-zero the pose to the origin?")) return;
      postJSON("/reset", {});
      resetMapView();
      return;
    }
    // Zoom around viewport centre when invoked via button.
    const factor = action === "in" ? 1.25 : 1 / 1.25;
    const newHalf = Math.max(VIEW_MIN_HALF,
                     Math.min(VIEW_MAX_HALF, VIEW.halfM / factor));
    VIEW.halfM = newHalf;
    liveScale *= factor;
    // Translate so the centre stays centred during the live scale.
    const w = mapViewport.clientWidth, h = mapViewport.clientHeight;
    liveTx = (w / 2) - (w / 2 - liveTx) * factor;
    liveTy = (h / 2) - (h / 2 - liveTy) * factor;
    applyLiveTransform();
    updateZoomLabel();
    postView();
  });
});

/*  arrow-key / button pan
 * Step size scales with the current zoom (20% of halfM per press) so
 * one tap covers a similar fraction of the visible area at any zoom
 * level, not "always 30 cm" which would feel huge when zoomed-in and
 * tiny when zoomed-out. Direction convention matches the drag-pan:
 * "pan left" reveals more of the world to the left, i.e. VIEW.cx
 * decreases and the rasterised image visibly slides to the right
 * during the live transform.
 */
const PAN_STEP_FRAC = 0.20;
function panBy(direction) {
  const step = VIEW.halfM * PAN_STEP_FRAC;
  const w = mapViewport.clientWidth;
  const pxPerM = w / (2 * VIEW.halfM);
  switch (direction) {
    case "left":  VIEW.cx -= step; liveTx += step * pxPerM; break;
    case "right": VIEW.cx += step; liveTx -= step * pxPerM; break;
    case "up":    VIEW.cy += step; liveTy += step * pxPerM; break;
    case "down":  VIEW.cy -= step; liveTy -= step * pxPerM; break;
    default: return;
  }
  applyLiveTransform();
  postView();
}

document.querySelectorAll("[data-pan-action]").forEach((btn) => {
  btn.addEventListener("click", () => panBy(btn.dataset.panAction));
});

// Keyboard arrow keys also pan, as long as focus isn't inside a text
// input. `preventDefault` stops the page from scrolling.
document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
            t.isContentEditable)) return;
  switch (e.key) {
    case "ArrowLeft":  panBy("left");  e.preventDefault(); break;
    case "ArrowRight": panBy("right"); e.preventDefault(); break;
    case "ArrowUp":    panBy("up");    e.preventDefault(); break;
    case "ArrowDown":  panBy("down");  e.preventDefault(); break;
    // Plus/minus for zoom, matching the toolbar buttons.
    case "+": case "=":
      document.querySelector('[data-zoom-action="in"]').click();
      e.preventDefault(); break;
    case "-": case "_":
      document.querySelector('[data-zoom-action="out"]').click();
      e.preventDefault(); break;
    case "0":
      resetMapView(); e.preventDefault(); break;
  }
});

// Once a fresh map frame arrives (server has caught up with our
// view change), drop the live CSS transform so we're back to a 1:1
// pixel-accurate render. Without this, repeated wheel events would
// accumulate the live transform indefinitely.
mapImg.addEventListener("load", () => {
  if (liveTx !== 0 || liveTy !== 0 || liveScale !== 1) {
    clearLiveTransform();
  }
});

updateZoomLabel();

// Sync to any previously-set server view on page load. If the server
// has all-None state (fresh boot), it'll respond with nulls and we
// just stick with the JS defaults.
fetch("/view", { cache: "no-store" })
  .then(r => r.ok ? r.json() : null)
  .then(s => {
    if (s && s.half_m   !== null && s.half_m   !== undefined) VIEW.halfM = s.half_m;
    if (s && s.centre_x !== null && s.centre_x !== undefined) VIEW.cx    = s.centre_x;
    if (s && s.centre_y !== null && s.centre_y !== undefined) VIEW.cy    = s.centre_y;
    updateZoomLabel();
  })
  .catch(() => { /* ignore; defaults already loaded */ });

/*  Manual-drive joystick (teleop)
 * Drag the on-screen knob (or hold W/A/S/D) to drive the ESP's two discrete
 * axes directly, overriding the autonomous brain. The firmware is bang-bang
 * (no speed channel), so the stick sets DIRECTION only: forward/back from the
 * vertical axis, left/right (CCW/CW) from the horizontal, with a centre
 * deadzone. A diagonal push sends both axes (the proxy round-robins -> an arc).
 *
 * Wire protocol: POST /manual {fwd, turn, active}, each of fwd/turn in
 * {-1,0,1}. SAFETY: while a command is held we re-POST a heartbeat every
 * HEARTBEAT_MS; the server expires the command soon after the heartbeats stop
 * (deadman), so closing the tab or a dropped network stops the bot.
 */
const JOY        = $("joystick");
const JOY_RING   = $("joy-ring");
const JOY_KNOB   = $("joy-knob");
const JOY_RADIUS   = 36;     // px the knob travels from centre before clamping
const JOY_DEADZONE = 0.34;   // fraction of radius before an axis engages
const HEARTBEAT_MS = 150;    // re-POST cadence while a command is held

let manualFwd = 0, manualTurn = 0, manualActive = false;
let joyHeartbeat = null;
let joyDragging = false;
const joyKeys = new Set();   // currently-held drive keys among w/a/s/d

const postManual = (fwd, turn, active) => postJSON("/manual", { fwd, turn, active });

// Drive the (fwd, turn) intent. No-op when unchanged (the heartbeat keeps the
// current command alive server-side); manages the heartbeat timer + active state
// on the start/stop edges.
function applyManual(fwd, turn) {
  if (fwd === manualFwd && turn === manualTurn) return;
  manualFwd = fwd; manualTurn = turn;
  const active = fwd !== 0 || turn !== 0;
  if (active) {
    if (!manualActive) {
      manualActive = true;
      JOY.classList.add("active");
      joyHeartbeat = setInterval(
        () => postManual(manualFwd, manualTurn, true), HEARTBEAT_MS);
    }
    postManual(fwd, turn, true);
  } else if (manualActive) {
    manualActive = false;
    JOY.classList.remove("active");
    clearInterval(joyHeartbeat); joyHeartbeat = null;
    postManual(0, 0, false);
  }
}

// Knob displacement (clamped to the ring) -> quantised {fwd, turn}. Screen y
// grows downward, so up = forward; screen x right = turn right = CW = -1, which
// matches the brain's turn-sign convention (+turn = CCW/left).
function joyMove(e) {
  const rect = JOY_RING.getBoundingClientRect();
  let dx = e.clientX - (rect.left + rect.width / 2);
  let dy = e.clientY - (rect.top + rect.height / 2);
  const dist = Math.hypot(dx, dy);
  if (dist > JOY_RADIUS) { dx *= JOY_RADIUS / dist; dy *= JOY_RADIUS / dist; }
  JOY_KNOB.style.transform = `translate(${dx}px, ${dy}px)`;
  const dz = JOY_DEADZONE * JOY_RADIUS;
  const fwd  = dy < -dz ? 1 : dy > dz ? -1 : 0;
  const turn = dx >  dz ? -1 : dx < -dz ? 1 : 0;
  applyManual(fwd, turn);
}

function joyEnd(e) {
  if (!joyDragging) return;
  joyDragging = false;
  try { JOY.releasePointerCapture(e.pointerId); } catch (_) {}
  JOY_KNOB.style.transform = "";
  applyManual(0, 0);
}

JOY.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  joyDragging = true;
  JOY.setPointerCapture(e.pointerId);
  joyMove(e);
});
JOY.addEventListener("pointermove", (e) => { if (joyDragging) joyMove(e); });
JOY.addEventListener("pointerup", joyEnd);
JOY.addEventListener("pointercancel", joyEnd);

// W/A/S/D hold-to-drive, mirroring the firmware's own drive chars. Held keys
// accumulate, so W+D = forward-right arc, same as a diagonal stick push.
// Opposite keys cancel (W+S = stop). Ignored while dragging the knob or typing.
const DRIVE_KEYS = new Set(["w", "a", "s", "d"]);
function manualFromKeys() {
  const fwd  = (joyKeys.has("w") ? 1 : 0) + (joyKeys.has("s") ? -1 : 0);
  const turn = (joyKeys.has("a") ? 1 : 0) + (joyKeys.has("d") ? -1 : 0);
  applyManual(fwd, turn);
}
document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
            t.isContentEditable)) return;
  const k = e.key.toLowerCase();
  if (!DRIVE_KEYS.has(k) || joyDragging) return;
  e.preventDefault();
  if (!joyKeys.has(k)) { joyKeys.add(k); manualFromKeys(); }
});
document.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (DRIVE_KEYS.has(k) && joyKeys.delete(k)) manualFromKeys();
});

setInterval(pollState, STATE_POLL_MS);
setInterval(refreshImages, IMG_POLL_MS);
pollState();
refreshImages();
