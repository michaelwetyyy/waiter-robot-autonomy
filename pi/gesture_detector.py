"""
MediaPipe open-hand "call waiter" gesture detector.

Detects a raised open hand (palm facing camera, all fingers extended) held
stable for >= 2 s. That's the "please come to my table" / "collect my empty
glass" signal.

This is a trimmed, self-contained copy of the original HandGestureDetector
(from the now-removed waiter-hardware tree; see git history), adapted so
rpi_ui doesn't need to import anything from outside this folder (rpi_ui gets
rsynced to the Pi on its own). The detection logic is unchanged; only the
standalone webcam test/`__main__` block was removed.

Imported by the Mac brain (mac/brain_source.py) to draw the hand skeleton on
the dashboard feed and drive the open-palm follow behaviour.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# MediaPipe is optional: it has no wheel for some platforms (e.g. Raspberry Pi
# OS Trixie / Python 3.13 on aarch64). When it's missing, the hand-skeleton
# gesture path disables itself and the rest of the dashboard — ArUco marker
# detection, the cups panel, and the order queue — keeps working on OpenCV
# alone. (Don't let one missing, hard-to-build dep take down ArUco too.)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GestureEvent:
    """Fired once when a stable open-hand gesture is confirmed.

    table_id is None when no ArUco marker is visible near the hand.
    """
    table_id: Optional[int]   # which table summoned (None if unknown)
    hand_cx: float            # hand centre x in normalised frame coords [0,1]
    hand_cy: float            # hand centre y in normalised frame coords [0,1]
    fired_at: float           # time.monotonic() when the gesture confirmed


@dataclass
class _HandTrack:
    """Internal per-hand state while we wait for the 2 s hold."""
    # Wrist position history: list of (x, y, timestamp) tuples.
    # We store timestamps so the stable check works correctly even if fps drops.
    history: list = field(default_factory=list)

    # When this hand last fired a gesture event (used for the cooldown).
    last_fired: float = 0.0

    # True once the hold threshold fired this continuous raise. Cleared on lift.
    fired_this_raise: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Gesture maths
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe landmark indices we care about.
# 0 = wrist. Then per finger: 4,8,12,16,20 are fingertips; 3,6,10,14,18 are
# the PIP joints (the middle knuckle, one below the tip).
TIPS = [4, 8, 12, 16, 20]   # thumb tip, index tip, middle tip, ring tip, pinky tip
PIPS = [3, 6, 10, 14, 18]   # the joint just below each tip


def _is_open_hand(landmarks) -> bool:
    """Return True if every finger is extended (not curled into the palm).

    Logic: for each finger, compute the 2D distance (in normalised image coords)
    from the wrist to the tip, and from the wrist to the PIP joint. If the tip
    is farther from the wrist than the PIP joint, the finger is extended.
    A closed fist has tips closer to the wrist than PIP joints. We require ALL
    five fingers extended.
    """
    wrist = landmarks.landmark[0]

    def dist_sq(i: int) -> float:
        p = landmarks.landmark[i]
        return (p.x - wrist.x) ** 2 + (p.y - wrist.y) ** 2

    return all(dist_sq(tip) > dist_sq(pip) for tip, pip in zip(TIPS, PIPS))


def _hand_centre(landmarks) -> tuple[float, float]:
    """Normalised (x, y) of the wrist landmark — used as a stable proxy for
    hand position. The wrist is more stable than the palm centre across frames."""
    w = landmarks.landmark[0]
    return w.x, w.y


# ──────────────────────────────────────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────────────────────────────────────

class HandGestureDetector:
    """Stateful open-hand gesture detector.

    Create one instance and call process() on every camera frame. It manages
    per-hand state internally across frames, so do NOT create a new instance
    per frame.

    Parameters
    ──────────
    hold_secs      : how long the open hand must be held (default 2.0 s)
    max_jitter     : max std-dev of wrist position over hold window (default 0.02,
                     in normalised coords where the whole frame is 1.0 wide)
    cooldown_secs  : after firing for a hand, ignore that hand for this long
    max_hands      : max simultaneous hands to track (default 4, covers a table of diners)
    """

    def __init__(
        self,
        hold_secs: float = 2.0,
        max_jitter: float = 0.02,
        cooldown_secs: float = 30.0,
        max_hands: int = 4,
    ):
        self.hold_secs = hold_secs
        self.max_jitter = max_jitter
        self.cooldown_secs = cooldown_secs

        # No MediaPipe -> become a no-op (ArUco/cups/queue still work).
        self.available = MEDIAPIPE_AVAILABLE
        self._tracks = {}
        self._last_detected_hands = []
        if not self.available:
            print("[gesture] MediaPipe unavailable — hand-skeleton gesture "
                  "detection OFF (ArUco detection still works).")
            return

        # Initialise the MediaPipe Hands solution.
        # static_image_mode=False  -> treat input as a video stream, so MediaPipe
        #   uses the tracker (cheaper) once a hand is detected, rather than
        #   running the detector from scratch every frame.
        # model_complexity=1       -> the FULL model. This now runs on the Mac
        #   (M4 Pro), not the Pi, so the lite model's only justification (CPU) is
        #   gone; complexity=1 acquires hands far more reliably on the dim/noisy
        #   IMX219 feed, which is the actual bottleneck here.
        # min_detection_confidence=0.3 -> lower the acquisition bar: under poor
        #   light the palm detector's score sits below the 0.5 default, so 0/N.
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
        )
        self._mp_draw = mp.solutions.drawing_utils

        # Per-hand tracks. MediaPipe doesn't give stable hand IDs across frames --
        # it just gives a list of detected hands in the current frame. We match
        # hands frame-to-frame by proximity (nearest wrist position). Dict key is
        # a small integer we assign ourselves, NOT a MediaPipe ID.
        self._tracks: dict[int, _HandTrack] = {}
        self._next_track_id = 0
        self._last_gesture_time: float = 0.0

        # Cached results from the last process() call, used by draw_debug()
        # so it doesn't need to re-run MediaPipe (which would corrupt the
        # tracker state and double the inference cost).
        self._last_results = None
        self._last_detected_hands: list = []

        # MediaPipe works best when a hand occupies a reasonable fraction of
        # the frame. Very high-res frames make distant hands tiny after
        # MediaPipe's internal resize, dropping detections. We downscale to
        # this width before inference; landmarks are returned in normalised
        # [0,1] coords so gesture logic is unaffected. Set to None to skip.
        self.inference_width: Optional[int] = 640

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process(
        self,
        rgb_frame: np.ndarray,
        visible_markers: dict | None = None,
    ) -> list[GestureEvent]:
        """Process one camera frame. Returns a (possibly empty) list of events.

        rgb_frame       : HxWx3 uint8, RGB order (not BGR -- MediaPipe wants RGB).
        visible_markers : {marker_id: image_centre (x, y)} from ArUco detection,
                           in normalised [0,1] coords. Pass None or {} if no
                           markers are visible.

        Call this on every frame even if you don't expect a gesture -- it needs
        the continuous stream to track hands and time the hold accurately.
        """
        if not self.available:
            return []
        if visible_markers is None:
            visible_markers = {}

        now = time.monotonic()

        # Downscale for inference if the frame is wider than inference_width.
        infer_frame = rgb_frame
        if self.inference_width is not None:
            h_orig, w_orig = rgb_frame.shape[:2]
            if w_orig > self.inference_width:
                scale = self.inference_width / w_orig
                new_h = int(h_orig * scale)
                infer_frame = cv2.resize(rgb_frame, (self.inference_width, new_h),
                                         interpolation=cv2.INTER_AREA)

        # Run MediaPipe on the (possibly downscaled) frame.
        infer_frame.flags.writeable = False
        results = self._mp_hands.process(infer_frame)
        infer_frame.flags.writeable = True

        # Cache results so draw_debug() can use them without re-running MediaPipe.
        self._last_results = results

        # Build a list of currently visible hand centres so we can match them
        # to our existing tracks.
        detected_hands = []
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                open_ = _is_open_hand(hand_lm)
                cx, cy = _hand_centre(hand_lm)
                detected_hands.append((hand_lm, open_, cx, cy))
        self._last_detected_hands = detected_hands

        # Match detected hands to existing tracks by nearest wrist distance.
        # Any track whose wrist matches no detection is "lost" this frame.
        matched_track_ids = set()
        for hand_lm, open_, cx, cy in detected_hands:
            track_id = self._match_or_create_track(cx, cy)
            matched_track_ids.add(track_id)
            track = self._tracks[track_id]

            if open_:
                # Hand is open this frame: append position + time to history.
                track.history.append((cx, cy, now))
                # Prune history older than hold_secs + a small buffer.
                cutoff = now - (self.hold_secs + 0.5)
                track.history = [h for h in track.history if h[2] >= cutoff]
            else:
                # Hand is closed/pointing: reset the hold timer.
                # Don't clear fired_this_raise here -- we clear it on lift (below).
                track.history.clear()

        # Any track not seen this frame: the hand was lifted. Reset its state.
        lost = set(self._tracks) - matched_track_ids
        for tid in lost:
            self._tracks[tid].history.clear()
            self._tracks[tid].fired_this_raise = False
        # Remove tracks that have been lost for a while (saves memory), but keep
        # tracks that are still in cooldown.
        for tid in lost:
            t = self._tracks[tid]
            if now - t.last_fired > self.cooldown_secs:
                del self._tracks[tid]

        # Check each active track for a confirmed gesture.
        events: list[GestureEvent] = []
        for tid, track in self._tracks.items():
            if track.fired_this_raise:
                continue  # already fired for this continuous raise
            if now - track.last_fired < self.cooldown_secs:
                continue  # in cooldown for this hand position
            if not self._is_stable(track.history, now):
                continue  # not held long enough, or too jittery

            # Gesture confirmed!
            track.fired_this_raise = True
            track.last_fired = now

            # Find the cx,cy from the most recent history entry.
            cx, cy = track.history[-1][0], track.history[-1][1]

            # Identify table: find the nearest visible ArUco marker in the frame.
            table_id = _nearest_marker(cx, cy, visible_markers)

            self._last_gesture_time = now
            events.append(GestureEvent(
                table_id=table_id,
                hand_cx=cx,
                hand_cy=cy,
                fired_at=now,
            ))

        return events

    def open_hands(self) -> list[tuple[float, float]]:
        """(cx, cy) normalised centres of every OPEN palm in the last frame
        process()'d. Empty when none are open / MediaPipe is unavailable.

        Used by the Mac server's auto-approach visual servo: it steers the bot
        toward an open palm and drives until the palm leaves the frame.
        """
        return [(cx, cy) for (_lm, open_, cx, cy)
                in getattr(self, "_last_detected_hands", []) if open_]

    def has_hands(self) -> bool:
        """True if the last process() saw any hand (open or closed). Lets a
        caller skip building a display overlay when there's nothing to draw."""
        return bool(getattr(self, "_last_detected_hands", []))

    def draw_debug(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Draw landmarks + gesture state onto a BGR frame for visualisation.

        Uses the cached results from the last process() call -- does NOT re-run
        MediaPipe. Call process() first every frame, then draw_debug().
        """
        # Nothing to draw (no hands, or MediaPipe unavailable): return the frame
        # as-is and skip the full-frame copy. Caller must not mutate the result.
        if not self.available or not self._last_detected_hands:
            return bgr_frame
        out = bgr_frame.copy()
        h, w = out.shape[:2]

        # Use cached results from the last process() call.
        detected_hands = self._last_detected_hands

        for hand_lm, open_, cx, cy in detected_hands:
            # Landmarks are in normalised coords, which map 1:1 onto the
            # display frame.
            self._mp_draw.draw_landmarks(
                out,
                hand_lm,
                mp.solutions.hands.HAND_CONNECTIONS,
            )
            px, py = int(cx * w), int(cy * h)

            # Show open/closed state above the wrist.
            label = "OPEN" if open_ else "closed"
            colour = (0, 255, 0) if open_ else (0, 100, 255)
            cv2.putText(out, label, (px - 30, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        # Bottom-of-screen timer: show the longest active hold across all open hands.
        max_held = 0.0
        for hand_lm, open_, cx, cy in detected_hands:
            if open_ and self._tracks:
                tid = self._nearest_track(cx, cy)   # read-only: never create in the draw path
                track = self._tracks.get(tid) if tid is not None else None
                if track and track.history:
                    held = track.history[-1][2] - track.history[0][2]
                    max_held = max(max_held, held)

        now2 = time.monotonic()

        if now2 - self._last_gesture_time < 3.0:
            # Show DETECTED for 3 s after a gesture fires.
            cv2.putText(out, "DETECTED", (w // 2 - 130, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        elif max_held > 0.0:
            timer_text = f"{min(max_held, self.hold_secs):.1f} / {self.hold_secs:.1f}s"
            cv2.putText(out, timer_text, (w // 2 - 110, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 3)

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _nearest_track(self, cx: float, cy: float):
        """Return the ID of the nearest existing track within the match radius
        (0.15 normalised units, ~15% of frame width), or None.

        Read-only -- does NOT create a track, so the draw path can query without
        spawning phantom tracks / leaking track IDs.
        """
        best_id, best_dist = None, float("inf")
        for tid, track in self._tracks.items():
            if not track.history:
                continue
            lx, ly, _ = track.history[-1]
            d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = tid
        return best_id if (best_id is not None and best_dist < 0.15) else None

    def _match_or_create_track(self, cx: float, cy: float) -> int:
        """Return the nearest existing track's ID, or create a new one.

        MediaPipe doesn't give stable hand IDs across frames -- the order of
        results.multi_hand_landmarks can change. We match each detected wrist to
        the nearest existing track (within 0.15 normalised units); if none
        matches, a new track is created.
        """
        tid = self._nearest_track(cx, cy)
        if tid is not None:
            return tid
        # New hand -- create a track.
        tid = self._next_track_id
        self._next_track_id += 1
        self._tracks[tid] = _HandTrack()
        return tid

    def _is_stable(self, history: list, now: float) -> bool:
        """True if the hand has been open and still for at least hold_secs.

        We look at the last hold_secs worth of history entries. If there aren't
        enough entries (hand appeared too recently), returns False. If the wrist
        spread (std dev of x and y positions) exceeds max_jitter, returns False.
        """
        if not history:
            return False

        # Grab only entries within the hold window.
        window = [h for h in history if h[2] >= now - self.hold_secs]

        # Need the window to span the full hold duration.
        if not window:
            return False
        duration = window[-1][2] - window[0][2]
        if duration < self.hold_secs:
            return False

        # Check positional stability: compute std dev of wrist x and y.
        xs = [h[0] for h in window]
        ys = [h[1] for h in window]
        spread = max(float(np.std(xs)), float(np.std(ys)))
        return spread < self.max_jitter


# ──────────────────────────────────────────────────────────────────────────────
# ArUco table identification helper
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_marker(cx: float, cy: float, visible_markers: dict) -> Optional[int]:
    """Return the ID of the ArUco marker whose image centre is nearest to (cx, cy).

    visible_markers format: { marker_id (int): (image_x, image_y) } in
    normalised [0,1] coords.

    Returns None if no markers are visible.
    """
    if not visible_markers:
        return None
    return min(
        visible_markers,
        key=lambda mid: (cx - visible_markers[mid][0]) ** 2
                       + (cy - visible_markers[mid][1]) ** 2,
    )
