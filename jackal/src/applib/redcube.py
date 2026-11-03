"""The red cube found in the live view and the cup brought over it: Fine Tune's FIND RED CUBE, and the app's
`find-cube`, the same way.

    cube = RedCube(feed)
    cube.find(motion, fit)          # park, look, come over the cube at 54 mm and down to the notch, 42
    cube.found                      # what the overlay draws: overlay.draw_cube()

The camera is the calibration's, held still (`camera/locate.py`). Once the arm
is down over the cube its servos are read: `found` then has where the arm was
sent, `sent`, and where its servos say it is, `landed`, desk mm. It only brings
the cup near the cube, as near as the calibration and the arm's own miss allow.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from maxarm import MoveResult, Space

from .camera.cubes import CubesNotFound
from .camera.feed import Feed
from .camera.fit import CameraFit, CameraNotFound
from .camera.locate import FoundCube, find_red_cube
from .camera.model import CUBE_MM
from .motion import Motion, Refused

# Where the cup is lowered to over a cube, to centre the cube under it: 2 mm over its top (the owner's,
# PILOT-impala). The camera page's too.
NOTCH_MM = CUBE_MM + 2.0
# The cup comes over the cube 54 mm over the desk, 14 over its top, then down to the notch (the owner's).
OVER_MM = 54.0


class RedCube:
    def __init__(self, feed: Feed) -> None:
        self.feed = feed
        self.lock = threading.Lock()
        # The red cube, as it was last found; after `find()`, where the arm was sent over it and where its servos
        # then said it was, desk space.
        self.found: Optional[dict] = None

    def find(self, motion: Motion, fit: CameraFit) -> MoveResult:
        """Park the arm out of the picture, find the red cube in the live view, come down over it."""
        parked = motion.reset()
        if not parked:
            return parked
        try:
            frame = self.feed.still()
        except RuntimeError as reason:
            raise Refused(str(reason)) from None
        found = self.look(fit, frame)
        sent = (found.x, found.y, NOTCH_MM)
        result = motion.approach(Space.DESK, sent, precise=True, hover=OVER_MM)
        if not result:
            return result
        motion.read()                               # where it landed, as its servos say: drawn there
        landed = (motion.spaces.convert(motion.measured, Space.ARM, Space.DESK)
                  if motion.measured is not None else None)
        missed = ""
        if landed is not None:
            with self.lock:
                if self.found is not None:
                    self.found = {**self.found, "sent": [_tidy(value) for value in sent],
                                  "landed": [_tidy(value) for value in landed]}
            missed = f"; the arm landed X {landed[0] - sent[0]:+.1f}, Y {landed[1] - sent[1]:+.1f} mm from there"
        return MoveResult(result.status, result.position,
                          f"the red cube stands at desk ({found.x:.1f}, {found.y:.1f}); the cup is "
                          f"{NOTCH_MM - CUBE_MM:.0f} mm over its top{missed}")

    def look(self, fit: CameraFit, frame: np.ndarray) -> FoundCube:
        """The red cube in `frame`, kept to be drawn; `Refused` if it is not there."""
        try:
            found = find_red_cube(frame, fit)
        except (ValueError, CubesNotFound, CameraNotFound) as reason:
            self.clear()
            raise Refused(f"no red cube found: {reason}") from None
        with self.lock:
            self.found = {**found.to_json(), "taken": time.strftime("%H:%M:%S"), "stamp": time.time_ns(),
                          "frame": [frame.shape[1], frame.shape[0]]}
        return found

    def clear(self) -> None:
        with self.lock:
            self.found = None


def _tidy(value: float) -> float:
    """Two decimals, as points are kept; never -0.0."""
    return round(value, 2) + 0.0
