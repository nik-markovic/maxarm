"""The camera calibration's requests: the arm over each cube, a point per cube, the snapshot.

The page works in desk space, through the saved `desk.json` or the library's
default desk: what the calibration maps the camera onto. Three 40 mm cubes
stand on the desk while it is used, so the arm crosses over them
(`OVER_CUBES_MM`) and comes down onto one slowly, corrected at the end.

| request                      | body    | does                                                          |
| ---------------------------- | ------- | ------------------------------------------------------------- |
| `POST /api/camera/jump`      | x, y    | over to x, y above the cubes, and stay at that height         |
| `POST /api/camera/move`      | x, y, z | there, corrected; over the cubes first if x or y change       |
| `POST /api/camera/record`    | colour  | keep where the servos say the cup is, as that cube's point    |
| `POST /api/camera/forget`    | colour  | forget that cube's point                                      |
| `POST /api/camera/visit`     | colour  | back to that cube's point, from above                         |
| `POST /api/camera/calibrate` |         | park the arm and take the frame the calibration is fitted to  |
| `POST /api/camera/check`     |         | park the arm and take a frame to check the calibration by     |
| `GET /api/camera/feed`       |         | the live view, JPEG after JPEG                                |
| `GET /api/camera/check.jpg`  |         | the latest check image                                        |

CALIBRATE and CHECK park the arm first, the way RESET does, so that it is not
in the picture. Fitting `camera.json` and drawing the cubes' outlines on the
check image are still to come: for now the check image is the frame as taken.
"""

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

from maxarm import MoveResult, Space

from common.arm_api import describe
from common.motion import OVER_CUBES_MM, Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

from . import store
from .feed import Feed
from .store import COLOURS, CubePoint

CUBE_MM = 40.0
# Where the cup is lowered to over a cube, to centre the cube under it: 4 mm over its top (the owner's).
NOTCH_MM = CUBE_MM + 4.0


class CameraSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.motion = worker.motion
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.points = store.load_points(self.config_dir)
        self.check = store.describe_check(self.config_dir)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"camera": {"cube": CUBE_MM, "notch": NOTCH_MM, "colours": list(COLOURS)}})
        for name, handler in (("jump", self.jump), ("move", self.move), ("record", self.record),
                              ("forget", self.forget), ("visit", self.visit), ("calibrate", self.calibrate),
                              ("check", self.check_calibration)):
            server.post(f"/api/camera/{name}", handler)
        server.get_stream("/api/camera/feed", self.feed.frames)
        server.get_bytes("/api/camera/check.jpg", self._check_image)

    # --- the arm ----------------------------------------------------------

    def jump(self, body: dict) -> dict:
        x, y = number(body, "x"), number(body, "y")
        return self.worker.run("camera-jump", f"over to ({x:.1f}, {y:.1f}), above the cubes",
                               lambda motion: motion.jump(Space.DESK, x, y, OVER_CUBES_MM), latest_only=True)

    def move(self, body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        return self.worker.run("camera-move", f"to desk {describe(target)}",
                               lambda motion: self._approach(motion, target), latest_only=True)

    def visit(self, body: dict) -> dict:
        colour, point = self._recorded(body)
        return self.worker.run("camera-visit", f"back over the {colour} cube",
                               lambda motion: self._approach(motion, point.target))

    @staticmethod
    def _approach(motion: Motion, target) -> MoveResult:
        return motion.approach(Space.DESK, target, precise=True, hover=OVER_CUBES_MM)

    # --- the points -------------------------------------------------------

    def record(self, body: dict) -> dict:
        colour = _colour(body)
        return self.worker.run("record", f"recording the {colour} cube",
                               lambda motion: self._record(motion, colour), uses_arm=False)

    def _record(self, motion: Motion, colour: str) -> str:
        """Where the servos say the cup is, read afresh: they do not always reach where they are sent."""
        if not motion.is_connected:
            raise Refused("the arm is not connected")
        motion.read()
        if motion.measured is None:
            raise Refused("the servos did not answer a position read; RECORD again")
        here = motion.spaces.convert(motion.measured, Space.ARM, Space.DESK)
        sent = motion.position(Space.DESK)
        point = CubePoint(*map(_tidy, here), sent=list(map(_tidy, sent)))
        with self.lock:
            self.points = {**self.points, colour: point}
            store.save_points(self.config_dir, self.points)
        return f"{colour} at desk {describe(point.target)}"

    def forget(self, body: dict) -> dict:
        colour, _ = self._recorded(body)
        with self.lock:
            self.points = {**self.points, colour: None}
            store.save_points(self.config_dir, self.points)
        return {"status": "done", "detail": f"the {colour} cube's point is forgotten"}

    # --- the snapshot -----------------------------------------------------

    def calibrate(self, body: dict) -> dict:
        return self.worker.run("calibrate", "CALIBRATE: parking the arm, then the snapshot", self._calibrate)

    def _calibrate(self, motion: Motion) -> Union[MoveResult, str]:
        with self.lock:
            missing = [colour for colour in COLOURS if self.points[colour] is None]
        if missing:
            raise Refused(f"record the {_listed(missing)} cube{'s' if len(missing) > 1 else ''} first")
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        store.save_frame(self.config_dir, frame)
        self.check = store.save_check(self.config_dir, frame)
        return (f"snapshot from {described}, kept as config/{store.FRAME_FILE}. Fitting config/{store.CAMERA_FILE} "
                f"to it is not built yet")

    def check_calibration(self, body: dict) -> dict:
        return self.worker.run("check", "CHECK: parking the arm, then a snapshot", self._check)

    def _check(self, motion: Motion) -> Union[MoveResult, str]:
        parked = motion.reset()
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        self.check = store.save_check(self.config_dir, frame)
        return f"snapshot from {described}. Drawing the cubes' outlines on it comes with the fit"

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        motion = self.motion
        with self.lock:
            points = dict(self.points)
        measured = motion.measured
        return {"camera": {
            "cubes": {colour: (asdict(point) if point else None) for colour, point in points.items()},
            "position": list(motion.position(Space.DESK)),
            "measured": list(motion.spaces.convert(measured, Space.ARM, Space.DESK)) if measured else None,
            "check": self.check,
            "calibration": store.describe_calibration(self.config_dir),
            "feed": self.feed.state(),
        }}

    def _check_image(self) -> Optional[Tuple[bytes, str]]:
        image = store.read_check(self.config_dir)
        return (image, "image/jpeg") if image else None

    def _recorded(self, body: dict) -> Tuple[str, CubePoint]:
        colour = _colour(body)
        with self.lock:
            point = self.points[colour]
        if point is None:
            raise BadRequest(f"the {colour} cube is not recorded")
        return colour, point


def _tidy(value: float) -> float:
    """Two decimals, as the points are kept; never -0.0."""
    return round(value, 2) + 0.0


def _listed(names) -> str:
    """'blue', 'blue and red', 'blue, green and red'."""
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]


def _colour(body: dict) -> str:
    colour = body.get("colour")
    if colour not in COLOURS:
        raise BadRequest(f"`colour` should be one of {', '.join(COLOURS)}, not {colour!r}")
    return colour
