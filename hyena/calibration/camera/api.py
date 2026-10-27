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
| `POST /api/camera/calibrate` |         | park the arm, take the frame, fit `camera.json` to it         |
| `POST /api/camera/validate`  |         | park the arm, take a frame, draw the calibration on it        |
| `GET /api/camera/feed`       |         | the live view, JPEG after JPEG                                |
| `GET /api/camera/check.jpg`  |         | the latest check image                                        |

CALIBRATE and VALIDATE park the arm first, the way RESET does, so that it is not
in the picture. CALIBRATE fits the camera to the cubes in its frame (`fit.py`)
and writes `camera.json`; the check image shows the fit, or, when there is none,
what was found instead. The red cube's points that refined the camera it replaces
are set aside. VALIDATE draws the saved calibration on a new frame, and fits that
frame too, to say how far the camera has moved since: against the cubes' own fit,
as a refinement would differ from it anyway.
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

from . import check, cubes, store
from .cubes import COLOURS, CubesNotFound
from .feed import Feed
from .fit import CameraFit, CameraNotFound, fit_camera
from .model import CUBE_MM
from .store import CubePoint

# Where the cup is lowered to over a cube, to centre the cube under it: 4 mm over its top (the owner's).
NOTCH_MM = CUBE_MM + 4.0
# VALIDATE's fit of a new frame further than this from the calibration's, anywhere in reach, says CALIBRATE again.
MOVED_MM = 2.0


class CameraSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.motion = worker.motion
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.points = store.load_points(self.config_dir)
        self._check: Tuple[Optional[int], Optional[dict]] = (None, None)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"camera": {"cube": CUBE_MM, "notch": NOTCH_MM, "colours": list(COLOURS)}})
        for name, handler in (("jump", self.jump), ("move", self.move), ("record", self.record),
                              ("forget", self.forget), ("visit", self.visit), ("calibrate", self.calibrate),
                              ("validate", self.validate)):
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
        return self.worker.run("calibrate", "CALIBRATE: parking the arm, the snapshot, the fit", self._calibrate)

    def _calibrate(self, motion: Motion) -> Union[MoveResult, str]:
        with self.lock:
            points = dict(self.points)
        missing = [colour for colour in COLOURS if points[colour] is None]
        if missing:
            raise Refused(f"record the {_listed(missing)} cube{'s' if len(missing) > 1 else ''} first")
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        try:
            fit = fit_camera(frame, {colour: (point.x, point.y) for colour, point in points.items()})
        except (CubesNotFound, CameraNotFound) as reason:
            self._save_check(check.draw_blobs(frame, cubes.find_blobs(frame), ["No calibration", str(reason)]))
            raise Refused(f"no calibration from this frame: {reason}. config/{store.CAMERA_FILE} is as it was; "
                          f"the check image shows what was found") from None
        set_aside = store.set_aside_refine_points(self.config_dir)   # first: the tests page reloads on camera.json
        store.save_frame(self.config_dir, frame)                # with the camera.json fitted to it
        store.save_camera(self.config_dir, fit)
        summary = store.describe_fit(fit)
        self._save_check(check.draw(frame, fit, ["camera.json, fitted just now", summary]))
        return (f"config/{store.CAMERA_FILE} saved from {described}: {summary}"
                + (f". The {set_aside} red cube points of the last calibration are set aside" if set_aside else ""))

    def validate(self, body: dict) -> dict:
        return self.worker.run("validate", "VALIDATE: parking the arm, then a snapshot", self._validate)

    def _validate(self, motion: Motion) -> Union[MoveResult, str]:
        saved = store.load_camera(self.config_dir)
        parked = motion.reset()
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        problem = check.frame_size_problem(frame, saved) if saved else "there is no camera.json yet: CALIBRATE"
        if problem:
            self._save_check(frame)
            raise Refused(f"the check image is the frame as taken: {problem}")
        moved = _moved(frame, saved)
        self._save_check(check.draw(frame, saved, [store.describe_calibration(self.config_dir)["saved"],
                                                   store.describe_fit(saved), moved]))
        return f"snapshot from {described}: {moved}"

    def _save_check(self, image) -> None:
        store.save_check(self.config_dir, image)

    def _check_image_described(self) -> Optional[dict]:
        """The check image, as `store.describe_check()` says, read again only when the file changes: the tests
        page's SAVE CALIBRATION draws one too."""
        path = self.config_dir / store.CHECK_FILE
        stamp = path.stat().st_mtime_ns if path.exists() else None
        if stamp != self._check[0]:
            self._check = (stamp, store.describe_check(self.config_dir))
        return self._check[1]

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
            "check": self._check_image_described(),
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


def _moved(frame, saved: CameraFit) -> str:
    """How far the camera now puts desk points from where the saved calibration's cubes did, if the cubes say."""
    points = {colour: (cube["x"], cube["y"]) for colour, cube in saved.cubes.items()}
    try:
        typical, worst = check.difference_mm(saved.cube_fit(), fit_camera(frame, points))
    except (CubesNotFound, CameraNotFound):
        return "the cubes are not where they were: see whether the outlines still sit on the desk as they should"
    return (f"the cubes, fitted again, put the desk {typical:.1f} mm from the calibration, {worst:.1f} mm at worst"
            f"{'' if worst < MOVED_MM else ': has the camera moved? CALIBRATE again'}")


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
