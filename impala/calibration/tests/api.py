"""The tests page's requests: does the camera calibration bring the arm to what the camera sees? And, the red
cube's test in reverse, the points that make it better.

Desk space, through the saved `desk.json` or the default desk, as on the camera
page; the camera through `camera.json`. The arm's other moves are the camera
page's (`/api/camera/jump`, `/api/camera/move`): over the cubes, then down.

| request                    | body | does                                                                   |
| -------------------------- | ---- | ---------------------------------------------------------------------- |
| `POST /api/tests/find-red` |      | park the arm, find the red cube in the live view, come down over it    |
| `POST /api/tests/touch`    | u, v | the desk seen there in the live view, as shares of its width and       |
|                            |      | height: over it, then down to 2 mm over the desk                       |
| `POST /api/tests/record`   |      | keep where the servos say the cup is, over the red cube; park the arm  |
|                            |      | and see where the cube stands in the snapshot: a point                 |
| `POST /api/tests/forget`   | id   | forget a point                                                         |
| `POST /api/tests/clear`    |      | forget every point                                                     |
| `POST /api/tests/save`     |      | refine `camera.json` with the points (`camera/refine.py`)              |

The page is there only with a camera calibration, `camera.json` as the camera
page's CALIBRATE writes it; the state says why not, if not. It also carries the
calibration for the page to draw by: desk space to shares of the live view's
width and height, which hold at any size of the same picture; and the arm's x/y
error, so that the cup is drawn its true size.

A point is the red cube set down, centred under the cup -- FIND RED CUBE brings the
cup over it, the person at the arm centres it -- and seen by the camera once the
arm is out of the way. They are kept in `camera-refine-points.json` and belong
to the camera as CALIBRATE fitted it: a new CALIBRATE sets them aside.
"""

import math
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from maxarm import Camera, MoveResult, Space
from maxarm.geometry import Position

from camera import check, store
from camera.api import NOTCH_MM
from camera.cubes import CubesNotFound
from camera.feed import Feed
from camera.fit import CameraFit, CameraNotFound
from camera.locate import SAME_SHAPE, FoundCube, find_red_cube
from camera.model import CUBE_MM
from camera.store import RefinePoint
from common.motion import OVER_CUBES_MM, Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

from . import refine

# FIND RED CUBE comes over the cube 54 mm over the desk, 14 over its top, then down to the notch (the owner's).
FIND_OVER_MM = 54.0
# ARM TO DESK comes over the point clicked 14 mm over the desk, then down to 2 (the owner's, PILOT-impala).
TOUCH_OVER_MM = 14.0
TOUCH_MM = 2.0
# The cup's skirt, its outside edge: 21 mm across (the owner's). The page draws it round the point clicked.
CUP_RADIUS_MM = 10.5
# RECORD POINT: further than half a cube from where the arm says the cup was, the cube was not under the cup.
MAX_OFF_MM = CUBE_MM / 2


class TestsSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.feed = feed
        self.config_dir = Path(config_dir)
        # The red cube, as FIND RED CUBE or RECORD POINT last found it; after FIND RED CUBE, where the arm was sent over
        # it and where its servos then said it was, desk space.
        self.found: Optional[dict] = None
        self.lock = threading.Lock()
        self.points = store.load_refine_points(self.config_dir)   # read again with each new camera.json
        self._version = 0                           # counts the points' changes
        self._loaded: Tuple[Optional[int], Optional[CameraFit]] = (None, None)
        self._described: Tuple[Optional[tuple], Optional[dict]] = (None, None)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"tests": {
            "notch": NOTCH_MM, "find": {"over": FIND_OVER_MM, "down": NOTCH_MM, "cube": CUBE_MM},
            "touch": {"over": TOUCH_OVER_MM, "down": TOUCH_MM, "cup_radius": CUP_RADIUS_MM},
            "refine": {"min": refine.MIN_POINTS, "suggested": refine.SUGGESTED_POINTS, "off": refine.OFF_MM}}})
        for name, handler in (("find-red", self.find_red), ("touch", self.touch), ("record", self.record),
                              ("forget", self.forget), ("clear", self.clear), ("save", self.save)):
            server.post(f"/api/tests/{name}", handler)

    # --- the tests --------------------------------------------------------

    def find_red(self, body: dict) -> dict:
        return self.worker.run("find-red", "FIND RED CUBE: parking the arm, looking, coming down over it",
                               self._find_red)

    def _find_red(self, motion: Motion) -> MoveResult:
        fit = self._usable()
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        found, _ = self._look(fit, self._still())
        sent = (found.x, found.y, NOTCH_MM)
        result = motion.approach(Space.DESK, sent, precise=True, hover=FIND_OVER_MM)
        if not result:
            return result
        motion.read()                               # where it landed, as its servos say: the page draws the cup there
        landed = (motion.spaces.convert(motion.measured, Space.ARM, Space.DESK)
                  if motion.measured is not None else None)
        missed = ""
        if landed is not None and self.found is not None:
            self.found = {**self.found, "sent": [_tidy(value) for value in sent],
                          "landed": [_tidy(value) for value in landed]}
            missed = f"; the arm landed X {landed[0] - sent[0]:+.1f}, Y {landed[1] - sent[1]:+.1f} mm from there"
        return MoveResult(result.status, result.position,
                          f"the red cube stands at desk ({found.x:.1f}, {found.y:.1f}); the cup is "
                          f"{NOTCH_MM - CUBE_MM:.0f} mm over its top{missed}")

    def touch(self, body: dict) -> dict:
        u, v = number(body, "u", 0.0, 1.0), number(body, "v", 0.0, 1.0)
        try:
            fit = self._usable()
        except Refused as reason:
            return {"status": "refused", "detail": str(reason)}
        width, height = fit.frame_size
        camera = Camera(fit.desk_to_pixel, fit.frame_size)
        x, y, _ = camera.to_desk((u * width - 0.5, v * height - 0.5, 0.0))
        if (camera.matrix @ (x, y, 0.0, 1.0))[2] <= 0:
            return {"status": "refused", "detail": "that is not the desk: the view is above the horizon there"}
        outcome = self.worker.run(
            "tests-touch", f"ARM TO DESK: over desk ({x:.1f}, {y:.1f}), down to {TOUCH_MM:.0f} mm",
            lambda motion: self._touch(motion, (x, y, TOUCH_MM)), latest_only=True)
        return {**outcome, "point": [round(x, 2), round(y, 2)]}

    @staticmethod
    def _touch(motion: Motion, target: Position) -> MoveResult:
        """Over the point, then down to it -- or nothing, if the arm cannot reach it."""
        x, y, _ = target
        motion.use(Space.DESK)
        if not all(motion.arm.can_reach(x, y, z) for z in (TOUCH_OVER_MM, TOUCH_MM)):
            raise Refused(f"desk ({round(x)}, {round(y)}) is out of the arm's reach, {math.hypot(x, y):.0f} mm "
                          f"from its base axis")
        return motion.approach(Space.DESK, target, precise=True, hover=TOUCH_OVER_MM)

    def _still(self) -> np.ndarray:
        try:
            return self.feed.still()
        except RuntimeError as reason:
            raise Refused(str(reason)) from None

    def _snapshot(self) -> np.ndarray:
        try:
            return self.feed.snapshot()[0]
        except RuntimeError as reason:
            raise Refused(str(reason)) from None

    def _look(self, fit: CameraFit, frame: np.ndarray) -> Tuple[FoundCube, Tuple[int, int]]:
        """The red cube in `frame`, and the frame's size; kept for the page to draw."""
        try:
            found = find_red_cube(frame, fit)
        except (ValueError, CubesNotFound, CameraNotFound) as reason:
            self.found = None
            raise Refused(f"no red cube found: {reason}") from None
        size = (frame.shape[1], frame.shape[0])
        self.found = {**found.to_json(), "taken": time.strftime("%H:%M:%S"), "stamp": time.time_ns(),
                      "frame": list(size)}
        return found, size

    # --- the red cube's points --------------------------------------------

    def record(self, body: dict) -> dict:
        return self.worker.run("tests-record", "RECORD POINT: reading the arm, parking it, the snapshot of the cube",
                               self._record)

    def _record(self, motion: Motion) -> Union[MoveResult, str]:
        """Where the servos say the cup is over the cube; then, the arm out of the way, where the camera sees it."""
        fit = self._usable()
        motion.read()
        if motion.measured is None:
            raise Refused("the servos did not answer a position read; RECORD POINT again")
        here = motion.spaces.convert(motion.measured, Space.ARM, Space.DESK)
        if not CUBE_MM <= here[2] <= OVER_CUBES_MM:
            raise Refused(f"the cup is {here[2]:.0f} mm over the desk, not over a cube's top: FIND RED CUBE brings "
                          f"it down to {NOTCH_MM:.0f} over the cube; centre the cube under it, then RECORD POINT")
        sent = motion.position(Space.DESK)
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        try:
            found, size = self._look(fit, self._snapshot())
        except Refused as reason:
            raise Refused(f"the arm is parked and nothing is recorded: {reason}") from None
        off = math.dist((found.x, found.y), here[:2])
        if off > MAX_OFF_MM:
            raise Refused(f"the camera sees the red cube {off:.0f} mm from where the arm said the cup was: it was "
                          f"not under the cup. Nothing is recorded")
        with self.lock:
            point = RefinePoint(max((point.id for point in self.points), default=0) + 1, *map(_tidy, here),
                                sent=list(map(_tidy, sent)), pixel=[round(value, 3) for value in found.pixel],
                                frame_size=list(size), taken=time.strftime("%d %b %H:%M:%S"))
            self._keep(self.points + [point])
        return (f"point {point.id}: the arm said ({point.x:.1f}, {point.y:.1f}); the camera, through the saved "
                f"calibration, sees the cube {off:.1f} mm from there")

    def forget(self, body: dict) -> dict:
        point_id = int(number(body, "id"))
        with self.lock:
            kept = [point for point in self.points if point.id != point_id]
            if len(kept) == len(self.points):
                raise BadRequest(f"there is no point {point_id}")
            self._keep(kept)
        return {"status": "done", "detail": f"point {point_id} is forgotten"}

    def clear(self, body: dict) -> dict:
        with self.lock:
            count = len(self.points)
            self._keep([])
        return {"status": "done", "detail": f"{count} point{'' if count == 1 else 's'} forgotten"}

    def save(self, body: dict) -> dict:
        return self.worker.run("tests-save", "SAVE CALIBRATION: camera.json refined by the red cube's points",
                               self._save, uses_arm=False)

    def _save(self, motion: Motion) -> str:
        """camera.json refined by the points; the check image, the refined calibration on the calibration's frame."""
        fit, problem = self._calibration()
        if problem:
            raise Refused(problem)
        with self.lock:
            points = list(self.points)
        refined = refine.refined(fit, points)
        store.save_camera(self.config_dir, refined)
        typical, worst = check.difference_mm(fit, refined)
        what = (f"refined just now by {len(points)} red cube points" if points
                else "put back just now as the three cubes alone gave it")
        frame = store.load_frame(self.config_dir)
        if frame is not None and check.frame_size_problem(frame, refined) is None:
            store.save_check(self.config_dir, check.draw(frame, refined, [
                f"camera.json, {what}, on the calibration's own frame", store.describe_fit(refined)]))
        offs = (f"; the points and cubes {refined.refined['off_mm'][0]:.1f} mm from the arm's points typically, "
                f"{refined.refined['off_mm'][1]:.1f} at worst" if refined.refined else "")
        return (f"config/{store.CAMERA_FILE} {what}: it puts the desk {typical:.1f} mm from where the saved one did, "
                f"{worst:.1f} at worst{offs}")

    def _keep(self, points: List[RefinePoint]) -> None:
        """The points, now and on file. Hold the lock."""
        store.save_refine_points(self.config_dir, points)
        self.points = points
        self._version += 1

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        fit, problem = self._calibration()
        return {
            "tests": {
                "problem": problem,
                "view_problem": self._view_problem(fit) if fit else None,
                "frame_size": list(fit.frame_size) if fit else None,
                "desk_to_view": fit.desk_to_shares.tolist() if fit else None,
                "desk_to_true": fit.desk_to_true.tolist() if fit else None,
                "found": self.found,
                "refine": self._refinement(fit) if fit else None,
            },
            "unavailable": {"tests": problem} if problem else {},
        }

    def _calibration(self) -> Tuple[Optional[CameraFit], Optional[str]]:
        """The saved camera calibration, read again only when the file changes; or why there is none."""
        path = self.config_dir / store.CAMERA_FILE
        if not path.exists():
            return None, "there is no camera calibration yet: CALIBRATE on the camera page"
        stamp = path.stat().st_mtime_ns
        if stamp != self._loaded[0]:
            with self.lock:                         # CALIBRATE sets the points aside before it writes camera.json
                self.points = store.load_refine_points(self.config_dir)
                self._version += 1
            self._loaded = (stamp, store.load_camera(self.config_dir))
        fit = self._loaded[1]
        return (fit, None) if fit else (None, "camera.json was not written by the camera page: CALIBRATE again")

    def _refinement(self, fit: CameraFit) -> dict:
        """The points as `refine.describe()` has them, worked out again only when they or camera.json change."""
        with self.lock:
            key, points = (self._loaded[0], self._version), list(self.points)
        if key != self._described[0]:
            self._described = (key, refine.describe(fit, points))
        return self._described[1]

    def _usable(self) -> CameraFit:
        """The calibration, or Refused, saying why the tests cannot run."""
        fit, problem = self._calibration()
        problem = problem or self._view_problem(fit)
        if problem:
            raise Refused(problem)
        return fit

    def _view_problem(self, fit: CameraFit) -> Optional[str]:
        """Why the live view cannot be used with the calibration, if it cannot: it must be the same picture."""
        size = self.feed.state()["size"]
        width, height = fit.frame_size
        if size is None or abs(size[0] * height / (size[1] * width) - 1) <= SAME_SHAPE:
            return None
        return (f"the live view is {size[0]} x {size[1]}, the calibration {width} x {height}: not the same "
                f"picture, so the tests cannot use it")


def _tidy(value: float) -> float:
    """Two decimals, as points are kept; never -0.0."""
    return round(value, 2) + 0.0
