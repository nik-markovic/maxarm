"""The desk calibration's requests: touch points, the plane through them, and trying the plane out.

Points are recorded in arm space, with no calibration in the way
(`common/motion.py`). The plane is tried out in desk space, through the saved
`desk.json`: what the library, and so every app, will use.

| request                      | body          | does                                                 |
| ---------------------------- | ------------- | ---------------------------------------------------- |
| `POST /api/desk/record`      |               | keep where the servos say the cup is: a touch point  |
| `POST /api/desk/delete`      | id            | forget one point                                     |
| `POST /api/desk/clear`       |               | forget them all                                      |
| `POST /api/desk/save`        |               | fit the plane and write `desk.json`                  |
| `POST /api/desk/visit`       | id            | back to a point in arm space, from above             |
| `POST /api/desk/test-point`  | id, height    | over a point in desk space, at a height of 0 to 4 mm |
| `POST /api/desk/test-height` | height        | the height of the point test in progress             |
| `POST /api/desk/test-input`  | x, y, z, hold | down onto a desk point, suck, hold, let go           |
| `POST /api/desk/move`        | x, y, z       | to a desk point, from above                          |

A point test comes down onto its point from above, like `grab()` -- unless the
arm is where the previous point test left it, nothing else having moved it
since. Then it glides there, slowly and level, so the cup's height over the
desk can be watched all the way. Without a `desk.json` the desk tests use the
library's default desk.

Each point's offset -- how far above the desk it is -- is given three ways, for
the page to choose from: off the plane fitted through the points (what SAVE
would write), off the saved `desk.json`, and off the default desk.
"""

import dataclasses
import threading
from pathlib import Path
from typing import Dict, List, Optional

from maxarm import MoveResult, MoveStatus, Space
from maxarm.spaces import DEFAULT_DESK_Z_MM

from common.arm_api import describe
from common.motion import Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

from . import plane, store
from .store import Point

# How high over the desk a point test goes, from 0: the owner's range (PILOT-falcon).
TEST_HEIGHT_MAX_MM = 4.0
HOLD_MAX_S = 30.0
DEFAULT_DESK = plane.Plane(0.0, 0.0, DEFAULT_DESK_Z_MM)


class DeskSession:
    def __init__(self, worker: Worker, config_dir: Path) -> None:
        self.worker = worker
        self.motion = worker.motion
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.points, self.next_id = store.load_points(self.config_dir)
        self.saved = store.describe_desk(self.config_dir)
        self.glide_from = -1        # `worker.count` of the point test the arm is still at; -1 for none
        self._fit: tuple = ((), None)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"desk": {
            "x_span": plane.X_SPAN_MIN_MM, "y_span": plane.Y_SPAN_MIN_MM,
            "deviation": plane.DEVIATION_LIMITS_MM, "test_height": TEST_HEIGHT_MAX_MM,
            "default_z": DEFAULT_DESK_Z_MM}})
        for name, handler in (("record", self.record), ("delete", self.delete), ("clear", self.clear),
                              ("save", self.save), ("visit", self.visit), ("test-point", self.test_point),
                              ("test-height", self.test_height), ("test-input", self.test_input),
                              ("move", self.move)):
            server.post(f"/api/desk/{name}", handler)

    # --- the points -------------------------------------------------------

    def record(self, body: dict) -> dict:
        return self.worker.run("record", "recording a point", self._record, uses_arm=False)

    def _record(self, motion: Motion) -> str:
        """Where the servos say the cup is, read afresh: they do not always reach where they are sent."""
        if not motion.is_connected:
            raise Refused("the arm is not connected")
        motion.read()
        if motion.measured is None:
            raise Refused("the servos did not answer a position read; RECORD again")
        with self.lock:
            point = Point(self.next_id, *(round(value, 2) for value in motion.measured),
                          sent=[round(value, 2) for value in motion.sent])
            self.points, self.next_id = self.points + [point], self.next_id + 1
            store.save_points(self.config_dir, self.points, self.next_id)
        return f"#{point.id} at {describe(point.target)}, sent to {describe(point.sent)}"

    def delete(self, body: dict) -> dict:
        gone = self._point(body)
        with self.lock:
            self.points = [point for point in self.points if point.id != gone.id]
            store.save_points(self.config_dir, self.points, self.next_id)
        return {"status": "done", "detail": f"#{gone.id} forgotten"}

    def clear(self, body: dict) -> dict:
        with self.lock:
            count, self.points, self.next_id = len(self.points), [], 1
            store.save_points(self.config_dir, self.points, self.next_id)
        return {"status": "done", "detail": f"{count} points forgotten"}

    def save(self, body: dict) -> dict:
        return self.worker.run("save", "saving desk.json", self._save, uses_arm=False)

    def _save(self, motion: Motion) -> str:
        with self.lock:
            points = list(self.points)
        targets = [point.target for point in points]
        problem = plane.problem_with(targets)
        if problem:
            raise Refused(problem)
        fitted = plane.fit(targets)
        store.save_desk(self.config_dir, fitted, points)
        motion.reload()
        self.saved = store.describe_desk(self.config_dir)
        self.glide_from = -1
        return f"desk.json saved: through {len(points)} points, the plane tilted {fitted.tilt_deg:.2f} deg"

    # --- going to them ----------------------------------------------------

    def visit(self, body: dict) -> dict:
        point = self._point(body)
        return self.worker.run("visit", f"back to #{point.id}, from above",
                               lambda motion: motion.approach(Space.ARM, point.target, precise=True))

    def test_point(self, body: dict) -> dict:
        point = self._point(body)
        height = number(body, "height", 0.0, TEST_HEIGHT_MAX_MM)
        return self.worker.run("test-point", f"testing #{point.id}, {height:.1f} mm over the desk",
                               lambda motion: self._test_point(motion, point, height))

    def _test_point(self, motion: Motion, point: Point, height: float) -> MoveResult:
        self._require_desk(motion)
        x, y, _ = motion.spaces.convert(point.target, Space.ARM, Space.DESK)
        target = (x, y, height)
        is_glide = self.glide_from == self.worker.count - 1
        result = motion.glide(Space.DESK, target) if is_glide else None
        if result is None:
            if motion.is_suction_on:
                motion.suction(False)
            result = motion.approach(Space.DESK, target, precise=True)
            if result and is_glide:
                result = dataclasses.replace(result, detail="no straight way there, so it went over")
        if result:
            self.glide_from = self.worker.count
        return result

    def test_height(self, body: dict) -> dict:
        height = number(body, "height", 0.0, TEST_HEIGHT_MAX_MM)
        return self.worker.run("test-height", f"point test height {height:.1f} mm",
                               lambda motion: self._test_height(motion, height), latest_only=True)

    def _test_height(self, motion: Motion, height: float) -> Optional[MoveResult]:
        if self.glide_from != self.worker.count - 1:
            return None                               # no point test in progress: nothing to move
        x, y, _ = motion.position(Space.DESK)
        result = motion.direct(Space.DESK, (x, y, height), precise=True)
        if result:
            self.glide_from = self.worker.count
        return result

    def test_input(self, body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        hold = number(body, "hold", 0.0, HOLD_MAX_S)
        return self.worker.run("test-input", f"suction test at desk {describe(target)}",
                               lambda motion: self._test_input(motion, target, hold))

    def _test_input(self, motion: Motion, target, hold: float) -> MoveResult:
        """Down onto the point the way grab() comes down, grip, hold, let go -- and stay there."""
        self._require_desk(motion)
        if motion.is_suction_on:
            released = motion.suction(False)
            if not released:
                return released
        result = motion.approach(Space.DESK, target, precise=True)
        if not result:
            return result
        gripped = motion.suction(True)
        if not gripped:
            return gripped
        if not motion.hold(hold):
            return MoveResult(MoveStatus.STOPPED, motion.position(Space.DESK), "RESET")
        return motion.suction(False)

    def move(self, body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        return self.worker.run("desk-move", f"to desk {describe(target)}",
                               lambda motion: self._move(motion, target), latest_only=True)

    def _move(self, motion: Motion, target) -> MoveResult:
        self._require_desk(motion)
        return motion.approach(Space.DESK, target, precise=True)

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            points = list(self.points)
        targets = [point.target for point in points]
        fitted = self._fitted(targets)
        motion, spaces = self.motion, self.motion.spaces
        nothing = [None] * len(points)
        offsets: Dict[str, List[Optional[float]]] = {
            "fit": plane.deviations(targets, fitted) if fitted and len(points) > 3 else nothing,
            "saved": ([spaces.convert(target, Space.ARM, Space.DESK)[2] for target in targets]
                      if spaces.is_desk_saved else nothing),
            "default": plane.deviations(targets, DEFAULT_DESK),
        }
        is_known = spaces.is_desk_known
        return {"desk": {
            "points": [{"id": point.id, "x": point.x, "y": point.y, "z": point.z, "sent": point.sent,
                        "off": {desk: values[index] for desk, values in offsets.items()}}
                       for index, point in enumerate(points)],
            "fit": {"tilt_deg": fitted.tilt_deg} if fitted else None,
            "off": {desk: _spread(values) for desk, values in offsets.items()},
            "span": _span(targets),
            "save_problem": plane.problem_with(targets),
            "saved": self.saved,
            "fit_height": fitted.height_of(motion.sent) if fitted else None,
            "position": list(motion.position(Space.DESK)) if is_known else None,
            "measured": (list(spaces.convert(motion.measured, Space.ARM, Space.DESK))
                         if is_known and motion.measured else None),
            "is_gliding": self.glide_from == self.worker.count,
        }}

    def _fitted(self, targets) -> Optional[plane.Plane]:
        """The desk through the points, fitted again only when they change: the page asks twice a second."""
        key = tuple(targets)
        if self._fit[0] != key:
            self._fit = (key, plane.fit(targets))
        return self._fit[1]

    def _point(self, body: dict) -> Point:
        wanted = int(number(body, "id"))
        with self.lock:
            for point in self.points:
                if point.id == wanted:
                    return point
        raise BadRequest(f"there is no point #{wanted}")

    @staticmethod
    def _require_desk(motion: Motion) -> None:
        if not motion.spaces.is_desk_known:
            raise Refused("no desk calibration is in use")


def _spread(offsets: List[Optional[float]]) -> Optional[dict]:
    known = [offset for offset in offsets if offset is not None]
    if not known:
        return None
    return {"rms": plane.rms(known), "worst": max(abs(offset) for offset in known)}


def _span(targets) -> Optional[dict]:
    if not targets:
        return None
    xs, ys = [target[0] for target in targets], [target[1] for target in targets]
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}
