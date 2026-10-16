"""The desk calibration's requests: touch points, the plane through them, and trying the plane out.

Points are recorded in arm space, with no calibration in the way
(`common/motion.py`). The plane is tried out in desk space, through the saved
`desk.json`: what the library, and so every app, will use.

| request                      | body          | does                                                 |
| ---------------------------- | ------------- | ---------------------------------------------------- |
| `POST /api/desk/record`      |               | keep where the arm was sent as a touch point         |
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
desk can be watched all the way.
"""

import dataclasses
import threading
from pathlib import Path
from typing import List, Optional

from maxarm import MoveResult, MoveStatus, Space

from common.arm_api import describe
from common.motion import Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

from . import plane, store
from .store import Point

# How high over the desk a point test goes, from 0: the owner's range (PILOT-falcon).
TEST_HEIGHT_MAX_MM = 4.0
HOLD_MAX_S = 30.0


class DeskSession:
    def __init__(self, worker: Worker, config_dir: Path) -> None:
        self.worker = worker
        self.motion = worker.motion
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.points: List[Point] = store.load_points(self.config_dir)
        self.saved = store.describe_desk(self.config_dir)
        self.glide_from = -1        # `worker.count` of the point test the arm is still at; -1 for none

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"desk": {
            "x_span": plane.X_SPAN_MIN_MM, "y_span": plane.Y_SPAN_MIN_MM,
            "deviation": plane.DEVIATION_LIMITS_MM, "test_height": TEST_HEIGHT_MAX_MM}})
        for name, handler in (("record", self.record), ("delete", self.delete), ("clear", self.clear),
                              ("save", self.save), ("visit", self.visit), ("test-point", self.test_point),
                              ("test-height", self.test_height), ("test-input", self.test_input),
                              ("move", self.move)):
            server.post(f"/api/desk/{name}", handler)

    # --- the points -------------------------------------------------------

    def record(self, body: dict) -> dict:
        return self.worker.run("record", "recording a point", self._record, uses_arm=False)

    def _record(self, motion: Motion) -> str:
        if not motion.is_connected:
            raise Refused("the arm is not connected")
        with self.lock:
            point = Point(max((point.id for point in self.points), default=0) + 1,
                          *(round(value, 2) for value in motion.sent),
                          list(motion.measured) if motion.measured else None)
            self.points = self.points + [point]
            store.save_points(self.config_dir, self.points)
        return f"#{point.id} at {describe(point.target)}"

    def delete(self, body: dict) -> dict:
        gone = self._point(body)
        with self.lock:
            self.points = [point for point in self.points if point.id != gone.id]
            store.save_points(self.config_dir, self.points)
        return {"status": "done", "detail": f"#{gone.id} forgotten"}

    def clear(self, body: dict) -> dict:
        with self.lock:
            count, self.points = len(self.points), []
            store.save_points(self.config_dir, self.points)
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
        return f"desk.json saved: a plane tilted {fitted.tilt_deg:.2f} deg, through {len(points)} points"

    # --- going to them ----------------------------------------------------

    def visit(self, body: dict) -> dict:
        point = self._point(body)
        return self.worker.run("visit", f"back to #{point.id}, from above",
                               lambda motion: motion.approach(Space.ARM, point.target))

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
            result = motion.approach(Space.DESK, target)
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
        result = motion.direct(Space.DESK, (x, y, height))
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
        result = motion.approach(Space.DESK, target)
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
        return motion.approach(Space.DESK, target)

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            points = list(self.points)
        targets = [point.target for point in points]
        fitted = plane.fit(targets)
        offsets = (plane.deviations(targets, fitted) if fitted and len(points) > 3
                   else [None] * len(points))
        motion = self.motion
        is_saved = motion.spaces.is_desk_calibrated
        return {"desk": {
            "points": [{"id": point.id, "x": point.x, "y": point.y, "z": point.z,
                        "measured": point.measured, "deviation": offset}
                       for point, offset in zip(points, offsets)],
            "fit": _summary(fitted, offsets),
            "span": _span(targets),
            "save_problem": plane.problem_with(targets),
            "saved": self.saved,
            "fit_z": fitted.z_at(motion.sent[0], motion.sent[1]) if fitted else None,
            "position": list(motion.position(Space.DESK)) if is_saved else None,
            "measured": (list(motion.spaces.convert(motion.measured, Space.ARM, Space.DESK))
                         if is_saved and motion.measured else None),
            "is_gliding": self.glide_from == self.worker.count,
        }}

    def _point(self, body: dict) -> Point:
        wanted = int(number(body, "id"))
        with self.lock:
            for point in self.points:
                if point.id == wanted:
                    return point
        raise BadRequest(f"there is no point #{wanted}")

    @staticmethod
    def _require_desk(motion: Motion) -> None:
        if not motion.spaces.is_desk_calibrated:
            raise Refused("there is no desk.json yet: record the points, then SAVE")


def _summary(fitted: Optional[plane.Plane], offsets: List[Optional[float]]) -> Optional[dict]:
    if fitted is None:
        return None
    known = [offset for offset in offsets if offset is not None]
    return {"tilt_x": fitted.tilt_x, "tilt_y": fitted.tilt_y, "offset": fitted.offset,
            "tilt_deg": fitted.tilt_deg,
            "rms": (sum(offset * offset for offset in known) / len(known)) ** 0.5 if known else None,
            "worst": max((abs(offset) for offset in known), default=None)}


def _span(targets) -> Optional[dict]:
    if not targets:
        return None
    xs, ys = [target[0] for target in targets], [target[1] for target in targets]
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}
