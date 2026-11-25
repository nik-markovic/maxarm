"""The camera calibration's requests: the arm over each cube, a point per cube, the snapshot.

The page works in desk space, through the saved `desk.json` or the library's
default desk: what the calibration maps the camera onto. Three 40 mm cubes
stand on the desk while it is used, so the arm crosses over them
(`OVER_CUBES_MM`) and comes down to the point as the library lands (`Motion.down_to()`).

| request                      | body    | does                                                          |
| ---------------------------- | ------- | ------------------------------------------------------------- |
| `POST /api/camera/move`      | x, y, z | there, landed; over the cubes first if x or y change          |
| `POST /api/camera/record`    | colour  | keep where the servos say the cup is, as that cube's point    |
| `POST /api/camera/forget`    | colour  | forget that cube's point                                      |
| `POST /api/camera/visit`     | colour  | back to that cube's point, from above                         |
| `POST /api/camera/calibrate` |         | park the arm, take the frame, fit `camera.json` to it         |
| `POST /api/camera/view`      | choice  | the live view in the camera's QUALITY or FPS mode; sets the   |
|                              |         | calibration aside                                             |
| `GET /api/camera/feed`       |         | the live view, JPEG after JPEG                                |
| `GET /api/camera/check.jpg`  |         | the latest check image                                        |
| `GET /api/camera/snapshot.png` |       | a still frame as CALIBRATE takes it, full size, lossless; the arm |
|                              |         | not parked, nothing drawn: for checks by hand                 |

CALIBRATE parks the arm first, the way RESET does, so that it is not in the
picture. CALIBRATE fits the camera to the cubes in its frame (`fit.py`),
finds the arm's base in it by its edges (`applib/camera/base.py`), lays the arm on
the camera by that base and the cubes' bearings (`anchor.py`) -- the frame SAVE
keeps -- and writes `camera.json` and the arm's correction the three cubes make,
`arm-true.json`, which the library converts with from then on; a cube the other two
whose reach is far from what the other two say is named and left out; the check
image shows the fit, or, when there is none, what was found instead, the cube that
does not fit marked in red. The points that refined the camera it replaces are set
aside, and the three cubes are the Fine Tune page's first points. What came of the
last CALIBRATE, in plain words, is the state's `camera.outcome` until the next.

QUALITY or FPS is which of the camera's modes the live view is (`devices.py`).
A calibration holds for its own mode only, so changing it sets the calibration
and everything in its pixels aside: the page warns first.
"""

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

from maxarm import MoveResult, Space, raw

from applib.camera import cubes, store
from applib.camera.base import find_base
from applib.camera.cubes import COLOURS, CubesNotFound
from applib.camera.devices import FPS, QUALITY
from applib.camera.feed import Feed
from applib.camera.fit import CameraNotFound, CubesDoNotFit, fit_camera
from applib.camera.model import CUBE_MM
from applib.camera.store import CubePoint
from applib.motion import OVER_CUBES_MM, Motion
from applib.redcube import NOTCH_MM
from applib.worker import Refused, Worker
from calibration.common.arm_api import describe
from calibration.common.server import BadRequest, Server, number

from . import anchor, check

class CameraSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.motion = worker.motion
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.points = store.load_points(self.config_dir)
        self._check: Tuple[Optional[int], Optional[dict]] = (None, None)
        self.outcome: Optional[dict] = None         # the last CALIBRATE's, for the page (`_tell()`)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"camera": {"cube": CUBE_MM, "notch": NOTCH_MM, "colours": list(COLOURS)}})
        for name, handler in (("move", self.move), ("record", self.record),
                              ("forget", self.forget), ("visit", self.visit), ("calibrate", self.calibrate),
                              ("view", self.view)):
            server.post(f"/api/camera/{name}", handler)
        server.get_stream("/api/camera/feed", self.feed.frames)
        server.get_bytes("/api/camera/check.jpg", self._check_image)
        server.get_bytes("/api/camera/snapshot.png", self._snapshot)

    # --- the arm ----------------------------------------------------------

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
        return motion.down_to(Space.DESK, target, hover=OVER_CUBES_MM)

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
        point = CubePoint(*map(_tidy, here), sent=list(map(_tidy, sent)), raw=store.raw_reading(motion.measured))
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
        """CALIBRATE, its outcome kept for the page in plain words (`outcome`), whatever it was."""
        try:
            return self._calibrated(motion)
        except Refused as reason:
            if self.outcome is None:                # not told more plainly on the way
                self._tell("fail", str(reason))
            raise
        except Exception as reason:
            self._tell("fail", f"CALIBRATE stopped: {reason}")
            raise

    def _calibrated(self, motion: Motion) -> Union[MoveResult, str]:
        self.outcome = None
        with self.lock:
            points = dict(self.points)
        missing = [colour for colour in COLOURS if points[colour] is None]
        if missing:
            raise Refused(f"record the {_listed(missing)} cube{'s' if len(missing) > 1 else ''} first")
        parked = motion.reset()                     # out of the picture
        if not parked:
            self._tell("fail", f"the arm did not park: {parked}", "RESET the arm, then CALIBRATE again.")
            return parked
        frame, described = self.feed.snapshot()
        try:
            fit = fit_camera(frame, {colour: _formula_desk(motion, point) for colour, point in points.items()})
        except CubesDoNotFit as reason:
            worst, (mm, there) = reason.worst, reason.apart[reason.worst]
            text = (f"the {worst} cube stands {mm:.0f} mm from where the arm recorded it, judged by the other two "
                    f"cubes. Nothing was saved.")
            todo = (f"Was it moved after RECORD, or was the cup off its centre? Bring the cup over the {worst} "
                    f"cube's centre, RECORD it again, then CALIBRATE. The check picture marks it in red.")
            self._save_check(check.mark(check.draw(frame, reason.fit, []), reason.fit, worst,
                                        f"{worst.upper()}: {mm:.0f} mm off where the arm recorded it", there,
                                        [f"CALIBRATE failed: the {worst} cube is {mm:.0f} mm off where the arm "
                                         f"recorded it"]))
            self._tell("fail", text, todo)
            raise Refused(f"no calibration: {text} {todo}") from None
        except (CubesNotFound, CameraNotFound) as reason:
            self._save_check(check.draw_blobs(frame, cubes.find_blobs(frame), ["CALIBRATE failed", str(reason)]))
            self._tell("fail", f"{reason}. Nothing was saved.",
                       "All three cubes must be whole and in view, nothing else of their colours near them. The "
                       "check picture outlines what was found.")
            raise Refused(f"no calibration from this frame: {reason}. config/{store.CAMERA_FILE} is as it was; "
                          f"the check image shows what was found") from None
        stamp = time.strftime("%d %b %H:%M")
        cube_points = store.cube_refine_points(fit, points, stamp)
        found = find_base(frame, fit.desk_to_pixel)
        tie = anchor.fit_tie(found, anchor.tie_points(fit, cube_points))
        anchored = anchor.anchored(fit, tie)
        set_aside = store.set_aside_refine_points(self.config_dir)   # first: the tests page reloads on camera.json
        store.save_refine_points(self.config_dir, cube_points)
        store.save_frame(self.config_dir, frame)                # with the camera.json fitted to it
        store.save_camera(self.config_dir, anchored, lens=self.feed.lens)   # the focus the frame was taken at
        store.save_correction(self.config_dir, tie.mesh(f"CALIBRATE {stamp}, the 3 cubes"))
        summary = store.describe_fit(anchored)
        image = check.draw(frame, anchored, ["camera.json, fitted just now", summary])
        colours = {point.id: point.cube for point in cube_points}
        base_line = "the arm's base found by its nuts" if found.by_nuts else \
            "the arm's base found by its whole shape only, less sure: is anything in front of its nuts?"
        if tie.misfits:
            named = [colours[point] for point in tie.misfits]
            text = "; ".join(f"the {colours[point]} cube does not fit: its reach is {miss:.0f} mm off what the "
                             f"other two say, so it is left out of the correction" for point, miss in tie.misfits.items())
            for point, miss in tie.misfits.items():
                image = check.mark(image, anchored, colours[point],
                                   f"{colours[point].upper()}: does not fit, {miss:.0f} mm, left out",
                                   anchor.to_anchored(tie.left[point]) if tie.left.get(point) is not None else None)
            self._tell("warn", f"saved, but {text}; {base_line}.",
                       f"Was the {_listed(named)} cube moved after RECORD, or the cup off its centre? RECORD it "
                       f"again, then CALIBRATE. The check picture marks it in red.")
        else:
            self._tell("ok", f"saved: the three cubes fit; {base_line}.")
        self._save_check(image)
        return (f"config/{store.CAMERA_FILE} saved from {described}: {summary}; {found.describe()}. "
                f"config/{store.CORRECTION_FILE}: the arm's correction the "
                + ("three cubes make" if not tie.misfits else "cubes that fit make")
                + (f". The {set_aside} points of the last calibration are set aside" if set_aside else "")
                + ". The three cubes are Fine Tune's points 1, 2 and 3")

    def _tell(self, status: str, text: str, todo: Optional[str] = None) -> None:
        """CALIBRATE's outcome as the page shows it until the next: 'ok', 'warn' (saved, with a problem) or 'fail'."""
        self.outcome = {"status": status, "when": _now(), "text": text[:1].upper() + text[1:], "todo": todo}

    def view(self, body: dict) -> dict:
        choice = body.get("choice")
        if choice not in (QUALITY, FPS):
            raise BadRequest(f"`choice` should be {QUALITY} or {FPS}, not {choice!r}")
        return self.worker.run("camera-view", f"the live view in its {choice.upper()} mode",
                               lambda motion: self._view(choice), uses_arm=False)

    def _view(self, choice: str) -> str:
        if choice == store.load_view_choice(self.config_dir):
            return f"the live view is already {choice.upper()}"
        with self.lock:
            set_aside = store.set_aside_calibration(self.config_dir)
            self.points = store.load_points(self.config_dir)
        store.save_view_choice(self.config_dir, choice)
        self.feed.reopen()
        return (f"the camera is now {choice.upper()}"
                + ("; the camera calibration, the cubes' points and the Fine Tune points are put aside: record the "
                   "cubes and CALIBRATE again"
                   if set_aside else ""))

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
            "outcome": self.outcome,
            "feed": self.feed.state(),
            "view": store.load_view_choice(self.config_dir),
        }}

    def _snapshot(self, query: dict) -> Optional[Tuple[bytes, str]]:
        """The frame CALIBRATE would take now -- fresh frames averaged, full size -- as PNG; None without a camera."""
        try:
            frame, _ = self.feed.snapshot()
        except Exception:                             # no camera, or no frame from it: a 404
            return None
        is_encoded, png = cv2.imencode(".png", frame)
        return (png.tobytes(), "image/png") if is_encoded else None

    def _check_image(self, query: dict) -> Optional[Tuple[bytes, str]]:
        image = store.read_check(self.config_dir)
        return (image, "image/jpeg") if image else None

    def _recorded(self, body: dict) -> Tuple[str, CubePoint]:
        colour = _colour(body)
        with self.lock:
            point = self.points[colour]
        if point is None:
            raise BadRequest(f"the {colour} cube is not recorded")
        return colour, point


def _now() -> str:
    return time.strftime("%H:%M")


def _tidy(value: float) -> float:
    """Two decimals, as the points are kept; never -0.0."""
    return round(value, 2) + 0.0


def _formula_desk(motion: Motion, point: store.CubePoint) -> Tuple[float, float]:
    """Where the arm read the cup over a cube by the formula alone (its raw reading), desk x, y: what CALIBRATE fits
    to, so that no earlier calibration's correction gets into the next."""
    if point.raw is None:
        return point.x, point.y
    x, y, _ = motion.spaces.convert(raw.formula_true(tuple(point.raw)), Space.ARM, Space.DESK)
    return x, y


def _listed(names) -> str:
    """'blue', 'blue and red', 'blue, green and red'."""
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]


def _colour(body: dict) -> str:
    colour = body.get("colour")
    if colour not in COLOURS:
        raise BadRequest(f"`colour` should be one of {', '.join(COLOURS)}, not {colour!r}")
    return colour
