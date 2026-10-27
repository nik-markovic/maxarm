"""The Tiles page's requests: where the 25 mm tiles are, their letters, which the arm can pick up, and a word of them.

Desk space, through the saved `desk.json` or the default desk; the camera through
`camera.json`, as on the Fine Tune page. The tiles are found in the snapshot, the
camera's full-size frame -- the one the calibration was fitted at -- with the arm
parked out of the picture first; their letters are read in the same frame
(`applib/letters/`), and they are placed in the work areas (`applib/areas.py`).

| request                       | body          | does                                                            |
| ----------------------------- | ------------- | --------------------------------------------------------------- |
| `POST /api/tiles/locate`      |               | park the arm, take the snapshot, find the tiles, read and place |
| `POST /api/tiles/visit`       | n             | the cup over tile n's centre, down to 4 mm over its top face    |
| `POST /api/tiles/place`       | letters, grip | the tiles that spell them carried to the dropoff arc, in order, |
|                               |               | each gripped with the cup at desk height `grip`, mm             |
| `GET /api/tiles/snapshot.jpg` |               | the snapshot the tiles were found in                            |

A visit tries the tile's centre: the cup comes down over it, and the person at the
arm sees whether it is centred on the tile. A tile the arm cannot reach is refused,
nothing moved.

PLACE sets a word down (`applib/word.py`), from the tiles of the last snapshot
that are ready to pick up and read; anything it cannot do is refused before the
arm moves. The tiles it carries are no longer where the snapshot has them, so
after it has moved any, it needs LOCATE TILES again, and the tiles moved cannot
be visited.
"""

import random
import threading
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import cv2

from maxarm import Camera, MaxArm, MoveResult, MoveStatus, Space

from applib import areas, word
from applib.letters import Letter, read_letters
from applib.tiles import TILE_HEIGHT_MM, TILE_MM, Tile, find_tiles
from camera import store
from camera.feed import Feed
from camera.fit import CameraFit
from camera.locate import SAME_SHAPE
from common.motion import Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

# A visit comes over the tile at the height the areas allow for, then down to 4 mm over its top face.
VISIT_OVER_MM = areas.OVER_TILE_MM
VISIT_MM = TILE_HEIGHT_MM + 4.0
# The snapshot as the page is sent it.
SNAPSHOT_QUALITY = 85
COUNTED = (areas.Status.READY, areas.Status.TOO_CLOSE, areas.Status.ELSEWHERE, areas.Status.OUT_OF_REACH)
# LOCATE TILES offers this many letters for PLACE, at random from those on tiles ready to pick up.
OFFERED_LETTERS = 5
# Where PLACE's cup grips a tile, desk mm, until the page's box says otherwise; and how far that box goes. It lets
# a tile go 1 mm under a visit's height.
GRIP_MM = 1.0
GRIP_RANGE_MM = (0.0, VISIT_MM)
DROP_MM = VISIT_MM - 1.0


class TilesSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.found: Optional[dict] = None          # the last LOCATE TILES: the tiles, placed, and when
        self.tiles: List[Tuple[Tile, areas.Place]] = []
        self.letters: List[Letter] = []            # the tiles' letters, in the same order
        self.snapshot: Optional[bytes] = None      # its frame, as a JPEG
        self.word: Optional[dict] = None           # PLACE on that snapshot: its letters, where each goes, how far
        self._loaded: Tuple[Optional[int], Optional[CameraFit]] = (None, None)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"tiles": {
            "tile": TILE_MM, "height": TILE_HEIGHT_MM, "visit": {"over": VISIT_OVER_MM, "down": VISIT_MM},
            "gap": areas.MIN_GAP_MM, "margin": areas.REACH_MARGIN_MM,
            "areas": [area.to_json() for area in (areas.PICKUP, areas.DROPOFF)],
            "word": {"arc": areas.WORD_ARC_MM, "gap": areas.WORD_GAP_MM, "most": areas.WORD_MAX_LETTERS,
                     "is_away": areas.IS_FACING_AWAY, "grip": GRIP_MM, "grip_range": list(GRIP_RANGE_MM),
                     "drop": DROP_MM}}})
        server.post("/api/tiles/locate", self.locate)
        server.post("/api/tiles/visit", self.visit)
        server.post("/api/tiles/place", self.place)
        server.get_bytes("/api/tiles/snapshot.jpg", self._snapshot)

    # --- the requests -----------------------------------------------------

    def locate(self, body: dict) -> dict:
        return self.worker.run("tiles-locate", "LOCATE TILES: parking the arm, the snapshot, finding the tiles",
                               self._locate)

    def _locate(self, motion: Motion):
        fit = self._usable()
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        camera = Camera(fit.desk_to_pixel, fit.frame_size)
        try:
            camera.check_frame(frame)
        except ValueError as reason:
            raise Refused(f"{reason}: the calibration holds for its own frame size only") from None
        started = time.monotonic()
        tiles = find_tiles(frame, camera)
        seconds = time.monotonic() - started
        started = time.monotonic()
        letters = read_letters(frame, camera, tiles)
        reading = time.monotonic() - started
        places = areas.place_tiles(tiles, _arithmetic_arm(self.config_dir).can_reach)
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])[1].tobytes()
        counts = {status.value: sum(place.status is status for place in places) for status in COUNTED}
        on_offer = [letters[index].letter for index in word.letters_on_offer(letters, places)]
        with self.lock:
            self.tiles = list(zip(tiles, places))
            self.letters = letters
            self.snapshot = jpeg
            self.word = None
            self.found = {
                "tiles": [{"n": index + 1, **tile.to_json(), **place.to_json(), **letter.to_json(),
                           "nearest": None if place.nearest is None else place.nearest + 1}
                          for index, ((tile, place), letter) in enumerate(zip(self.tiles, letters))],
                "counts": counts, "taken": time.strftime("%H:%M:%S"), "stamp": time.time_ns(),
                "frame": [frame.shape[1], frame.shape[0]], "seconds": round(seconds, 2),
                "reading": round(reading, 2), "described": described,
                "on_offer": "".join(sorted(on_offer)),
                "offer": "".join(random.sample(on_offer, min(OFFERED_LETTERS, len(on_offer)))),
            }
        return f"{describe_counts(len(tiles), counts)}{describe_letters(letters)}"

    def visit(self, body: dict) -> dict:
        n = int(number(body, "n", 1))
        with self.lock:
            if not 1 <= n <= len(self.tiles):
                raise BadRequest(f"there is no tile {n}: LOCATE TILES found {len(self.tiles)}")
            tile, place = self.tiles[n - 1]
            is_moved = n in self._moved()
        if is_moved:
            return {"status": "refused", "detail": f"PLACE has carried tile {n} off: LOCATE TILES again"}
        if place.status is areas.Status.OUT_OF_REACH:
            return {"status": "refused", "detail": f"tile {n} is out of the arm's reach: {place.detail}"}
        return self.worker.run(
            "tiles-visit", f"over tile {n}, desk ({tile.x:.1f}, {tile.y:.1f}), down to {VISIT_MM:.0f} mm",
            lambda motion: self._visit(motion, tile), latest_only=True)

    @staticmethod
    def _visit(motion: Motion, tile: Tile) -> MoveResult:
        result = motion.approach(Space.DESK, (tile.x, tile.y, VISIT_MM), precise=True, hover=VISIT_OVER_MM)
        if not result:
            return result
        return MoveResult(result.status, result.position,
                          f"the cup is {VISIT_MM - TILE_HEIGHT_MM:.0f} mm over the tile's centre, as the camera has it")

    def place(self, body: dict) -> dict:
        grip = number(body, "grip", *GRIP_RANGE_MM)
        with self.lock:
            if not self.found:
                return {"status": "refused", "detail": "LOCATE TILES first: the word is made of the tiles it finds"}
            if self._moved():
                return {"status": "refused", "detail": "PLACE has moved tiles since the snapshot: LOCATE TILES again"}
            tiles, places = [tile for tile, _ in self.tiles], [place for _, place in self.tiles]
            letters, stamp = self.letters, self.found["stamp"]
        try:
            steps = word.plan_word(str(body.get("letters", "")), tiles, letters, places,
                                   _arithmetic_arm(self.config_dir).can_reach, grip_mm=grip, drop_mm=DROP_MM)
        except word.WordRefused as reason:
            return {"status": "refused", "detail": str(reason)}
        spelled = "".join(step.letter for step in steps)
        return self.worker.run("tiles-place", f"PLACE {spelled}: each tile to the dropoff arc in turn, gripped at "
                               f"{grip:g} mm",
                               lambda motion: self._place(motion, steps, stamp))

    def _place(self, motion: Motion, steps: List[word.Step], stamp: int) -> Union[MoveResult, str]:
        spelled = "".join(step.letter for step in steps)
        with self.lock:
            if not self.found or self.found["stamp"] != stamp:
                raise Refused("LOCATE TILES has run since PLACE was asked for: PLACE again")
            self.word = {"letters": spelled, "status": "placing", "detail": "",
                         "steps": [{"n": step.index + 1, "letter": step.letter, **step.slot.to_json(),
                                    "status": "waiting", "detail": ""} for step in steps]}
        try:
            for index, step in enumerate(steps):
                if motion.stopping.is_set():
                    self._end("stopped", "RESET")
                    return MoveResult(MoveStatus.STOPPED, motion.position(Space.DESK), "RESET")
                self._mark(index, "carrying")
                result = motion.carry(step.start, step.end, step.upright, step.facing, word.CLEARANCE_MM)
                self._mark(index, "placed" if result else result.status.value, result.detail)
                if not result:
                    if result.status is not MoveStatus.STOPPED:
                        motion.reset()                  # out of the picture, letting go of anything held
                    detail = (f"letter {index + 1} of {len(steps)}, {step.letter} from tile {step.index + 1}: "
                              f"{result.detail}")
                    self._end(result.status.value, detail)
                    return MoveResult(result.status, result.position, detail)
            parked = motion.reset()
            detail = f"{spelled} set down along the dropoff arc" + ("" if parked else
                                                                    f"; the arm did not park: {parked.detail}")
            self._end("done", detail)
            return detail
        except Exception as error:
            self._end("error", str(error) or repr(error))
            raise

    def _mark(self, index: int, status: str, detail: str = "") -> None:
        with self.lock:
            if self.word:
                self.word["steps"][index].update(status=status, detail=detail)

    def _end(self, status: str, detail: str) -> None:
        with self.lock:
            if self.word:
                self.word.update(status=status, detail=detail)

    def _moved(self) -> Set[int]:
        """The tiles PLACE has started on, by number: no longer, or perhaps not, where the snapshot has them."""
        return {step["n"] for step in self.word["steps"] if step["status"] != "waiting"} if self.word else set()

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        fit, problem = self._calibration()
        with self.lock:
            found = self.found
            placing = None if self.word is None else {**self.word, "steps": [dict(step) for step in self.word["steps"]]}
        return {
            "tiles": {
                "problem": problem,
                "view_problem": self._view_problem(fit) if fit else None,
                "frame_size": list(fit.frame_size) if fit else None,
                "desk_to_view": fit.desk_to_shares.tolist() if fit else None,
                "desk_to_true": fit.desk_to_true.tolist() if fit else None,
                "found": found,
                "word": placing,
            },
            "unavailable": {"tiles": problem} if problem else {},
        }

    def _snapshot(self) -> Optional[Tuple[bytes, str]]:
        with self.lock:
            return (self.snapshot, "image/jpeg") if self.snapshot else None

    def _calibration(self) -> Tuple[Optional[CameraFit], Optional[str]]:
        """The saved camera calibration, read again only when the file changes; or why there is none."""
        path = self.config_dir / store.CAMERA_FILE
        if not path.exists():
            return None, "there is no camera calibration yet: CALIBRATE on the camera page"
        stamp = path.stat().st_mtime_ns
        if stamp != self._loaded[0]:
            self._loaded = (stamp, store.load_camera(self.config_dir))
        fit = self._loaded[1]
        return (fit, None) if fit else (None, "camera.json was not written by the camera page: CALIBRATE again")

    def _usable(self) -> CameraFit:
        fit, problem = self._calibration()
        if problem:
            raise Refused(problem)
        return fit

    def _view_problem(self, fit: CameraFit) -> Optional[str]:
        """Why the live view cannot show the tiles in place, if it cannot: it must be the calibration's picture."""
        size = self.feed.state()["size"]
        width, height = fit.frame_size
        if size is None or abs(size[0] * height / (size[1] * width) - 1) <= SAME_SHAPE:
            return None
        return (f"the live view is {size[0]} x {size[1]}, the calibration {width} x {height}: not the same "
                f"picture, so the tiles are drawn on the snapshot only")


def describe_counts(found: int, counts: dict) -> str:
    """'11 tiles: 7 ready to pick up, 2 too close, 1 outside the pickup area, 1 out of reach'."""
    if not found:
        return "no tiles found: are they in view, and the arm out of the picture?"
    words = {"ready": "ready to pick up", "too close": "too close", "elsewhere": "outside the pickup area",
             "out of reach": "out of reach"}
    parts = [f"{counts[status]} {words[status]}" for status in words if counts.get(status)]
    return f"{found} tile{'' if found == 1 else 's'}: {', '.join(parts)}"


def describe_letters(letters: List[Letter]) -> str:
    """'; letters MTANNEV?EOW, 1 not read' -- left to right, '?' for one not read, '-' for a blank tile."""
    if not letters:
        return ""
    shown = "".join(letter.letter if letter.is_read else "-" if letter.is_blank else "?" for letter in letters)
    unread = sum(not letter.is_read and not letter.is_blank for letter in letters)
    blank = sum(letter.is_blank for letter in letters)
    notes = ([f"{unread} not read"] if unread else []) + ([f"{blank} blank"] if blank else [])
    return f"; letters {shown}" + "".join(f", {note}" for note in notes)


def _arithmetic_arm(config_dir: Path) -> MaxArm:
    """A MaxArm in desk space that is never connected: `can_reach()` is arithmetic, and the port is not opened."""
    arm = MaxArm(config_dir=config_dir)
    arm.use_coordinates(Space.DESK)
    return arm
