"""The tiles on the desk, found, read and placed, and a word of them set down: what the Tiles page and the
/IOTCONNECT app both do, the same way.

    work = TileWork(feed, CONFIG_DIR)
    work.detect(motion, fit)                       # park the arm, take the snapshot, find, read and place the tiles
    steps, stamp = work.plan("TEAM", grip_mm=1.0, reader="back")   # raises word.WordRefused, saying why
    work.place(motion, steps, stamp, fit)           # each tile carried to the dropoff arc; then found again
    work.scene()                                    # what the overlay draws: overlay.draw_tiles()
    work.summary()                                  # the letters, and which tiles are in trouble

Desk space, through the saved `desk.json` or the default desk; the camera
through `camera.json`. The tiles are found in the snapshot, the camera's
full-size frame -- the one the calibration was fitted at -- with the arm parked
out of the picture first; their letters are read in the same frame
(`letters/`), and each is placed in the work areas (`areas.py`).

A word is set down from the tiles of the last snapshot that are ready to pick up
and read (`word.py`); anything that cannot be done is refused before the arm
moves. The tiles it carries are no longer where the snapshot has them, so once
it is over -- done, refused or failed on the way, but not stopped -- the arm
parks and the tiles are found again in a new snapshot; where the tiles it carried
were is kept with them, to be drawn, word after word, until the next `detect()`:
to put them back by.

Everything is kept as the pages are sent it (`found`, `word`): plain JSON, which
the overlay draws from too. Calls come from one thread at a time, the one that
moves the arm; `scene()` and `summary()` are safe from any.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

from maxarm import Camera, MaxArm, MoveResult, MoveStatus, Space

from . import areas, word
from .camera.feed import Feed
from .camera.fit import CameraFit
from .camera.locate import SAME_SHAPE
from .letters import Letter, read_letters
from .motion import Motion, Refused
from .tiles import TILE_HEIGHT_MM, Tile, find_tiles

COUNTED = (areas.Status.READY, areas.Status.TOO_CLOSE, areas.Status.ELSEWHERE, areas.Status.OUT_OF_REACH)
# A detection offers this many letters for a word, at random from those on tiles ready to pick up.
OFFERED_LETTERS = 5
# Where the cup lets a tile go, desk mm: 2 mm over its top, where it let go while the library still sent the cup 1 mm
# under the 3 mm over asked (the owner's, PILOT-hyena), so the tile falls no further than it did.
DROP_MM = TILE_HEIGHT_MM + 2.0
# A tile's grip is not retried: the library's seal verdict holds against the desk, not for a light tile lifted into
# the cup, and a retry can shift the tile (STATUS-jackal §17). The camera tells the tiles that did not arrive.
# What is kept of a tile a word carried off, to draw where it was and how its letter lay: until the next detection.
CARRIED_KEYS = ("n", "x", "y", "corners", "letter", "is_read", "upright")


class TileWork:
    def __init__(self, feed: Feed, config_dir: Path) -> None:
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.found: Optional[dict] = None          # the last detection: the tiles, placed and read, and when
        self.tiles: List[Tuple[Tile, areas.Place]] = []
        self.letters: List[Letter] = []            # the tiles' letters, in the same order
        self.word: Optional[dict] = None           # a word on that snapshot: its letters, where each goes, how far

    # --- what the arm does --------------------------------------------------------------------------------------

    def detect(self, motion: Motion, fit: CameraFit) -> Union[MoveResult, str]:
        """Park the arm out of the picture, take the snapshot, and find, read and place the tiles in it."""
        parked = motion.reset()
        if not parked:
            return parked
        return self._find(fit)

    def plan(self, text: str, grip_mm: float, reader: str) -> Tuple[List[word.Step], int]:
        """How the word goes down, from the last detection, and that detection's stamp; raises `word.WordRefused`
        or `Refused`, saying why."""
        with self.lock:
            if not self.found:
                raise Refused("no tiles found yet: find them first, a word is made of them")
            if self.moved():
                raise Refused("tiles have been carried off since the snapshot: find them again first")
            tiles, places = [tile for tile, _ in self.tiles], [place for _, place in self.tiles]
            letters, stamp = self.letters, self.found["stamp"]
        return word.plan_word(text, tiles, letters, places, self.reach(), grip_mm=grip_mm, drop_mm=DROP_MM,
                              reader=reader), stamp

    def place(self, motion: Motion, steps: List[word.Step], stamp: int, fit: CameraFit) -> Union[MoveResult, str]:
        """Each tile carried to its place in turn; then the arm parked and the tiles found again. `stamp`: the
        detection's the steps were planned on."""
        spelled = "".join(step.letter for step in steps)
        with self.lock:
            if not self.found or self.found["stamp"] != stamp:
                raise Refused("the tiles have been found again since the word was planned on them: place it again")
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
                    detail = (f"letter {index + 1} of {len(steps)}, {step.letter} from tile {step.index + 1}: "
                              f"{result.detail}")
                    self._end(result.status.value, detail)
                    if result.status is MoveStatus.STOPPED:
                        return MoveResult(result.status, result.position, detail)
                    return MoveResult(result.status, result.position, self._find_again(motion, fit, detail))
            detail = f"{spelled} set down along the dropoff arc"
            self._end("done", detail)
            return self._find_again(motion, fit, detail)
        except Exception as error:
            self._end("error", str(error) or repr(error))
            raise

    # --- what is known ------------------------------------------------------------------------------------------

    def moved(self) -> Set[int]:
        """The tiles a word has started on, by number: no longer, or perhaps not, where the snapshot has them."""
        return {step["n"] for step in self.word["steps"] if step["status"] != "waiting"} if self.word else set()

    def scene(self) -> dict:
        """What the overlay draws: the last detection, the word on it, and the tiles it has carried off."""
        with self.lock:
            steps = [dict(step) for step in self.word["steps"]] if self.word else []
            placing = {**self.word, "steps": steps} if self.word else None
            return {"found": self.found, "word": placing, "moved": sorted(self.moved())}

    def summary(self) -> dict:
        """The tiles' letters, left to right, lower case -- '?' for one not read, '-' for a blank tile -- and those of
        the tiles out of reach and too close to another; "" for none."""
        with self.lock:
            tiles = self.found["tiles"] if self.found else []

        def letters(status: Optional[str] = None) -> str:
            return "".join(_shown(tile).lower() for tile in tiles if status is None or tile["status"] == status)

        return {"tiles": letters(), "err_no_reach": letters(areas.Status.OUT_OF_REACH.value),
                "err_too_close": letters(areas.Status.TOO_CLOSE.value)}

    def view_problem(self, fit: CameraFit) -> Optional[str]:
        """Why the live view cannot show the tiles in place, if it cannot: it must be the calibration's picture."""
        size = self.feed.state()["size"]
        width, height = fit.frame_size
        if size is None or abs(size[0] * height / (size[1] * width) - 1) <= SAME_SHAPE:
            return None
        return (f"the live view is {size[0]} x {size[1]}, the calibration {width} x {height}: not the same "
                f"picture, so the tiles cannot be drawn over it")

    def reach(self) -> Callable[[float, float, float], bool]:
        """`can_reach(x, y, z)`, desk mm, as the saved calibration has the desk now: arithmetic, a MaxArm in desk
        space that is never connected, so the port is not opened."""
        arm = MaxArm(config_dir=self.config_dir)
        arm.use_coordinates(Space.DESK)
        return arm.can_reach

    # --- inside -------------------------------------------------------------------------------------------------

    def _find(self, fit: CameraFit, carried: Optional[List[dict]] = None, placed: str = "") -> str:
        """The snapshot, the tiles in it found, read and placed; the arm out of the picture. `carried`: where the
        tiles a word carried were, and `placed`, how it went."""
        try:
            frame, described = self.feed.snapshot()
        except RuntimeError as reason:
            raise Refused(str(reason)) from None
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
        places = areas.place_tiles(tiles, self.reach())
        counts = {status.value: sum(place.status is status for place in places) for status in COUNTED}
        on_offer = [letters[index].letter for index in word.letters_on_offer(letters, places)]
        with self.lock:
            self.tiles = list(zip(tiles, places))
            self.letters = letters
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
                "carried": carried or [], "placed": placed,
            }
        return f"{describe_counts(len(tiles), counts)}{describe_letters(letters)}"

    def _find_again(self, motion: Motion, fit: CameraFit, placed: str) -> str:
        """After a word: the arm parked, letting go of anything held, and the tiles found again."""
        with self.lock:
            carried = self.found["carried"] + [{key: tile[key] for key in CARRIED_KEYS}
                                               for tile in self.found["tiles"] if tile["n"] in self.moved()]
        parked = motion.reset()                         # out of the picture
        if not parked:
            return f"{placed}; the arm did not park, so the tiles are not found again: {parked.detail}"
        return f"{placed}; found again: {self._find(fit, carried, placed)}"

    def _mark(self, index: int, status: str, detail: str = "") -> None:
        with self.lock:
            if self.word:
                self.word["steps"][index].update(status=status, detail=detail)

    def _end(self, status: str, detail: str) -> None:
        with self.lock:
            if self.word:
                self.word.update(status=status, detail=detail)


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


def _shown(tile: dict) -> str:
    """A tile's letter as it is said: the letter, '?' not read, '-' blank."""
    return tile["letter"] if tile["is_read"] else "-" if not tile["letter"] else "?"
