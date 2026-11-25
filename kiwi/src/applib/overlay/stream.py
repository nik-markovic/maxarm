"""What the app's stream shows over the camera's picture: the tiles, the red cube, a bar saying what is going on, and
each command's answer.

    overlay = StreamOverlay(camera_file, size, work.scene, lambda: cube.found)
    overlay.set_doing("Placing TEAM")          # what a command has the arm do; "" when it is done
    overlay.watch_arm(app.arm_state)           # what the arm's worker does, for a page too, and why it cannot
    overlay.set_cloud_status("cloud: online")  # /IOTCONNECT's connection, shown only when it is not online
    overlay.watch_agent(agent.activity)        # the language model, if there is one
    overlay.show_answer("TEAM set down", True) # a command's answer, for a while
    layer = overlay.current()                  # the layer for this frame of the stream, on GStreamer's thread
    overlay.burn(frame)                        # or into a frame of the stream's size, for a picture

The tiles and the cube are drawn as the Tiles and Fine Tune pages draw them
(`tiles.py`, `cube.py`): one overlay, here and there, by the calibration
`camera.json` has now -- a new CALIBRATE or SAVE is drawn by from its next frame;
without one, the bar alone. Along the bottom, a bar. Left, the arm: what it is
doing, for a command or a page, and for how long, green; why it cannot be used,
red; or ready, with the tiles last seen and when. Right, only what is worth
seeing on a stream someone is already watching: the cloud when it is not
online -- commands cannot come -- and the language model on the Ara240, if
there is one: loading, not running, ready. While the model answers a request, a
line over the bar: the request, what the model is doing, how long it has taken.

Over the bar, the latest answer, yellow, or red when the command was refused or
failed: up to three lines, for longer the longer it is, a newer one in its place.

The overlay is drawn in OpenCV's Hershey font, which has ASCII alone: other
letters are put the nearest ASCII way. The layer is drawn again only when
something it shows has changed, on the stream's thread, over the work areas drawn
once; GStreamer lays it over every frame (`stream.py`).
"""

from __future__ import annotations

import time
import unicodedata
from typing import Callable, List, Optional, Tuple

import numpy as np

from applib.agent.connector import FAILED, LOADING, READY, STOPPED, WAITING
from applib.camera.fit import CameraFit
from applib.camera.store import CameraFile
from applib.redcube import NOTCH_MM

from . import inks
from .canvas import Canvas, Layer, View, ink
from .cube import draw_cube
from .tiles import draw_areas, draw_reach, draw_tiles

# Sizes in design px: a stream 900 px wide draws as a page does; a wider one, larger.
DESIGN_WIDTH_PX = 900
ROW_HEIGHT = 26.0
BAR_FONT = 14.0
MARGIN = 10.0
GAP = 24.0                  # between the bar's pieces
ANSWER_FONT = 15.0
ANSWER_WIDTH = 760.0        # an answer's lines at most this wide
ANSWER_LINES = 3            # the rest is in the command's acknowledgment
ANSWER_S = 12.0             # an answer of one line stays this long,
ANSWER_LINE_S = 4.0         # and this much longer for every line more
REQUEST_CHARS = 48          # a request to the model, shown on the bar at most this long

MUTED = ink("#c4ccd6")
GREEN = ink(inks.GREEN)
YELLOW = ink(inks.YELLOW)
ORANGE = ink(inks.REACH_MIN)
RED = ink(inks.UNREAD)
AGENT = ink("#6fd3ff")      # the language model's: a color of its own, none of the tiles' or the areas'

# Beyond the font's ASCII: what the model and the app write that NFKD does not take apart.
ASCII = str.maketrans({"·": "-", "—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
                       "×": "x", "°": " deg"})


class StreamOverlay:
    def __init__(self, camera_file: CameraFile, size: Tuple[int, int], scene: Callable[[], dict],
                 cube: Callable[[], Optional[dict]]) -> None:
        self.size = size
        self.scale = size[0] / DESIGN_WIDTH_PX
        self.camera_file = camera_file
        self.fit: Optional[CameraFit] = None    # what `view` and the areas were drawn by
        self.view: Optional[View] = None
        self.scene = scene
        self.cube = cube
        self.agent: Callable[[], Optional[dict]] = lambda: None
        self.arm: Callable[[], Tuple[str, str]] = lambda: ("", "")
        self.is_cube_shown = False
        self.doing, self.doing_since = "", 0.0
        self.timed = ""                         # the doing `doing_since` times
        self.cloud = ""
        self.answer: Tuple[str, ...] = ()       # the latest answer's lines
        self.is_answer_ok, self.answer_until = True, 0.0
        self._measure = Canvas((1, 1), self.scale)      # for how wide text is drawn
        self._layer: Optional[Layer] = None
        self._key: Optional[str] = None
        self._areas: Optional[Layer] = None     # what changes only with the calibration: most of the drawing

    def set_doing(self, doing: str) -> None:
        """What a command has the arm do: shown in place of the worker's own label for it."""
        self.doing = doing

    def watch_arm(self, state: Callable[[], Tuple[str, str]]) -> None:
        """What the arm's worker is doing -- "" when nothing -- and why the arm cannot be used, "" once it can."""
        self.arm = state

    def set_cloud_status(self, status: str) -> None:
        """`iotc.py`'s 'cloud: connecting', 'online', 'reconnecting', 'offline'."""
        state = status.split(":")[-1].strip()
        self.cloud = "" if state == "online" else f"Cloud {state}: no commands come"

    def watch_agent(self, activity: Callable[[], Optional[dict]]) -> None:
        """The language model's `activity()`: its state, the request it answers, the word set down."""
        self.agent = activity

    def show_cube(self, is_shown: bool) -> None:
        self.is_cube_shown = is_shown

    def show_answer(self, text: str, is_ok: bool) -> None:
        """A command's answer, in place of the one before."""
        lines = _wrapped(self._measure, _ascii(text))
        self.answer, self.is_answer_ok = tuple(lines), is_ok
        self.answer_until = time.monotonic() + ANSWER_S + ANSWER_LINE_S * (len(lines) - 1)

    def burn(self, frame: np.ndarray) -> None:
        """The overlay into `frame`, the stream's size, in place."""
        self.current().burn(frame)

    def current(self) -> Layer:
        """The layer as it should look now: the same object until something it shows changes."""
        self._follow_calibration()
        scene, cube = self.scene(), (self.cube() if self.is_cube_shown else None)
        answer = self.answer if time.monotonic() < self.answer_until else ()
        rows = self._rows(scene)
        key = repr((id(self.fit), _scene_key(scene), cube and (cube["stamp"], cube.get("landed")), rows, answer, self.is_answer_ok))
        if key != self._key:
            self._layer, self._key = self._draw(scene, cube, rows, answer), key
        return self._layer

    # --- what the bar says ------------------------------------------------------------------------------------

    def _rows(self, scene: dict) -> List[Tuple[List[Tuple[str, tuple]], List[Tuple[str, tuple]]]]:
        """The bar's rows, top first: each its left and its right pieces, (text, color)."""
        activity = self.agent()
        thinking = activity and activity["thinking"]
        rows = []
        if thinking:
            request = _ascii(" ".join(thinking["text"].split()))
            if len(request) > REQUEST_CHARS:
                request = request[:REQUEST_CHARS - 3].rstrip() + "..."
            color = ORANGE if thinking["doing"] == "Stopping" else AGENT
            rows.append(([(f'Ara240: "{request}"   {_ascii(thinking["doing"])}   {thinking["seconds"]} s', color)],
                         []))
        right = [(self.cloud, ORANGE)] if self.cloud else []
        if activity and not thinking:
            right += _model_state(activity["model"])
        rows.append(([self._arm(scene)], right))
        return rows

    def _arm(self, scene: dict) -> Tuple[str, tuple]:
        working, problem = self.arm()
        doing = (self.doing or working) if working else ""
        if doing and not self.timed:
            self.doing_since = time.monotonic()
        self.timed = doing
        if doing:
            seconds = int(time.monotonic() - self.doing_since)
            return f"{_ascii(_doing(doing, scene))}   {seconds // 60}:{seconds % 60:02d}", GREEN
        if problem == "connecting":
            return "Arm connecting: the board resets and homes it", YELLOW
        if problem:
            return _ascii(f"Arm not connected: {problem}"), RED
        found = scene.get("found")
        if not found:
            return "Ready for a command   no tiles seen yet", MUTED
        return f"Ready for a command   {len(found['tiles'])} tiles, seen {_ago(found['stamp'])}", MUTED

    # --- drawing ----------------------------------------------------------------------------------------------

    def _follow_calibration(self) -> None:
        """The view and the areas drawn again when camera.json has changed; none without one."""
        fit, _ = self.camera_file.get()
        if fit is self.fit:
            return
        self.fit = fit
        self.view = View(fit.desk_to_pixel, fit.frame_size, self.size, fit.desk_to_true) if fit else None
        self._areas = None
        if self.view is not None:
            areas = Canvas(self.size, self.scale)
            draw_areas(areas, self.view)
            draw_reach(areas, self.view, 0.35)
            self._areas = areas.layer()

    def _draw(self, scene: dict, cube: Optional[dict], rows, answer: Tuple[str, ...]) -> Layer:
        canvas = Canvas(self.size, self.scale, under=self._areas)
        if self.view is not None:
            draw_tiles(canvas, self.view, scene, show_areas=False)
            if cube:
                draw_cube(canvas, self.view, cube, NOTCH_MM)
        top = self._draw_bar(canvas, rows)
        if answer:
            self._draw_answer(canvas, answer, top)
        return canvas.layer()

    def _draw_bar(self, canvas: Canvas, rows) -> float:
        """The bar, its rows bottom up; where its top is."""
        width, height = self.size
        top = height - ROW_HEIGHT * self.scale * len(rows)
        canvas.fill([(0, top), (width, top), (width, height), (0, height)], inks.BGR["pill"], 0.72)
        for index, (left, right) in enumerate(rows):
            middle = top + (index + 0.5) * ROW_HEIGHT * self.scale
            end = DESIGN_WIDTH_PX - MARGIN
            for text, color in reversed(right):
                end -= canvas.text_width(text, BAR_FONT)
                canvas.text(text, (end * self.scale, middle), color, BAR_FONT, anchor="left")
                end -= GAP
            start = MARGIN
            for text, color in left:
                text = _fitted(canvas, text, end - start)
                canvas.text(text, (start * self.scale, middle), color, BAR_FONT, anchor="left")
                start += canvas.text_width(text, BAR_FONT) + GAP
        return top

    def _draw_answer(self, canvas: Canvas, lines: Tuple[str, ...], top: float) -> None:
        height = (22 + 1.5 * ANSWER_FONT * (len(lines) - 1)) * self.scale
        color = YELLOW if self.is_answer_ok else RED
        canvas.label("\n".join(lines), (self.size[0] / 2, top - 8 * self.scale - height / 2), color, ANSWER_FONT)


def _model_state(status: dict) -> List[Tuple[str, tuple]]:
    """The model's state, when there is one to say: nothing while it is looked for or where it is not installed."""
    state = status["state"]
    if state == READY:
        return [("Ara240 ready", MUTED)]
    if state == LOADING:
        left = max(0, int(status["typical_s"] - (time.time() - status["since"]))) if status["since"] else 0
        return [(f"Ara240 loading the model, ~{left // 60 + 1} min" if left else "Ara240 loading, nearly done",
                 YELLOW)]
    if state == WAITING:
        return [("Ara240 waiting for its service", YELLOW)]
    if state == STOPPED:
        return [("Ara240 model not running", ORANGE)]
    if state == FAILED:
        return [("Ara240 not answering", RED)]
    return []


def _doing(doing: str, scene: dict) -> str:
    """What the arm is doing; while a word is set down, how far it has got."""
    word = scene.get("word")
    if not word or word["status"] != "placing":
        return doing
    placed = sum(step["status"] == "placed" for step in word["steps"])
    carrying = next((step for step in word["steps"] if step["status"] == "carrying"), None)
    return (f"{doing}: {placed} of {len(word['steps'])} set down"
            + (f", carrying {carrying['letter']} from tile {carrying['n']}" if carrying else ""))


def _ago(stamp_ns: int) -> str:
    seconds = time.time() - stamp_ns / 1e9
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    return f"{int(seconds // 3600)} h ago"


def _ascii(text: str) -> str:
    """`text` as the Hershey font can draw it."""
    text = unicodedata.normalize("NFKD", text.translate(ASCII))
    return text.encode("ascii", "ignore").decode("ascii")


def _fitted(canvas: Canvas, text: str, width: float) -> str:
    """`text`, cut short with '...' if it is wider than `width` design px."""
    if canvas.text_width(text, BAR_FONT) <= width:
        return text
    while text and canvas.text_width(text + "...", BAR_FONT) > width:
        text = text[:-1]
    return text.rstrip() + "..."


def _wrapped(canvas: Canvas, text: str) -> List[str]:
    """`text` in lines of at most ANSWER_WIDTH, broken between words; ANSWER_LINES at most, the last cut short."""
    lines: List[str] = []
    for word in text.split():
        if lines and canvas.text_width(f"{lines[-1]} {word}", ANSWER_FONT) <= ANSWER_WIDTH:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    if len(lines) > ANSWER_LINES:
        lines = lines[:ANSWER_LINES]
        while lines[-1] and canvas.text_width(lines[-1] + "...", ANSWER_FONT) > ANSWER_WIDTH:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def _scene_key(scene: dict) -> tuple:
    found, word = scene.get("found"), scene.get("word")
    return (found and found["stamp"], word and (word["status"], tuple(step["status"] for step in word["steps"])),
            tuple(scene.get("moved") or ()))
