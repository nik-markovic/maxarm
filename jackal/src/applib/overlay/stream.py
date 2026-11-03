"""What the app's stream shows over the camera's picture: the tiles, the red cube, and a bar saying what is going on.

    overlay = StreamOverlay(fit, size, work.scene, lambda: cube.found)
    overlay.set_doing("Placing TEAM")          # what the app is doing; "" when it waits for a command
    overlay.show_answer("TEAM set down", True) # a command's answer, for a few seconds
    overlay.burn(frame)                        # every frame of the stream, its size, on the stream's thread

The tiles and the cube are drawn as the Tiles and Fine Tune pages draw them
(`tiles.py`, `cube.py`): one overlay, here and there. Along the bottom, a bar:
what the app is doing -- placing a word, how far it has got -- or the letters on
the tiles and where the reader stands when it waits; the cloud and the stream on
the right. Over the bar, a command's
answer for a few seconds, white, or red when it was refused or failed.

The layer is drawn again only when something it shows has changed, on the
stream's thread, over the work areas drawn once; every frame only has it burnt in
(`Layer.burn()`).
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import numpy as np

from applib.camera.fit import CameraFit
from applib.redcube import NOTCH_MM

from . import inks
from .canvas import Canvas, Layer, View, ink
from .cube import draw_cube
from .tiles import draw_areas, draw_reach, draw_tiles

# Sizes in design px: a stream 900 px wide draws as a page does; a wider one, larger.
DESIGN_WIDTH_PX = 900
BAR_HEIGHT = 30.0
BAR_FONT = 14.0
ANSWER_FONT = 15.0
ANSWER_S = 6.0
# The longest answer shown, in characters: the rest is in the command's acknowledgement.
ANSWER_MAX_CHARS = 90
MUTED = ink("#c4ccd6")
GREEN = ink("#41c363")


class StreamOverlay:
    def __init__(self, fit: CameraFit, size: Tuple[int, int], scene: Callable[[], dict],
                 cube: Callable[[], Optional[dict]]) -> None:
        self.size = size
        self.scale = size[0] / DESIGN_WIDTH_PX
        self.view = View(fit.desk_to_pixel, fit.frame_size, size, fit.desk_to_true)
        self.scene = scene
        self.cube = cube
        self.is_cube_shown = False
        self.doing = ""
        self.resting = ""                         # what the bar says while nothing is being done
        self.status = {"cloud": "cloud: off", "stream": "stream: off"}
        self.answer, self.is_answer_ok, self.answer_until = "", True, 0.0
        self._layer: Optional[Layer] = None
        self._key: Optional[str] = None
        areas = Canvas(size, self.scale)
        draw_areas(areas, self.view)
        draw_reach(areas, self.view, 0.35)
        self._areas = areas.layer()             # what never changes while the app runs: most of the drawing

    def set_doing(self, doing: str) -> None:
        self.doing = doing

    def set_resting(self, text: str) -> None:
        self.resting = text

    def show_cube(self, is_shown: bool) -> None:
        self.is_cube_shown = is_shown

    def set_cloud_status(self, status: str) -> None:
        self.status = {**self.status, "cloud": status}

    def set_stream_status(self, status: str) -> None:
        self.status = {**self.status, "stream": status}

    def show_answer(self, text: str, is_ok: bool) -> None:
        """A command's answer, one line, for a few seconds; a newer one replaces it."""
        line = " ".join(text.split())
        self.answer = line if len(line) <= ANSWER_MAX_CHARS else line[:ANSWER_MAX_CHARS - 3].rstrip() + "..."
        self.is_answer_ok = is_ok
        self.answer_until = time.monotonic() + ANSWER_S

    def burn(self, frame: np.ndarray) -> None:
        """The overlay into `frame`, the stream's size, in place."""
        scene, cube = self.scene(), (self.cube() if self.is_cube_shown else None)
        answer = self.answer if time.monotonic() < self.answer_until else ""
        key = repr((_scene_key(scene), cube and (cube["stamp"], cube.get("landed")), self.doing, self.resting,
                    self.status, answer, self.is_answer_ok))
        if key != self._key:
            self._layer, self._key = self._draw(scene, cube, answer), key
        self._layer.burn(frame)

    def _draw(self, scene: dict, cube: Optional[dict], answer: str) -> Layer:
        canvas = Canvas(self.size, self.scale, under=self._areas)
        draw_tiles(canvas, self.view, scene, show_areas=False)
        if cube:
            draw_cube(canvas, self.view, cube, NOTCH_MM)
        self._draw_bar(canvas, _doing(self.doing, scene) or self.resting, bool(self.doing), answer)
        return canvas.layer()

    def _draw_bar(self, canvas: Canvas, text: str, is_doing: bool, answer: str) -> None:
        width, height = self.size
        bar = BAR_HEIGHT * self.scale
        top = height - bar
        canvas.fill([(0, top), (width, top), (width, height), (0, height)], inks.BGR["pill"], 0.72)
        middle = top + bar / 2
        margin = 10 * self.scale
        canvas.text(text, (margin, middle), GREEN if is_doing else MUTED, BAR_FONT, anchor="left")
        right = "   ".join((self.status["cloud"], self.status["stream"]))
        canvas.text(right, (width - margin - canvas.text_width(right, BAR_FONT) * self.scale, middle), MUTED,
                    BAR_FONT, anchor="left")
        if answer:
            colour = inks.BGR["white"] if self.is_answer_ok else inks.BGR["unread"]
            canvas.label(answer, (width / 2, top - 18 * self.scale), colour, ANSWER_FONT)


def _doing(doing: str, scene: dict) -> str:
    """What the app is doing; while a word is set down, how far it has got."""
    word = scene.get("word")
    if not doing or not word or word["status"] != "placing":
        return doing
    placed = sum(step["status"] == "placed" for step in word["steps"])
    carrying = next((step for step in word["steps"] if step["status"] == "carrying"), None)
    return (f"{doing}: {placed} of {len(word['steps'])} set down"
            + (f", carrying {carrying['letter']} from tile {carrying['n']}" if carrying else ""))


def _scene_key(scene: dict) -> tuple:
    found, word = scene.get("found"), scene.get("word")
    return (found and found["stamp"], word and (word["status"], tuple(step["status"] for step in word["steps"])),
            tuple(scene.get("moved") or ()))
