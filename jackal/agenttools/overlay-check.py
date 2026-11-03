#!/usr/bin/env python3
"""The overlay drawn on the owner's stills, as the Tiles and Fine Tune pages and the stream show it: to look at.

    ../.venv/bin/python agenttools/overlay-check.py        # writes agenttools/overlay-check/*.jpg

- tiles-found.jpg      still-1's tiles as found, at the stream's size
- tiles-unread.jpg     the same, tile 4's letter taken as not read, tile 9 as blank
- tiles-placing.jpg    TEAM planned on them: T set down, E being carried, A and M to come
- readers.jpg          where TEAM's letters go for each reader, back, front, left and right
- stream.jpg           the stream's own overlay, as `StreamOverlay` burns it into a frame: TEAM being set down,
                       the bar, an answer over it
- cube.jpg             the red cube in config/camera-frame.png, found, the arm "landed" 1.2 mm off it

Nothing moves and nothing is written but these pictures. The tiles are found
and read for real, by `applib/tilework.py`, on a still that stands in for the
camera; the word's progress and the arm's landing are made up, to be drawn.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from maxarm.spaces import CONFIG_DIR                                 # noqa: E402

from applib.camera import store                                      # noqa: E402
from applib.camera.fit import CameraFit                              # noqa: E402
from applib.overlay import Canvas, View, draw_cube, draw_tiles       # noqa: E402
from applib.overlay.stream import StreamOverlay                      # noqa: E402
from applib.redcube import NOTCH_MM, RedCube                         # noqa: E402
from applib.stream import stream_size                                # noqa: E402
from applib.tilework import TileWork                                 # noqa: E402

OUT = HERE / "overlay-check"
STILL = HERE / "tile-shots" / "still-1.png"
STILL_CAMERA = HERE / "tile-shots" / "still-camera.json"


class StillFeed:
    """The camera, as a still frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def snapshot(self):
        return self.frame, "a still"

    def still(self) -> np.ndarray:
        return self.frame

    def state(self) -> dict:
        return {"size": [self.frame.shape[1], self.frame.shape[0]]}


def drawn(frame: np.ndarray, fit: CameraFit, draw, caption: str) -> np.ndarray:
    """`draw(canvas, view)` burnt into `frame` at the stream's size, a caption over it."""
    size = stream_size(fit.frame_size)
    picture = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    canvas = Canvas(size, size[0] / 900)
    draw(canvas, View(fit.desk_to_pixel, fit.frame_size, size, fit.desk_to_true))
    canvas.layer().burn(picture)
    cv2.rectangle(picture, (0, 0), (size[0], 30), (20, 20, 20), -1)
    cv2.putText(picture, caption, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return picture


def placing(work: TileWork, letters: str, reader: str, statuses) -> dict:
    """The scene with a word planned on the tiles, each letter's tile as `statuses` says, as `place()` keeps it."""
    steps, _ = work.plan(letters, 1.0, reader)
    word = {"letters": letters, "status": "placing", "detail": "",
            "steps": [{"n": step.index + 1, "letter": step.letter, **step.slot.to_json(), "status": status,
                       "detail": ""} for step, status in zip(steps, statuses)]}
    moved = [step["n"] for step in word["steps"] if step["status"] != "waiting"]
    return {"found": work.found, "word": word, "moved": moved}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    frame = cv2.imread(str(STILL))
    fit = CameraFit.from_json(json.loads(STILL_CAMERA.read_text()))
    work = TileWork(StillFeed(frame), CONFIG_DIR)
    print(work._find(fit))
    scene = work.scene()

    def write(name: str, picture: np.ndarray) -> None:
        cv2.imwrite(str(OUT / name), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {(OUT / name).relative_to(HERE.parent)}")

    write("tiles-found.jpg", drawn(frame, fit, lambda canvas, view: draw_tiles(canvas, view, scene),
                                   "still-1: the tiles as found"))
    tiles = [dict(tile) for tile in scene["found"]["tiles"]]
    tiles[3]["is_read"] = False
    tiles[8].update(letter="", is_read=False)
    unread = {**scene, "found": {**scene["found"], "tiles": tiles}}
    write("tiles-unread.jpg", drawn(frame, fit, lambda canvas, view: draw_tiles(canvas, view, unread),
                                    "tile 4 not read (red, ?), tile 9 blank (-)"))
    progress = placing(work, "TEAM", "back", ["placed", "carrying", "waiting", "waiting"])
    write("tiles-placing.jpg", drawn(frame, fit, lambda canvas, view: draw_tiles(canvas, view, progress),
                                     "TEAM: T set down, E being carried, A and M to come; reader back"))
    panels = []
    for reader in ("back", "front", "left", "right"):
        slots = placing(work, "TEAM", reader, ["waiting"] * 4)
        panels.append(drawn(frame, fit, lambda canvas, view: draw_tiles(canvas, view, slots),
                            f"TEAM for a reader {reader}: each letter's top, and the order along the arc"))
    write("readers.jpg", np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))

    size = stream_size(fit.frame_size)
    overlay = StreamOverlay(fit, size, lambda: progress, lambda: None)
    overlay.set_doing("Placing TEAM")
    overlay.set_cloud_status("cloud: online")
    overlay.set_stream_status("stream: 1 watching")
    overlay.show_answer("11 tiles: 9 ready to pick up, 2 too close; letters MTANNEVTEOW.", True)
    picture = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    overlay.burn(picture)
    write("stream.jpg", picture)

    cubes_frame = store.load_frame(CONFIG_DIR)
    cubes_fit = store.load_camera(CONFIG_DIR)
    if cubes_frame is None or cubes_fit is None:
        print("no config/camera-frame.png or camera.json: no cube.jpg")
        return 0
    cube = RedCube(StillFeed(cubes_frame))
    found = cube.look(cubes_fit, cubes_frame)
    landed = {**cube.found, "sent": [found.x, found.y, NOTCH_MM], "landed": [found.x + 1.2, found.y - 0.4, NOTCH_MM]}
    write("cube.jpg", drawn(cubes_frame, cubes_fit, lambda canvas, view: draw_cube(canvas, view, landed, NOTCH_MM),
                            f"the red cube in camera-frame.png at ({found.x:.1f}, {found.y:.1f}); 'landed' made up"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
