#!/usr/bin/env python3
"""The tile finder and the letters on a still, drawn: no arm, no camera.

    ../.venv/bin/python agenttools/try-tiles.py agenttools/tile-shots/still-1.png \
        agenttools/tile-shots/still-camera.json
    ../.venv/bin/python agenttools/try-tiles.py FRAME.png                  # config/camera.json

Prints each tile found -- desk x, y, turn, how far its weakest side stands out, its letter, how sure, where its top
points, the dot -- and writes `<frame>-tiles.jpg` beside the frame: each tile's top face outlined where the finder put
it, its number and letter over it, with a legend.
"""

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maxarm import Camera                                     # noqa: E402
from applib.letters import read_letters                       # noqa: E402
from applib.tiles import TILE_HEIGHT_MM, find_tiles           # noqa: E402

INK = (255, 0, 255)


def main(argv) -> int:
    frame_path = Path(argv[1])
    camera = Camera.load(argv[2] if len(argv) > 2 else ROOT / "config" / "camera.json")
    frame = cv2.imread(str(frame_path))
    started = time.monotonic()
    tiles = find_tiles(frame, camera)
    print(f"{len(tiles)} tiles in {time.monotonic() - started:.2f} s")
    started = time.monotonic()
    letters = read_letters(frame, camera, tiles)
    print(f"read in {time.monotonic() - started:.2f} s")
    drawn = frame.copy()
    for number, (tile, letter) in enumerate(zip(tiles, letters), 1):
        shown = letter.letter if letter.is_read else "?" if letter.letter else "-"
        reading = (f"{shown} {letter.confidence:4.2f} (best {letter.letter or 'blank'}, next {letter.runner_up} "
                   f"{letter.runner_up_confidence:4.2f}) up {letter.upright:6.1f}{' dot' if letter.has_dot else ''}"
                   if letter.letter else f"blank, ink {letter.ink_contrast:.2f}")
        print(f"  {number:2d}  desk ({tile.x:7.1f}, {tile.y:7.1f})  turn {tile.turn:6.1f}  "
              f"{math.hypot(tile.x, tile.y):5.1f} mm out  weakest side {tile.contrast:5.1f} x what lies beyond it  "
              f"{reading}")
        face = np.array([camera.from_desk((x, y, TILE_HEIGHT_MM))[:2] for x, y in tile.corners])
        cv2.polylines(drawn, [np.round(face).astype(np.int32)], True, INK, 1, cv2.LINE_AA)
        cv2.circle(drawn, (round(tile.pixel[0]), round(tile.pixel[1])), 4, INK, -1, cv2.LINE_AA)
        top = face[np.argmin(face[:, 1])]
        for ink, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(drawn, str(number), (round(top[0]) - 8, round(top[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        ink, width, cv2.LINE_AA)
        if letter.is_read:
            # As the Tiles page puts its badge: the corner 135 degrees on from where the letter's top points.
            toward = math.radians(letter.upright + 135.0)
            x, y = max(tile.corners, key=lambda corner: (corner[0] - tile.x) * math.cos(toward)
                       + (corner[1] - tile.y) * math.sin(toward))
            at = np.round(camera.from_desk((x, y, TILE_HEIGHT_MM))[:2]).astype(int)
            cv2.circle(drawn, tuple(at), 15, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(drawn, shown, (at[0] - 9, at[1] + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    legend = (f"{len(tiles)} tiles found: pink, each top face as the finder fitted it (corners drawn sharp), "
              f"a dot at its centre, the table's number above it; orange, the letter read, where the page puts it: "
              f"the corner at the letter's bottom right")
    cv2.rectangle(drawn, (0, 0), (drawn.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(drawn, legend, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    out = frame_path.with_name(f"{frame_path.stem}-tiles.jpg")
    cv2.imwrite(str(out), drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
