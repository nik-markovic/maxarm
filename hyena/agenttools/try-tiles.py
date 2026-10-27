#!/usr/bin/env python3
"""The tile finder on a still, drawn: no arm, no camera.

    ../.venv/bin/python agenttools/try-tiles.py agenttools/tile-shots/still-1.png \
        agenttools/tile-shots/still-camera.json
    ../.venv/bin/python agenttools/try-tiles.py FRAME.png                  # config/camera.json

Prints each tile found -- desk x, y, turn, how far its weakest side stands out -- and writes `<frame>-tiles.jpg`
beside the frame: each tile's top face outlined where the finder put it, numbered, with a legend.
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
from applib.tiles import TILE_HEIGHT_MM, find_tiles           # noqa: E402

INK = (255, 0, 255)


def main(argv) -> int:
    frame_path = Path(argv[1])
    camera = Camera.load(argv[2] if len(argv) > 2 else ROOT / "config" / "camera.json")
    frame = cv2.imread(str(frame_path))
    started = time.monotonic()
    tiles = find_tiles(frame, camera)
    print(f"{len(tiles)} tiles in {time.monotonic() - started:.2f} s")
    drawn = frame.copy()
    for number, tile in enumerate(tiles, 1):
        print(f"  {number:2d}  desk ({tile.x:7.1f}, {tile.y:7.1f})  turn {tile.turn:6.1f}  "
              f"{math.hypot(tile.x, tile.y):5.1f} mm out  weakest side {tile.contrast:5.1f} x what lies beyond it")
        face = np.array([camera.from_desk((x, y, TILE_HEIGHT_MM))[:2] for x, y in tile.corners])
        cv2.polylines(drawn, [np.round(face).astype(np.int32)], True, INK, 1, cv2.LINE_AA)
        cv2.circle(drawn, (round(tile.pixel[0]), round(tile.pixel[1])), 4, INK, -1, cv2.LINE_AA)
        corner = face[np.argmin(face[:, 1])]
        cv2.putText(drawn, str(number), (round(corner[0]) - 8, round(corner[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(drawn, str(number), (round(corner[0]) - 8, round(corner[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 2, cv2.LINE_AA)
    legend = (f"{len(tiles)} tiles found: pink, each top face as the finder fitted it (corners drawn sharp), "
              f"a dot at its centre; the number above it is the table's")
    cv2.rectangle(drawn, (0, 0), (drawn.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(drawn, legend, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    out = frame_path.with_name(f"{frame_path.stem}-tiles.jpg")
    cv2.imwrite(str(out), drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
