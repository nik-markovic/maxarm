#!/usr/bin/env python3
"""Run the tile finder on a frame and draw what it found.

    ./agenttools/find-tiles.py training/tiles/baseline/tiles-1.png  # -> files/tiles-1-found.jpg
    ./agenttools/find-tiles.py --live NAME           # snapshot to files/NAME.png first

The overlay is the camera frame with each tile's box drawn where the fit puts
it -- top face heavy, the 4 mm down to the desk light -- the letter read, and
an arrow from the face's centre towards the letter's top. A scene the finder
refuses -- tiles touching along their edges -- is drawn too, with each letter it
could not place ringed in red.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import read                             # noqa: E402
import scene                            # noqa: E402
from capture import capture             # noqa: E402
from mapping import DeskMapping         # noqa: E402

BOX_BGR = (255, 0, 255)
LABEL_BGR = (0, 255, 255)
REFUSED_BGR = (0, 0, 255)
BOX_OPACITY = 0.45


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="a frame, or with --live the name to save a snapshot as")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.live:
        frame, source = capture()
        path = ROOT / "files" / f"{args.image}.png"
        cv2.imwrite(str(path), frame)
        print(f"{source} -> {path.relative_to(ROOT)}")
    else:
        path = Path(args.image)
        frame = cv2.imread(str(path))

    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    reader = read.Reader()
    started = time.monotonic()
    unplaced = []
    try:
        tiles = scene.find_tiles(frame, mapping, reader)
    except scene.UnplacedLetters as refused:
        tiles, unplaced = refused.tiles, refused.letters
        print(f"REFUSED: {refused}")
    print(f"{len(tiles)} tiles in {time.monotonic() - started:.1f} s")
    print("  letter  conf   AACS x, y (mm)     baseline   board x, y, z at the top face")
    for tile in sorted(tiles, key=lambda t: (t.centre_mm[0], t.centre_mm[1])):
        board = mapping.aacs_to_board((*tile.centre_mm, scene.TILE_HEIGHT_MM))
        print(f"  {tile.letter}       {tile.confidence:4.2f}  ({tile.centre_mm[0]:7.1f},{tile.centre_mm[1]:7.1f})"
              f"  {tile.baseline_deg:7.1f} deg  ({board[0]:6.1f},{board[1]:7.1f},{board[2]:5.1f})")
    out = ROOT / "files" / (path.stem + "-found.jpg")
    cv2.imwrite(str(out), overlay(frame, mapping, tiles, unplaced))
    print(f"-> {out}")
    return 1 if unplaced else 0


def overlay(frame: np.ndarray, mapping: DeskMapping, tiles, unplaced=()) -> np.ndarray:
    """Boxes blended in lightly so the tile edges show through; labels and arrows solid."""
    boxes = frame.copy()
    height = scene.TILE_HEIGHT_MM
    for tile in tiles:
        corners = tile.desk_corners_mm
        box = np.vstack((np.column_stack((corners, np.full(4, height))),
                         np.column_stack((corners, np.zeros(4)))))
        pixels = np.round(mapping.world_to_pixel(box)).astype(np.int32)
        for a, b in ((0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(boxes, tuple(pixels[a]), tuple(pixels[b]), BOX_BGR, 1, cv2.LINE_AA)
        cv2.polylines(boxes, [pixels[4:]], True, BOX_BGR, 1, cv2.LINE_AA)
        cv2.polylines(boxes, [pixels[:4]], True, BOX_BGR, 2, cv2.LINE_AA)
    drawn = cv2.addWeighted(boxes, BOX_OPACITY, frame, 1.0 - BOX_OPACITY, 0.0)
    for tile in tiles:
        angle = np.radians(tile.desk_baseline_deg)
        up = np.array([np.sin(angle), -np.cos(angle)])
        centre = np.array(tile.desk_centre_mm)
        arrow = mapping.world_to_pixel(np.array([[*centre, height], [*(centre + 7.0 * up), height]]))
        cv2.arrowedLine(drawn, tuple(np.round(arrow[0]).astype(int)), tuple(np.round(arrow[1]).astype(int)),
                        LABEL_BGR, 2, cv2.LINE_AA, tipLength=0.3)
        label = tuple(np.round(mapping.world_to_pixel(
            np.array([[*(centre - 14.0 * up), height]]))[0]).astype(int))
        cv2.putText(drawn, tile.letter, label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(drawn, tile.letter, label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, LABEL_BGR, 3, cv2.LINE_AA)
    for letter in unplaced:
        ring = np.array([[*(np.array(letter.desk_centre_mm) + 12.0 * np.array([np.cos(a), np.sin(a)])), height]
                         for a in np.linspace(0, 2 * np.pi, 48)])
        cv2.polylines(drawn, [np.round(mapping.world_to_pixel(ring)).astype(np.int32)], True,
                      REFUSED_BGR, 3, cv2.LINE_AA)
    return drawn


if __name__ == "__main__":
    sys.exit(main())
