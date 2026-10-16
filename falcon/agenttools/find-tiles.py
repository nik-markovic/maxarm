#!/usr/bin/env python3
"""What main.py would see: the tiles in a frame, drawn, with the slots and what the arm can reach.

    ../.venv/bin/python agenttools/find-tiles.py --live          # snapshot -> files/live-found.jpg
    ../.venv/bin/python agenttools/find-tiles.py training/tiles/baseline/tiles-1.png

Each tile's box is drawn where the fit puts it -- top face heavy, the 4 mm down
to the desk light -- with its letter and an arrow towards the letter's top.
A tile the arm cannot reach is drawn grey; main.py's slots are the numbered
rings. Letters the finder could not place on a tile are ringed in red. Nothing moves.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                             # noqa: E402
import numpy as np                                     # noqa: E402

from calibration.camera.capture import capture         # noqa: E402
from maxarm import Camera, MaxArm, Space               # noqa: E402
from tiles import TILE_HEIGHT_MM, Reader, UnplacedLetters, find_tiles  # noqa: E402
import main as example                                 # noqa: E402

BOX_BGR = (255, 0, 255)
OUT_OF_REACH_BGR = (160, 160, 160)
LABEL_BGR = (0, 255, 255)
SLOT_BGR = (255, 255, 0)
REFUSED_BGR = (0, 0, 255)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", help="a saved frame; leave out with --live")
    parser.add_argument("--live", action="store_true", help="snapshot the camera first")
    args = parser.parse_args()
    if args.live:
        frame, source = capture()
        path = ROOT / "files" / "live.png"
        cv2.imwrite(str(path), frame)
        print(f"{source} -> {path.relative_to(ROOT)}")
    elif args.image:
        path = Path(args.image)
        frame = cv2.imread(str(path))
    else:
        parser.error("give a frame, or --live")

    camera = Camera.load(ROOT / "config" / "camera.json")
    arm = MaxArm()                       # not connected: only asked what it can reach
    arm.use_coordinates(Space.CAMERA)
    unplaced = []
    try:
        tiles = find_tiles(frame, camera, Reader())
    except UnplacedLetters as refused:
        tiles, unplaced = refused.tiles, refused.letters
        print(f"REFUSED: {refused}")
    print(f"{len(tiles)} tiles")
    print("  letter  conf   pixel           angle   desk x, y      reachable")
    for tile in sorted(tiles, key=lambda t: t.px):
        reach = arm.can_reach(tile.px, tile.py, TILE_HEIGHT_MM)
        print(f"  {tile.letter}       {tile.confidence:4.2f}  ({tile.px:6.0f}, {tile.py:6.0f})"
              f"  {tile.angle:6.1f}  ({tile.desk_centre[0]:5.0f}, {tile.desk_centre[1]:5.0f})"
              f"   {'yes' if reach else 'no'}")
    out = ROOT / "files" / (path.stem + "-found.jpg")
    cv2.imwrite(str(out), overlay(frame, camera, arm, tiles, unplaced))
    print(f"-> {out.relative_to(ROOT)}")
    return 1 if unplaced else 0


def overlay(frame, camera: Camera, arm: MaxArm, tiles, unplaced) -> np.ndarray:
    drawn = frame.copy()
    for tile in tiles:
        colour = BOX_BGR if arm.can_reach(tile.px, tile.py, TILE_HEIGHT_MM) else OUT_OF_REACH_BGR
        corners = tile.desk_corners
        top = camera.to_pixels(np.column_stack((corners, np.full(4, TILE_HEIGHT_MM))))
        bottom = camera.to_pixels(np.column_stack((corners, np.zeros(4))))
        for a, b in zip(top, bottom):
            cv2.line(drawn, _px(a), _px(b), colour, 1, cv2.LINE_AA)
        cv2.polylines(drawn, [np.round(bottom).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
        cv2.polylines(drawn, [np.round(top).astype(np.int32)], True, colour, 2, cv2.LINE_AA)
        baseline = np.radians(tile.desk_angle)
        letter_up = np.array([np.sin(baseline), -np.cos(baseline)])
        centre = np.array(tile.desk_centre)
        arrow = camera.to_pixels(np.array([[*centre, TILE_HEIGHT_MM],
                                           [*(centre + 7.0 * letter_up), TILE_HEIGHT_MM]]))
        cv2.arrowedLine(drawn, _px(arrow[0]), _px(arrow[1]), LABEL_BGR, 2, cv2.LINE_AA, tipLength=0.3)
        label = _px(camera.to_pixels(np.array([[*(centre - 14.0 * letter_up), TILE_HEIGHT_MM]]))[0])
        cv2.putText(drawn, tile.letter, label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(drawn, tile.letter, label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, LABEL_BGR, 3, cv2.LINE_AA)
    for number, (px, py) in enumerate(example.SLOTS, 1):
        cv2.circle(drawn, (int(px), int(py)), 28, SLOT_BGR, 2, cv2.LINE_AA)
        cv2.putText(drawn, str(number), (int(px) - 9, int(py) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    SLOT_BGR, 2, cv2.LINE_AA)
    for letter in unplaced:
        ring = np.array([[*(np.array(letter.desk_centre) + 12.0 * np.array([np.cos(a), np.sin(a)])),
                          TILE_HEIGHT_MM] for a in np.linspace(0, 2 * np.pi, 48)])
        cv2.polylines(drawn, [np.round(camera.to_pixels(ring)).astype(np.int32)], True, REFUSED_BGR, 3,
                      cv2.LINE_AA)
    return drawn


def _px(point) -> tuple:
    return (int(round(point[0])), int(round(point[1])))


if __name__ == "__main__":
    sys.exit(main())
