#!/usr/bin/env python3
"""The reader, through the whole tile finder, against every saved scene whose letters are known.

What a rebuilt reader has to pass before it replaces files/reader.tflite: the
last step of make-reader.sh. Everything it needs is in `baseline/`: the scenes,
and the calibration they were taken with -- not the live one in config/, which
belongs to whatever camera and desk are set up now, or to none.

For each frame: the letters found must be exactly the letters on the desk, each
read with confidence, and where a scene's orientations were confirmed against
the real tiles by the owner, each tile's baseline must be within a few degrees.
Positions are matched loosely: a recalibration moves them by millimetres.

And every scene that breaks the rule -- tiles touching along their edges -- must
be refused with `UnplacedLetters`, naming the letters it could not place.

    ../.venv/bin/python training/tiles/check-reader.py
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASELINE = HERE / "baseline"
sys.path.insert(0, str(ROOT))

import cv2                              # noqa: E402

from maxarm import Camera, Space, Spaces  # noqa: E402
from tiles import Reader, TILE_HEIGHT_MM, UnplacedLetters, find_tiles  # noqa: E402

# The letters on the desk in each fixture, in any order.
SCENES = {
    "tiles-1.png": "AODLTEWVADDREEN",
    # Flashlight high, half the tiles on white letter paper, half on the desk: 19 tiles, owner's count.
    "tiles-3.png": "AABDDDEEEILNNORTUVW",
}
MIN_CONFIDENCE = 0.9
# Confirmed by the owner against the real tiles, 2026-09-29: (letter, arm x, y, baseline deg).
# Arm space, because that is where they were recorded: each tile found is converted there.
ORIENTATIONS = {
    "tiles-1.png": [("O", -120, -281, 34), ("A", -119, -254, 14), ("L", -103, -217, -9),
                    ("T", -86, -311, 72), ("W", -80, -190, 9), ("E", -80, -245, 16),
                    ("V", -54, -220, -7), ("A", -38, -280, -148), ("D", -34, -182, 12),
                    ("R", -10, -245, 61), ("D", -10, -182, 8), ("E", 16, -274, -133),
                    ("D", 19, -177, 0), ("E", 20, -206, 13), ("N", 52, -202, 10)],
}
SAME_PLACE_MM = 6.0
SAME_TURN_DEG = 5.0
# Tiles snug along their edges (a D, E, N clump), flashlight low and high: must be refused.
REFUSED = ("tiles-2-low.png", "tiles-2-high.png")


def main() -> int:
    camera = Camera.load(BASELINE / "camera.json")
    spaces = Spaces(BASELINE)
    reader = Reader()
    failures = 0
    for name, letters in SCENES.items():
        frame = cv2.imread(str(BASELINE / name))
        started = time.monotonic()
        problems = []
        try:
            tiles = find_tiles(frame, camera, reader)
        except UnplacedLetters as refused:
            tiles = refused.tiles
            problems.append(f"refused: {refused}")
        elapsed = time.monotonic() - started
        found = "".join(sorted(tile.letter for tile in tiles))
        expected = "".join(sorted(letters))
        if found != expected:
            problems.append(f"read {found}, desk has {expected}")
        for tile in tiles:
            if tile.confidence < MIN_CONFIDENCE:
                problems.append(f"{tile.letter} at {tile.desk_centre} read at {tile.confidence:.2f}")
        in_arm = [(tile.letter, *in_arm_space(spaces, tile)) for tile in tiles]
        for letter, x, y, baseline in ORIENTATIONS.get(name, []):
            near = [found for found in in_arm if found[0] == letter
                    and abs(found[1] - x) < SAME_PLACE_MM and abs(found[2] - y) < SAME_PLACE_MM]
            if not near:
                problems.append(f"no {letter} near ({x}, {y})")
            elif abs((near[0][3] - baseline + 180) % 360 - 180) > SAME_TURN_DEG:
                problems.append(f"{letter} at ({x}, {y}) faces {near[0][3]:.0f} deg, "
                                f"confirmed {baseline}")
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {name}: {len(tiles)} tiles in {elapsed:.1f} s")
        for problem in problems:
            print(f"       {problem}")
    for name in REFUSED:
        try:
            tiles = find_tiles(cv2.imread(str(BASELINE / name)), camera, reader)
        except UnplacedLetters as refused:
            print(f"ok   {name}: refused, {''.join(letter.letter for letter in refused.letters)} unplaced")
            continue
        failures += 1
        print(f"FAIL {name}: {len(tiles)} tiles and no refusal; its tiles touch along their edges")
    return 1 if failures else 0


def in_arm_space(spaces: Spaces, tile):
    """The tile's centre and baseline as the arm has them: x, y, degrees."""
    top = (tile.px, tile.py, TILE_HEIGHT_MM)
    x, y, _ = spaces.convert(top, Space.CAMERA, Space.ARM)
    return x, y, spaces.convert_angle(top, tile.angle, Space.CAMERA, Space.ARM)


if __name__ == "__main__":
    sys.exit(main())
