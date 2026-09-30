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
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402

import read                             # noqa: E402
import scene                            # noqa: E402
from mapping import DeskMapping         # noqa: E402

# The letters on the desk in each fixture, in any order.
SCENES = {
    "tiles-1.png": "AODLTEWVADDREEN",
    # Flashlight high, half the tiles on white letter paper, half on the desk: 19 tiles, owner's count.
    "tiles-3.png": "AABDDDEEEILNNORTUVW",
}
MIN_CONFIDENCE = 0.9
# Confirmed by the owner against the real tiles, 2026-09-29: (letter, AACS x, y, baseline deg).
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
    mapping = DeskMapping.load(BASELINE / "calibration.json")
    reader = read.Reader()
    failures = 0
    for name, letters in SCENES.items():
        frame = cv2.imread(str(BASELINE / name))
        started = time.monotonic()
        problems = []
        try:
            tiles = scene.find_tiles(frame, mapping, reader)
        except scene.UnplacedLetters as refused:
            tiles = refused.tiles
            problems.append(f"refused: {refused}")
        elapsed = time.monotonic() - started
        found = "".join(sorted(tile.letter for tile in tiles))
        expected = "".join(sorted(letters))
        if found != expected:
            problems.append(f"read {found}, desk has {expected}")
        for tile in tiles:
            if tile.confidence < MIN_CONFIDENCE:
                problems.append(f"{tile.letter} at {tile.centre_mm} read at {tile.confidence:.2f}")
        for letter, x, y, baseline in ORIENTATIONS.get(name, []):
            near = [t for t in tiles if t.letter == letter
                    and abs(t.centre_mm[0] - x) < SAME_PLACE_MM and abs(t.centre_mm[1] - y) < SAME_PLACE_MM]
            if not near:
                problems.append(f"no {letter} near ({x}, {y})")
            elif abs((near[0].baseline_deg - baseline + 180) % 360 - 180) > SAME_TURN_DEG:
                problems.append(f"{letter} at ({x}, {y}) faces {near[0].baseline_deg:.0f} deg, "
                                f"confirmed {baseline}")
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {name}: {len(tiles)} tiles in {elapsed:.1f} s")
        for problem in problems:
            print(f"       {problem}")
    for name in REFUSED:
        try:
            tiles = scene.find_tiles(cv2.imread(str(BASELINE / name)), mapping, reader)
        except scene.UnplacedLetters as refused:
            print(f"ok   {name}: refused, {''.join(letter.letter for letter in refused.letters)} unplaced")
            continue
        failures += 1
        print(f"FAIL {name}: {len(tiles)} tiles and no refusal; its tiles touch along their edges")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
