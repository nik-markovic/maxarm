#!/usr/bin/env python3
"""The letters on the tiles: read, and which way up, on the owner's three stills and on tiles drawn at every turn.

The stills' letters and ways up are the owner's tiles as the pictures show them (STATUS-hyena §8, §10); the drawn
tiles (`tile_scene.py`) are straight strokes through a known camera, turned every way, two dotted and two blank.
Neither says how the reader does on letters it has not seen: the owner's eleven tiles are all the real ones there
are, laid out three ways.
"""

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

from maxarm import Camera                                     # noqa: E402
from applib.letters import read_letters                       # noqa: E402
from applib.letters import face                               # noqa: E402
from applib.tiles import find_tiles                           # noqa: E402
import tile_scene                                             # noqa: E402
from harness import check                                     # noqa: E402

SHOTS = HERE / "tile-shots"
STILL_CAMERA = SHOTS / "still-camera.json"
ANY = (0, 1, 2, 3)
# Each still's tiles in the finder's order: the letter, and the quarter turns of the face (face.py) at which it is
# upright, read off the pictures (STATUS-hyena §8). The M and the W carry the owner's dot.
STILL_LETTERS = {
    "still-1.png": [("M", (1,)), ("T", (2,)), ("A", (1,)), ("N", (0, 2)), ("N", (0, 2)), ("E", (0,)), ("V", (0,)),
                    ("T", (3,)), ("E", (0,)), ("O", ANY), ("W", (0,))],
    "still-2.png": [("M", (1,)), ("N", (0, 2)), ("W", (0,)), ("O", ANY)],
    "still-3.png": [("T", (1,)), ("M", (1,)), ("A", (0,)), ("O", ANY), ("E", (0,)), ("T", (0,)), ("V", (0,)),
                    ("E", (0,)), ("W", (0,)), ("N", (1, 3))],
}
DOTTED = "MW"
# Drawn: letters and blanks turned every way, in view of the drawn camera, in reach or not -- reading does not care.
DRAWN = [(-150.0, -150.0, 10.0), (-100.0, -200.0, 125.0), (-120.0, -100.0, -140.0), (-60.0, -230.0, 75.0),
         (-200.0, -60.0, 200.0), (-180.0, -210.0, -60.0), (-40.0, -160.0, 160.0), (-230.0, -130.0, -100.0),
         (-80.0, -120.0, 30.0), (-150.0, -250.0, 250.0)]
DRAWN_LETTERS = "ETNLVZWM  "


def is_upright_at(letter, tile, quarters) -> bool:
    """Whether the reading's way up is one of `quarters`, and its uprights are all of them."""
    expected = sorted(round(face.upright_deg(tile, quarter), 1) for quarter in quarters)
    return (round(letter.upright, 1) in expected
            and (quarters == ANY or sorted(round(angle, 1) for angle in letter.uprights) == expected))


def test_the_owners_letters() -> bool:
    print("the owner's stills, through the calibration of the day: every letter, its way up, the dots:")
    results = []
    camera = Camera.load(STILL_CAMERA)
    for name, truth in STILL_LETTERS.items():
        frame = cv2.imread(str(SHOTS / name))
        tiles = find_tiles(frame, camera)
        started = time.monotonic()
        letters = read_letters(frame, camera, tiles)
        seconds = time.monotonic() - started
        read = "".join(letter.letter if letter.is_read else "?" for letter in letters)
        wrong_way = [index + 1 for index, (tile, letter, (_, quarters)) in enumerate(zip(tiles, letters, truth))
                     if not is_upright_at(letter, tile, quarters)]
        dots = "".join(letter.letter for letter in letters if letter.has_dot)
        least = min(letter.confidence for letter in letters)
        results += [
            check(f"{name}: {''.join(letter for letter, _ in truth)}, every one read",
                  read == "".join(letter for letter, _ in truth),
                  f"read {read}, least sure {least:.2f}, next best at most "
                  f"{max(letter.runner_up_confidence for letter in letters):.2f}; {seconds:.2f} s"),
            check(f"{name}: each the right way up", not wrong_way and len(letters) == len(truth),
                  f"tiles {wrong_way} wrong" if wrong_way else
                  ", ".join(f"{letter.letter} {letter.upright:.0f}" for letter in letters)),
            check(f"{name}: a dot on the M and the W, on nothing else",
                  dots == "".join(letter for letter, _ in truth if letter in DOTTED), dots),
        ]
    return all(results)


def read_drawn(camera_at=tile_scene.CAMERA_AT, looking_at=tile_scene.LOOKING_AT):
    """DRAWN through a camera there: the tiles found, their letters, and which drawn tiles are wholly in view."""
    scene = tile_scene.render(DRAWN, desk="wood", letters=DRAWN_LETTERS, camera_at=camera_at, looking_at=looking_at)
    camera = Camera(scene.desk_to_pixel, tile_scene.SIZE)
    tiles = find_tiles(scene.frame, camera)
    width, height = tile_scene.SIZE
    in_view = [index for index, (x, y, _) in enumerate(DRAWN)
               if all(0 <= px < width and 0 <= py < height for px, py, _ in
                      (camera.from_desk((x + dx, y + dy, 2.0)) for dx in (-13, 13) for dy in (-13, 13)))]
    return tiles, read_letters(scene.frame, camera, tiles), in_view


def wrongly_read(tiles, letters, indices):
    """Of the drawn tiles `indices`: those read as another letter, and those read the wrong way up."""
    misread, wrong_way = [], []
    for (x, y, turn), drawn in ((DRAWN[index], DRAWN_LETTERS[index]) for index in indices):
        index = min(range(len(tiles)), key=lambda each: math.hypot(tiles[each].x - x, tiles[each].y - y))
        letter = letters[index]
        shown = letter.letter if letter.is_read else "?" if letter.letter else " "
        if shown != drawn:
            misread.append(f"{drawn!r} as {shown!r} ({letter.confidence:.2f})")
        elif drawn != " ":
            # The drawn letter's top points 90 degrees clockwise of the tile's x, as desk angles go (tile_scene).
            off = min(abs(face.wrap_deg(angle - (turn - 90.0))) for angle in letter.uprights)
            if off > 3.0 or (drawn not in "NZ" and len(letter.uprights) != 1):
                wrong_way.append(f"{drawn} {letter.upright:.0f} for {face.wrap_deg(turn - 90.0):.0f}")
    return misread, wrong_way


def test_drawn_letters() -> bool:
    print("drawn tiles turned every way, two dotted, two blank: each read, the right way up:")
    tiles, letters, _ = read_drawn()
    misread, wrong_way = wrongly_read(tiles, letters, range(len(DRAWN)))
    blank = [letter for letter in letters if letter.is_blank]
    return all([
        check("all ten found", len(tiles) == len(DRAWN), f"{len(tiles)}"),
        check("each letter read, the blanks blank", not misread, "; ".join(misread)),
        check("each the right way up, to the tile's own turn; N and Z either way", not wrong_way, "; ".join(wrong_way)),
        check("a blank: no letter, no way up, its ink faint", len(blank) == 2
              and all(letter.upright is None and letter.ink_contrast < 0.25 for letter in blank),
              ", ".join(f"{letter.ink_contrast:.2f}" for letter in blank)),
        check("the dots found on the W and the M", sorted(letter.letter for letter in letters if letter.has_dot)
              == ["M", "W"]),
    ])


def test_quarter_turns() -> bool:
    print("a face's quarter turns: each the last turned a quarter, and the dot's corner taken to the bottom right:")
    camera = Camera.load(STILL_CAMERA)
    frame = cv2.imread(str(SHOTS / "still-1.png"))
    tile = find_tiles(frame, camera)[0]
    windows = [face.window(frame, camera, tile, quarter, 27.0, 64).astype(np.float32) for quarter in range(4)]
    # The window at the next quarter turn shows this one turned anticlockwise on the page, as np.rot90 turns.
    turned = [float(np.abs(np.rot90(windows[quarter]) - windows[(quarter + 1) % 4]).mean()) for quarter in range(4)]
    other = float(np.abs(np.rot90(windows[0], -1) - windows[1]).mean())
    corners = {(1, 1): 0, (1, -1): 3, (-1, -1): 2, (-1, 1): 1}
    return all([
        check("each quarter turn the last one turned anticlockwise, to within interpolation", max(turned) < 2.5
              and other > 10, f"mean grey difference {np.round(turned, 2)}; turned the other way {other:.1f}"),
        check("the dot's corner: the quarter turn that shows it at the bottom right",
              all(face.quarter_with(*corner) == quarter for corner, quarter in corners.items())),
        check("a letter upright at quarter turn q points q x 90 - 90 degrees on from the tile's turn",
              face.upright_deg(tile, 1) == face.wrap_deg(tile.turn) and
              face.upright_deg(tile, 0) == face.wrap_deg(tile.turn - 90.0)),
    ])


def test_camera_on_the_left() -> bool:
    """The owner's camera stands to the arm's right; another rig's may stand to its left, the picture mirrored."""
    print("the same drawn tiles, the camera on the arm's left instead: those in view read, the right way up:")
    # tile_scene's camera positions are true mm, whose x is the desk's negated: true +266 is desk -266, the arm's left.
    tiles, letters, in_view = read_drawn((266.0, -287.0, 206.0), (-110.0, -150.0, 0.0))
    misread, wrong_way = wrongly_read(tiles, letters, in_view)
    return all([
        check("the tiles wholly in view found", len(in_view) >= 4 and len(tiles) >= len(in_view),
              f"{len(in_view)} in view: {''.join(DRAWN_LETTERS[index] for index in in_view)!r}; {len(tiles)} found"),
        check("each read, the blank blank", not misread, "; ".join(misread)),
        check("each the right way up", not wrong_way, "; ".join(wrong_way)),
    ])


TESTS = [test_the_owners_letters, test_drawn_letters, test_camera_on_the_left, test_quarter_turns]


if __name__ == "__main__":
    from harness import run
    sys.exit(run(TESTS, "letters"))
