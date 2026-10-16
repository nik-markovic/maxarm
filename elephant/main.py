#!/usr/bin/env python3
"""Pick up the tiles the camera sees and line them up in a row, upright where the cup allows.

    ./main.py                 # snapshot the camera, then move the arm
    ./main.py FRAME.png       # use a frame taken earlier instead

Everything is in camera coordinates: a tile is where the camera sees it, a slot
is a pixel, and the library turns both into arm moves. The cup turns 90 degrees
either way, so a tile that would need more goes into its slot as it was found.

Ctrl-C holds the arm where it is; a second one quits.
"""

import sys
from pathlib import Path

import cv2

from calibration.capture import capture
from maxarm import Camera, MaxArm, MoveStatus, Space
from tiles import TILE_HEIGHT_MM, Reader, find_tiles

CAMERA = Path(__file__).resolve().parent / "config" / "camera.json"
# Where the word goes, left to right: a row of free desk across the bottom of the
# owner's frame, 24 mm apart, as the camera sees the top of a tile set down there.
SLOTS = [(365, 1554), (600, 1552), (829, 1550), (1053, 1549), (1271, 1547), (1485, 1545),
         (1694, 1544)]
UPRIGHT = 0.0                         # the letter's baseline along +px: it reads upright
DROP_HEIGHT_MM = TILE_HEIGHT_MM + 1.0  # let go a millimetre above where it will sit


def main(argv) -> int:
    frame = cv2.imread(argv[1]) if len(argv) > 1 else capture()[0]
    tiles = find_tiles(frame, Camera.load(CAMERA), Reader())
    with MaxArm() as arm:
        arm.stop_on_ctrl_c()
        arm.use_coordinates(Space.CAMERA)
        arm.set_clearance(TILE_HEIGHT_MM + 10.0)
        reachable = [tile for tile in tiles if arm.can_reach(tile.px, tile.py, TILE_HEIGHT_MM)]
        for tile, (px, py) in zip(reachable, SLOTS):
            result, facing = arm.grab(tile.px, tile.py, TILE_HEIGHT_MM, angle=tile.angle), ""
            if result:
                result, facing = arm.drop(px, py, DROP_HEIGHT_MM, angle=UPRIGHT), "upright"
                if result.status is MoveStatus.REFUSED:          # more turning than the cup has
                    result, facing = arm.drop(px, py, DROP_HEIGHT_MM, angle=tile.angle), "as found"
            print(f"{tile.letter} {facing}: {result}")
            if result.status is MoveStatus.STOPPED:
                break
        arm.home()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
