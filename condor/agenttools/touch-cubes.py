#!/usr/bin/env python3
"""Touch the top of each cube and stand there for a couple of seconds.

    ../.venv/bin/python agenttools/touch-cubes.py

Hover, down onto the top face, rest, back off, next cube. Ctrl-C stops the arm.
"""

import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))

from maxarm import MaxArm, tuning                                     # noqa: E402
from arm import CUBE_POSITIONS, Stop, slowed_to                       # noqa: E402
from mapping import DeskPlane                                         # noqa: E402

# x, y, AACS z, seconds to stand still after arriving, slow.
# AACS z is height above the desk: 40 is a cube's top face, 70 is clear above it.
# Slow means the board is given 40 mm/s instead of 200 for that one command.
MOVES = (
    (100, -254, 70, 0, False),
    (100, -254, 40, 2, True),        # green cube top
    (100, -254, 70, 0, False),

    (-100, -254, 70, 0, False),
    (-100, -254, 40, 2, True),       # blue cube top
    (-100, -254, 70, 0, False),

    (0, -90, 70, 0, True),           # red is the near cube and the awkward one:
    (0, -90, 40, 2, True),           # red cube top
    (0, -90, 70, 0, False),
)


def main() -> int:
    desk = DeskPlane.through(list(CUBE_POSITIONS.values()))
    arm = MaxArm()
    print("connecting; this resets the board, which homes the arm -- stand clear")
    arm.connect()
    stop = Stop(arm)
    try:
        arm.home()
        for x, y, z, rest_s, is_slow in MOVES:
            if stop.is_requested:
                return 1
            target = desk.aacs_to_board((x, y, z))
            print(f"AACS ({x}, {y}, {z}) -> board z {target[2]:.1f}  ", end="", flush=True)
            with slowed_to(tuning.DESCENT_SPEED_MM_S) if is_slow else contextlib.nullcontext():
                print(arm.move_to(*target))
            time.sleep(rest_s)
        arm.home()
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
