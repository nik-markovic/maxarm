#!/usr/bin/env python3
"""One snapshot from the camera, as RECORD takes one (frames averaged, the calibration's mode and focus). No arm.

    .venv/bin/python impala/agenttools/cube-grid/grab.py [name]

Writes <name>.png (frame.png) beside this file.
"""

import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera.feed import Feed                                     # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "frame"
    feed = Feed(config_dir=ROOT / "config")
    feed.start()
    try:
        image, how = feed.snapshot()
    finally:
        feed.close()
    cv2.imwrite(str(HERE / f"{name}.png"), image)
    print(how, "->", HERE / f"{name}.png")


if __name__ == "__main__":
    main()
