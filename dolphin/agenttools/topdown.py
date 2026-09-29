#!/usr/bin/env python3
"""Write the top-down view of a frame at tile height, for looking at.

    ./agenttools/topdown.py files/tiles-1.png      # -> files/tiles-1-top.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402

import topdown                          # noqa: E402
from mapping import DeskMapping         # noqa: E402

TILE_HEIGHT_MM = 4.0


def main() -> int:
    path = Path(sys.argv[1])
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    view = topdown.render(cv2.imread(str(path)), mapping, TILE_HEIGHT_MM)
    out = path.with_name(path.stem + "-top.png")
    cv2.imwrite(str(out), view.image)
    print(f"{view.image.shape[1]}x{view.image.shape[0]} at {view.px_per_mm:g} px/mm -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
