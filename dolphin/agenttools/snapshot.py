#!/usr/bin/env python3
"""Take one still of the scene and save it losslessly under files/.

    ./agenttools/snapshot.py tiles-1        # -> files/tiles-1.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibration"))

import cv2                              # noqa: E402

from capture import capture             # noqa: E402


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    image, source = capture()
    path = ROOT / "files" / f"{name}.png"
    cv2.imwrite(str(path), image)
    print(f"{image.shape[1]}x{image.shape[0]} from {source} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
