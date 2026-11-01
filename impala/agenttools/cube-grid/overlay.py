#!/usr/bin/env python3
"""The calibration's desk drawn on a snapshot: desk space every 10 mm at z 0 (magenta), as Fine Tune's grid is, and
optionally the same at another height (cyan), all through camera.json's desk_to_pixel in perspective.

    .venv/bin/python impala/agenttools/cube-grid/overlay.py frame.png out.png [height-mm]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    frame = cv2.imread(str(HERE / sys.argv[1]))
    camera = json.loads((ROOT / "config" / "camera.json").read_text())
    matrix = np.array(camera["desk_to_pixel"])

    def px(x, y, z):
        p = matrix @ (x, y, z, 1.0)
        return p[:2] / p[2]

    steps = np.arange(-300, 301, 10)
    layers = [(0.0, (255, 0, 255))] + ([(float(sys.argv[3]), (255, 255, 0))] if len(sys.argv) > 3 else [])
    for z, colour in layers:
        for k in steps:
            width = 2 if k % 50 == 0 else 1
            for line in ([px(k, a, z) for a in steps], [px(a, k, z) for a in steps]):
                cv2.polylines(frame, [np.round(np.array(line) * 4).astype(np.int32)], False, colour, width,
                              cv2.LINE_AA, shift=2)
    cv2.imwrite(str(HERE / sys.argv[2]), frame)


if __name__ == "__main__":
    main()
