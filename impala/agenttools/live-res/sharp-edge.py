"""How sharp the picture is, on a crisp edge: the sheet's cut edge against the dark desk, its 10-90% rise in pixels,
averaged along 200 px of it. Frames given on the command line. No arm.

    ../.venv/bin/python agenttools/live-res/sharp-edge.py FRAME.png ...
"""
import sys

import cv2
import numpy as np


def paper_edge_width(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # The sheet's left edge: the brightest region's boundary, left of the cube, rows 1150-1350.
    widths = []
    for row in range(1150, 1351, 10):
        line = gray[row, 600:1400]
        bright = line > 0.5 * (line.max() + np.percentile(line, 5))
        if not bright.any():
            continue
        start = int(np.argmax(bright)) + 600                      # first bright pixel from the left: the edge
        profile = gray[row - 2:row + 3, start - 15:start + 16].mean(axis=0)
        low, high = profile[:5].mean(), profile[-5:].mean()
        level = (profile - low) / (high - low)
        offsets = np.arange(-15, 16)
        try:
            cross = [offsets[np.argmax(level >= t)] - (level[np.argmax(level >= t)] - t) /
                     max(level[np.argmax(level >= t)] - level[np.argmax(level >= t) - 1], 1e-6) for t in (0.1, 0.9)]
            widths.append(cross[1] - cross[0])
        except IndexError:
            pass
    return float(np.median(widths)), len(widths)


for name in sys.argv[1:]:
    width, count = paper_edge_width(cv2.imread(name))
    print(f"{name}: paper edge 10-90% {width:.2f} px (horizontal, {count} rows)")
