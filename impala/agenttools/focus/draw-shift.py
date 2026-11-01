"""The reference frame with arrows: where desk points in reach move in the picture at manual focus 100 and 250, and
after an autofocus hunt, against autofocus as the app opens it. Arrows drawn 5 times their length. Reads the
frames c920-focus.py wrote.

    ../.venv/bin/python agenttools/focus/draw-shift.py FRAMES_DIR OUT.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

frames, out = Path(sys.argv[1]), sys.argv[2]
source = Path(__file__).with_name("c920-focus.py").read_text()
namespace = {"__name__": "draw", "__file__": str(Path(__file__).with_name("c920-focus.py"))}
exec(source.split("camera = open_camera()")[0].replace("out = Path(sys.argv[1])", "out = Path('/tmp')"), namespace)
warp_to, desk_moved_mm, plane, DESK = (namespace[name] for name in ("warp_to", "desk_moved_mm", "plane", "DESK"))

reference = cv2.imread(str(frames / "reference-af.png"))
picture = reference.copy()
pixels = np.c_[DESK, np.ones(len(DESK))] @ plane.T
pixels = pixels[:, :2] / pixels[:, 2:]
cases = [("focus-100.png", (0, 200, 255), "manual focus 100"), ("focus-250.png", (60, 60, 255), "manual focus 250"),
         ("hunt-1.png", (255, 200, 0), "after an autofocus hunt")]
for row, (name, ink, label) in enumerate(cases):
    warp = warp_to(reference, cv2.imread(str(frames / name)))
    moved = np.c_[pixels, np.ones(len(pixels))] @ warp.T
    mm = desk_moved_mm(warp)
    for start, end in zip(pixels, moved):
        tip = start + 5 * (end - start)
        cv2.arrowedLine(picture, tuple(map(int, start)), tuple(map(int, tip)), ink, 2, cv2.LINE_AA, tipLength=0.2)
    text = f"{label}: desk points moved {np.median(mm):.2f} mm median, {mm.max():.2f} worst"
    cv2.putText(picture, text, (20, 40 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(picture, text, (20, 40 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, 2, cv2.LINE_AA)
note = "arrows: a desk point in reach (100, 180, 260 mm) moved, drawn x5 (the hunt's too small to see)"
cv2.putText(picture, note, (20, 40 + 34 * len(cases)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
cv2.putText(picture, note, (20, 40 + 34 * len(cases)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            cv2.LINE_AA)
cv2.imwrite(out, picture)
