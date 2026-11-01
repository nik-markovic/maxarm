#!/usr/bin/env python3
"""Where the cup is in each frame settle.py kept, against its last frame there, and what the servos said then.

    .venv/bin/python impala/agenttools/arm-to-desk/settle-check.py

The cup is found by its patch of the last frame at the point (the cup's rim, round where the arm's settled reading
puts it), matched in every earlier frame, to a tenth of a pixel; its move turned to desk mm through camera.json at
the cup's height. Beside it, the base servo's reading over the same time, as desk mm along its arc.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from camera import store                                                   # noqa: E402

PATCH = 70                                     # half the patch, frame pixels
SEARCH = 30                                    # how far it is looked for, each way


def subpixel(scores):
    """The best place in a match's scores, refined by a parabola each way."""
    _, _, _, (x, y) = cv2.minMaxLoc(scores)
    def refine(left, middle, right):
        bend = left - 2 * middle + right
        return 0.0 if bend == 0 else 0.5 * (left - right) / bend
    dx = refine(scores[y, x - 1], scores[y, x], scores[y, x + 1]) if 0 < x < scores.shape[1] - 1 else 0.0
    dy = refine(scores[y - 1, x], scores[y, x], scores[y + 1, x]) if 0 < y < scores.shape[0] - 1 else 0.0
    return np.array([x + dx, y + dy])


def mm_per_px(fit, point):
    """Desk mm per frame pixel at a point, at its height: the desk's plane there, through the camera."""
    x, y, z = point
    def seen(dx, dy):
        p = fit.desk_to_pixel @ (x + dx, y + dy, z, 1.0)
        return p[:2] / p[2]
    return np.linalg.inv(np.column_stack((seen(1, 0) - seen(0, 0), seen(0, 1) - seen(0, 0))))


def main():
    fit = store.load_camera(IMPALA / "config")
    data = json.loads((HERE / "settle.json").read_text())
    for index, run in enumerate(data["runs"]):
        frames = [(when, name) for when, point, name in data["frames"] if point == index]
        x0, y0 = run["window"][:2]
        settled = run["settled_desk"]
        centre = fit.desk_to_pixel @ (*settled, 1.0)
        centre = centre[:2] / centre[2] - (x0, y0)
        last = cv2.cvtColor(cv2.imread(str(HERE / "settle-frames" / frames[-1][1])), cv2.COLOR_BGR2GRAY)
        if index == len(data["runs"]) - 1:            # the last point's last frames are the arm parked
            frames = [(when, name) for when, name in frames if when <= run["final"]]
            last = cv2.cvtColor(cv2.imread(str(HERE / "settle-frames" / frames[-1][1])), cv2.COLOR_BGR2GRAY)
        cx, cy = np.round(centre).astype(int)
        template = last[cy - PATCH:cy + PATCH, cx - PATCH:cx + PATCH].astype(np.float32)
        to_mm = mm_per_px(fit, settled)
        polls = np.array([p for p in run["polls"] if p[1] is not None], float)
        print(f"\npoint {index} {run['target'][:2]}: the cup in each frame from its place in the last, desk mm "
              f"(s after the move ended; the app's read ended at {run['read'] - run['done']:.2f} s)")
        for when, name in frames:
            after = when - run["done"]
            if after < -1.5:
                continue
            image = cv2.cvtColor(cv2.imread(str(HERE / "settle-frames" / name)), cv2.COLOR_BGR2GRAY)
            area = image[cy - PATCH - SEARCH:cy + PATCH + SEARCH, cx - PATCH - SEARCH:cx + PATCH + SEARCH]
            scores = cv2.matchTemplate(area.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
            shift = subpixel(scores) - SEARCH
            moved = to_mm @ shift
            near = polls[np.abs(polls[:, 0] - when) < 0.3] if len(polls) else []
            base = f"base servo {near[:, 1].min():.0f}-{near[:, 1].max():.0f}" if len(near) else ""
            print(f"  {after:5.1f} s: {shift[0]:+6.2f} {shift[1]:+6.2f} px = {moved[0]:+5.2f} {moved[1]:+5.2f} mm "
                  f"(match {scores.max():.3f}) {base}")


if __name__ == "__main__":
    main()
