#!/usr/bin/env python3
"""CALIBRATE's steps one at a time on config/ as it is (frame and readings, the turn fix off): how well the cubes'
edges fit after step 3 -- the cubes standing where their own shapes put them, a turn, shift and stretch between the
arm's readings and them -- and after step 4, each cube held where the arm read it; and the cubes' spacing as the
step-3 camera sees it against the arm's readings. Nothing written.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/steps.py
"""
import itertools, json, math
from pathlib import Path
import cv2
import numpy as np
from maxarm import raw
from applib.camera import cubes, edges, fit as F

CONFIG = Path("config")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
recorded = json.loads((CONFIG / "camera-points.json").read_text())["cubes"]
points = {c: (p["x"], p["y"]) for c, p in recorded.items()}

blobs = cubes.find_blobs(frame)
starts = [(*F._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
overlap, params, naming = max(starts, key=lambda s: s[0])
params = F._fit_outlines(naming, points, params)
step3, d3, found = F._fit_edges(frame, naming, points, params)
step4 = F._fit_on_arm(frame, points, step3, found)


def edge_fit(p):
    d = np.concatenate([edges.signed_distances(c, *f) for c, f in zip(F._corners(p, points), found)])
    per = [np.abs(edges.signed_distances(c, *f)) for c, f in zip(F._corners(p, points), found)]
    return (f"{np.median(np.abs(d)):.2f} px typically, {np.percentile(np.abs(d), 90):.2f} at 90%  ("
            + ", ".join(f"{col} {np.median(e):.2f}/{np.percentile(e, 90):.1f}" for col, e in zip(F.COLOURS, per)) + ")")


print(f"step 3, the cubes where their shapes put them: edges {edge_fit(step3)}; focal {step3[F.FOCAL]:.0f}")
print(f"step 4, each cube held at its arm reading:    edges {edge_fit(step4)}; focal {step4[F.FOCAL]:.0f}")

seen = {c: centre for c, centre in zip(F.COLOURS, F._centres(step3, points))}   # true mm, as the camera sees them
arm = {c: F.HANDEDNESS @ np.array(points[c]) for c in F.COLOURS}
low, high, skew = F.stretch_of(F._desk_to_true(step3))
print(f"\nthe arm's readings against the cubes as the step-3 camera sees them: the arm's distances x{low:.4f} to "
      f"x{high:.4f}, askew {skew:.2f} deg")


def turned(c, turn):
    p = recorded[c]
    rx, ry, _ = raw.raw_position((p["x"], p["y"], p["z"] + raw.DESK_Z_MM))
    r, b = math.hypot(rx, ry), math.atan2(rx, -ry)
    r += raw._further_out(max(p["z"], 0.0))
    b += math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return np.array((r * math.sin(b), -r * math.cos(b)))


print(f"{'between':14} {'camera (step 3)':>16} {'arm, no turn fix':>17} {'arm, turn fix':>14}")
for a, b in itertools.combinations(F.COLOURS, 2):
    cam = float(np.linalg.norm(seen[a] - seen[b]))
    print(f"{a + '-' + b:14} {cam:13.1f} mm {np.linalg.norm(arm[a] - arm[b]):14.1f} mm "
          f"{np.linalg.norm(turned(a, (-1.36, 0.33)) - turned(b, (-1.36, 0.33))):11.1f} mm")
