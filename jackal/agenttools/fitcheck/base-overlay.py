#!/usr/bin/env python3
"""The arm's base drawn on CALIBRATE's frame, two ways, to see where each camera puts it:

  cyan     through camera.json (CALIBRATE: the camera tied to the arm's readings);
  magenta  through the camera from the cubes' edges alone (no arm numbers), laid on the arm by the turn and shift
           (no stretch) that best puts the three cubes' centres on the arm's readings.

What is drawn, from the owner's measurements (work/PILOT-kiwi-idea.md), in arm mm round the arm's axis, the base
square along the arm's x and y:
  the four suction cups' rims on the desk: circles of 21 mm radius round (+-60, +-60), z 0;
  the screw nuts on top of the base plate (10 mm up) at the same centres, and their tips 18 mm up: a short line;
  the base plate's four central edges, 124.5 mm apart, 58 mm long, 10 mm up;
  the arm's axis: a line from the desk up 100 mm.
Writes out/base-overlay-<time>.jpg (the whole frame) and out/base-overlay-<time>-base.jpg (the base, 2x).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/base-overlay.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

CONFIG = Path("config")
OUT = Path("agenttools/fitcheck/out")
CUP_CENTRE, CUP_RADIUS = 60.0, 21.0
PLATE_HALF, PLATE_TOP, EDGE_HALF = 62.25, 10.0, 29.0
TIP = 18.0
CYAN, MAGENTA, WHITE = (255, 255, 0), (255, 0, 255), (255, 255, 255)

src = Path("agenttools/fitcheck/edges-only-draw.py").read_text().split("\nfor stamp, config in FRAMES.items():")[0]
ns = {"__name__": "edges"}
exec(compile(src, "edges-only-draw", "exec"), ns)
F = ns["F"]


def shapes():
    """Each piece of the base as a polyline of arm-space points (x, y, z)."""
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            c = np.array((sx * CUP_CENTRE, sy * CUP_CENTRE))
            ring = [(*(c + CUP_RADIUS * np.array((np.cos(a), np.sin(a)))), 0.0) for a in np.linspace(0, 2 * np.pi, 49)]
            out.append(ring)
            out.append([(*c, PLATE_TOP), (*c, TIP)])
    for s in (1, -1):
        out.append([(s * PLATE_HALF, -EDGE_HALF, PLATE_TOP), (s * PLATE_HALF, EDGE_HALF, PLATE_TOP)])
        out.append([(-EDGE_HALF, s * PLATE_HALF, PLATE_TOP), (EDGE_HALF, s * PLATE_HALF, PLATE_TOP)])
    out.append([(0.0, 0.0, 0.0), (0.0, 0.0, 100.0)])
    return [np.array(p, float) for p in out]


def tie(arm_xy, cube_xy):
    """The turn (or turn and mirror) and shift that best put arm_xy on cube_xy: a function of arm points."""
    ma, mc = arm_xy.mean(0), cube_xy.mean(0)
    u, _, vt = np.linalg.svd((cube_xy - mc).T @ (arm_xy - ma))
    rot = u @ vt                                   # a mirror allowed: the cube frame may be the board's mirror
    left = cube_xy - (mc + (arm_xy - ma) @ rot.T)
    return (lambda p: np.column_stack(((p[:, :2] - ma) @ rot.T + mc, p[:, 2]))), np.linalg.norm(left, axis=1), rot


def draw(image, pixels_of, colour, zoom=1.0, origin=(0, 0), width=2):
    for piece in shapes():
        pix = (pixels_of(piece) - np.array(origin)) * zoom
        cv2.polylines(image, [np.round(pix * 16).astype(np.int32)], False, colour, width, cv2.LINE_AA, 4)


frame = cv2.imread(str(CONFIG / "camera-frame.png"))
rec = json.loads((CONFIG / "camera-points.json").read_text())["cubes"]
saved = json.loads((CONFIG / "camera.json").read_text())
matrix = np.array(saved["desk_to_pixel"], float)


def through_saved(p):
    h = np.column_stack((p, np.ones(len(p)))) @ matrix.T
    return h[:, :2] / h[:, 2:3]


cam, placed, _, misses = ns["edges_only"](frame, {c: (p["x"], p["y"]) for c, p in rec.items()})
arm_xy = np.array([(rec[c]["x"], rec[c]["y"]) for c in F.COLOURS])
cube_xy = np.array([placed[k][0][:2] for k in range(3)], float)
to_cubes, left, rot = tie(arm_xy, cube_xy)


def through_edges(p):
    return cam.project(to_cubes(p))


stamp = time.strftime("%H%M")
whole = frame.copy()
draw(whole, through_saved, CYAN)
draw(whole, through_edges, MAGENTA)
lines = [f"cyan: camera.json (CALIBRATE)   magenta: the camera from the cubes' edges alone, laid on the arm by turn+shift",
         f"cubes' centres from the arm's readings after the tie: {', '.join(f'{c} {d:.1f}' for c, d in zip(F.COLOURS, left))} mm "
         f"({'mirrored' if np.linalg.det(rot) < 0 else 'not mirrored'})",
         "drawn: the base's suction-cup rims (r 21 on a 120 square, desk), the nuts 10-18 mm up, the plate's edges, the axis"]
for i, text in enumerate(lines):
    for thick, col in ((5, (0, 0, 0)), (2, WHITE)):
        cv2.putText(whole, text, (20, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, thick, cv2.LINE_AA)
OUT.mkdir(exist_ok=True)
cv2.imwrite(str(OUT / f"base-overlay-{stamp}.jpg"), whole, [cv2.IMWRITE_JPEG_QUALITY, 92])
pts = np.vstack([through_saved(p) for p in shapes()] + [through_edges(p) for p in shapes()])
lo = np.maximum(pts.min(0) - 60, 0).astype(int)
hi = np.minimum(pts.max(0) + 60, (frame.shape[1], frame.shape[0])).astype(int)
crop = cv2.resize(frame[lo[1]:hi[1], lo[0]:hi[0]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
draw(crop, through_saved, CYAN, 2, lo, 2)
draw(crop, through_edges, MAGENTA, 2, lo, 2)
cv2.imwrite(str(OUT / f"base-overlay-{stamp}-base.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"out/base-overlay-{stamp}*.jpg;", lines[1])
