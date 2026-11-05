#!/usr/bin/env python3
"""The arm as a check on the camera that matches the grid sheets: the 4 Nov points' cubes placed through it, then
the arm's readings laid on them by a turn and a shift, as SAVE does; against the same through 4 Nov's camera.json.
The camera has turned ~4 px since 4 Nov 10:23: the image motion is measured on bare desk (wood grain) round the
sheets and taken out by an affine map of the 4 Nov pixels into today's -- no cubes, no arm in that.

    arm-through-real.py
"""
import importlib.util
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tj", HERE / "today-joint.py")
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)
from applib.camera import edges, store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from applib.camera.model import cube_corners  # noqa: E402
from applib.camera.sighting import Sighting  # noqa: E402
from camera import refine  # noqa: E402

JACKAL = tj.JACKAL
then = cv2.imread(str(JACKAL / "config" / "camera-frame.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
now = cv2.imread(str(HERE / "out" / "scene.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
patches = [(60, 1330, 700, 1470), (800, 1400, 1400, 1470), (1050, 0, 1240, 120), (2000, 0, 2300, 200),
           (1450, 1420, 2100, 1472), (2250, 250, 2540, 520)]
src, dst = [], []
for x0, y0, x1, y1 in patches:
    a, b = then[y0:y1, x0:x1], now[y0:y1, x0:x1]
    (dx, dy), peak = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    if peak < 0.3:
        print(f"   patch {x0},{y0}: no match (peak {peak:.2f}), left out")
        continue
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    src.append(c); dst.append((c[0] + dx, c[1] + dy))
    print(f"   patch at ({c[0]:.0f}, {c[1]:.0f}): moved ({dx:+.2f}, {dy:+.2f}) px, peak {peak:.2f}")
src, dst = np.array(src), np.array(dst)
A = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T          # today = A @ [then, 1]
left = np.linalg.norm(np.c_[src, np.ones(len(src))] @ A.T - dst, axis=1)
print(f"affine 4 Nov -> today from {len(src)} desk patches, each within {left.max():.2f} px")


def moved(points):
    return np.c_[points, np.ones(len(points))] @ A.T


state = pickle.load(open("/tmp/today-joint.pkl", "rb"))
real = tj.matrix(state["both"][:tj.CAM])
saved = store.load_camera(JACKAL / "config")
config = Path("/tmp/c1104")
stored = [p for p in store.load_refine_points(config, store.load_camera(config)) if p.edges]
arm = np.array([(p.x, p.y) for p in stored])

# the turn of a cube in the real camera's desk frame against the 4 Nov frame: through pixels, both ways
T4 = saved.true_to_pixel


def desk_of(T, pixels):
    w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[pixels, np.ones(len(pixels))].T)
    return (w[:2] / w[2]).T


grid = np.array([(x, y) for x in range(-200, 201, 50) for y in range(-250, -79, 50)], float)
g4 = grid @ saved.desk_to_true.T
pix4 = (T4 @ np.c_[g4, np.zeros(len(g4)), np.ones(len(g4))].T); pix4 = (pix4[:2] / pix4[2]).T
greal = desk_of(real, moved(pix4))
u, sv, vt = np.linalg.svd((g4 - g4.mean(0)).T @ (greal - greal.mean(0)))
R = (u @ vt).T
turned = math.atan2(R[1, 0], R[0, 0])
print(f"the real camera's desk frame is turned {math.degrees(turned):+.2f} deg from 4 Nov's (det {np.linalg.det(R):+.0f})")


def place(T, p, frame_moved):
    starts, ends, found = p.sighting.edges
    found = moved(found) if frame_moved else found
    pixel = moved(np.array([p.pixel]))[0] if frame_moved else np.array(p.pixel)
    centre = desk_of(T, np.array([pixel]))[0]
    turn = math.radians(p.turn_deg) + (turned if frame_moved else 0.0)
    x = np.array([*centre, turn])
    for _ in range(40):
        def res(y):
            q = T @ np.c_[cube_corners(y[:2], y[2]), np.ones(8)].T
            return edges.signed_distances((q[:2] / q[2]).T, starts, ends, found)
        r = res(x)
        w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
        J = np.column_stack([(res(x + np.eye(3)[i] * h) - r) / h for i, h in enumerate((1e-3, 1e-3, 1e-6))])
        step = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
        x = x + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return x[:2], float(np.median(np.abs(res(x))))


for name, T, frame_moved in (("4 Nov camera.json (12-point SAVE)", T4, False),
                             ("the camera that matches the sheets", real, True)):
    placed = [place(T, p, frame_moved) for p in stored]
    cubes = np.array([c for c, _ in placed])
    edge = np.array([e for _, e in placed])
    turn, shift, mirror = link_arm(arm, cubes)
    back = np.linalg.solve(turn @ mirror, (cubes - shift).T).T
    lo = []
    for i in range(len(arm)):
        rest = [j for j in range(len(arm)) if j != i]
        t, s, m = link_arm(arm[rest], cubes[rest])
        lo.append(float(np.linalg.norm(np.linalg.solve(t @ m, cubes[i] - s) - arm[i])))
    print(f"\n{name}: cubes' edges {np.median(edge):.2f} px typ; the arm laid by a turn and a shift: left out "
          f"{np.median(lo):.2f} mm typ, {max(lo):.2f} worst")
    for p, a, c, l in sorted(zip(stored, arm, back, lo), key=lambda t: math.atan2(t[1][0], -t[1][1])):
        b = math.atan2(a[0], -a[1]); d = c - a
        print(f"   {p.id:3d} bear {math.degrees(b):+6.1f} reach {np.hypot(*a):5.0f}: out {d @ (math.sin(b), -math.cos(b)):+5.2f} "
              f"round {d @ (math.cos(b), math.sin(b)):+5.2f}  left out {l:4.2f}")
