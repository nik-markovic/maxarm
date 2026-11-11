#!/usr/bin/env python3
"""The three cubes by their edges alone: one square-pixel camera and each cube's place and turn, fitted to the edges
found in CALIBRATE's frame, no arm numbers anywhere -- only where the desk's origin is and which way it points are
held (no picture says that): the green cube at the origin, blue along +x from it. The principal point held loosely
to the picture's middle as the app holds it, and free. The cubes' spacing with its uncertainty (from the fit's
own scatter), against the arm's readings made true with and without the turn fix.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/edges-only.py
"""
import itertools, json, math
from pathlib import Path
import cv2
import numpy as np
from maxarm import raw
from applib.camera import cubes, edges, fit as F
from applib.camera.model import Pinhole, cube_corners, least_squares

FRAMES = {"17:30 (now)": Path("config"), "10:31 (this morning)": Path("../work/config-jackal-1110-1720")}
TURN_THEN = (-1.36, 0.33)        # raw.py's turn fix as it was when the morning's readings were taken


def readings(config, turn_now):
    """The three readings as recorded, and made true with the turn fix and without it."""
    rec = json.loads((config / "camera-points.json").read_text())["cubes"]
    out = {}
    for c, p in rec.items():
        z = p["z"] + raw.DESK_Z_MM
        r0 = math.hypot(p["x"], p["y"])
        b_true = math.atan2(p["x"], -p["y"])
        # the servos' own bearing: undo the turn fix the reading was made with (fixed point, as raw.py does)
        b = b_true
        for _ in range(8):
            b = b_true - math.radians(turn_now[0] * math.sin(2 * b) + turn_now[1] * (math.cos(2 * b) - 1))
        made = {}
        for name, turn in (("turn fix", (-1.36, 0.33)), ("no turn fix", (0.0, 0.0))):
            bb = b + math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
            made[name] = np.array((r0 * math.sin(bb), -r0 * math.cos(bb)))
        out[c] = made
    return rec, out


for label, config in FRAMES.items():
    frame = cv2.imread(str(config / "camera-frame.png"))
    turn_now = (0.0, 0.0) if config == Path("config") else TURN_THEN
    rec, made = readings(config, turn_now)
    points = {c: (p["x"], p["y"]) for c, p in rec.items()}
    blobs = cubes.find_blobs(frame)
    starts = [(*F._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
    _, params, naming = max(starts, key=lambda s: s[0])
    params = F._fit_outlines(naming, points, params)
    step3, _, found = F._fit_edges(frame, naming, points, params)
    centres = F._centres(step3, points)
    order = list(F.COLOURS)                                   # blue, green, red
    g, bl, rd = order.index("green"), order.index("blue"), order.index("red")
    axis = (centres[bl] - centres[g]) / np.linalg.norm(centres[bl] - centres[g])
    normal = np.array((-axis[1], axis[0]))

    def layout(p):
        """Cube centres (true mm) from the parameters: green fixed, blue along the axis, red free."""
        cs = [None] * 3
        cs[g] = centres[g]
        cs[bl] = centres[g] + p[10] * axis
        cs[rd] = centres[g] + p[12] * axis + p[13] * normal
        return cs

    def residuals(p, hold_centre):
        cam = Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9])
        turns = {g: p[9], bl: p[11], rd: p[14]}
        cs = layout(p)
        parts = [edges.signed_distances(cam.project(cube_corners(cs[k], turns[k])), *found[k]) for k in range(3)]
        if hold_centre:
            parts.append((p[1:3] - np.array([frame.shape[1] / 2, frame.shape[0] / 2])) / (F.CENTRE_SPREAD * frame.shape[1]))
        return np.concatenate(parts)

    rel_r = centres[rd] - centres[g]
    start = np.concatenate((step3[:9], [step3[F.TURNS][g], np.linalg.norm(centres[bl] - centres[g]),
                                        step3[F.TURNS][bl], rel_r @ axis, rel_r @ normal, step3[F.TURNS][rd]]))
    steps = np.concatenate((F.STEPS[:9], [1e-5, 0.01, 1e-5, 0.01, 0.01, 1e-5]))
    print(f"\n=== {label}: the edges CALIBRATE found, {sum(len(f[2]) for f in found)} points")
    for hold in (True, False):
        p, r, normal_m = least_squares(lambda t: residuals(t, hold), start, steps, huber=F.EDGE_HUBER_PX, iterations=200)
        e = np.abs(r[:-2] if hold else r)
        sigma2 = float(np.sum(np.minimum(e, F.EDGE_HUBER_PX) * e) / max(len(e) - len(p), 1))
        cov = sigma2 * np.linalg.inv(normal_m)
        cs = layout(p)

        def dist(q, a, b):
            c2 = layout(q)
            return float(np.linalg.norm(c2[a] - c2[b]))

        print(f"  principal point {'held loosely' if hold else 'free'}: edges {np.median(e):.2f} px typically, "
              f"{np.percentile(e, 90):.2f} at 90%; focal {p[0]:.0f} px, aim ({p[1]:.0f}, {p[2]:.0f}), lens "
              f"{Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9]).position[2]:.0f} mm over the desk")
        for a, b in itertools.combinations(range(3), 2):
            grad = np.array([(dist(p + np.eye(len(p))[i] * steps[i], a, b) - dist(p, a, b)) / steps[i]
                             for i in range(len(p))])
            sd = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
            na, nb = order[a], order[b]
            print(f"    {na}-{nb}: edges alone {dist(p, a, b):6.1f} +- {sd:.1f} mm | arm with the turn fix "
                  f"{np.linalg.norm(made[na]['turn fix'] - made[nb]['turn fix']):6.1f}, without "
                  f"{np.linalg.norm(made[na]['no turn fix'] - made[nb]['no turn fix']):6.1f}")
