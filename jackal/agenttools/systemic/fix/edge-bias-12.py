#!/usr/bin/env python3
"""The 4 Nov points' cube edges under the camera that matches today's sheets (carried back by the measured 4 px),
against under the 4 Nov camera.json: every edge point found, signed outward from its cube (+ outside the 40 mm cube
drawn), by kind of edge and by whether the face beside it is lit or in shade, each cube placed by its edges through
that camera. What differs between the two columns is what pulls a cube-only fit to the wrong camera.

    edge-bias-12.py
"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("atr", HERE / "arm-through-real.py")
atr = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(atr)
from applib.camera.model import camera_position, cube_corners, visible_edges  # noqa: E402

KINDS = ("top outline", "upright outline", "bottom outline", "top inner", "upright inner")
cameras = (("4 Nov camera.json", atr.T4, False), ("sheets' camera", atr.real, True))
rows = {name: {k: [] for k in KINDS} for name, _, _ in cameras}
per_point = []
for p in atr.stored:
    line = [f"{p.id:3d}"]
    for name, T, frame_moved in cameras:
        centre, _ = atr.place(T, p, frame_moved)
        # the turn as placed: refit quickly with the same routine's start and keep its result
        starts, ends, found = p.sighting.edges
        found = atr.moved(found) if frame_moved else found
        turn = math.radians(p.turn_deg) + (atr.turned if frame_moved else 0.0)
        lens = camera_position(T)
        corners3 = cube_corners(centre, turn)
        q = T @ np.c_[corners3, np.ones(8)].T
        corners = (q[:2] / q[2]).T
        m = T @ np.array([centre[0], centre[1], 20.0, 1.0]); middle = m[:2] / m[2]
        outline = {tuple(sorted((e.start, e.end))): e.is_outline for e in visible_edges(centre, turn, lens)}
        by = {k: [] for k in KINDS}
        for a, b, f in zip(starts, ends, found):
            key = tuple(sorted((int(a), int(b))))
            if key not in outline:
                continue
            side = corners[b] - corners[a]
            normal = np.array([-side[1], side[0]]) / np.linalg.norm(side)
            if (middle - corners[a]) @ normal > 0:
                normal = -normal
            d = (f - corners[a]) @ normal
            lo, hi = sorted(key)
            kind = ("top" if lo >= 4 else "bottom" if hi < 4 else "upright") + (" outline" if outline[key] else " inner")
            if abs(d) < 5:
                by[kind].append(d)
        for k in KINDS:
            rows[name][k] += by[k]
        line.append(" ".join(f"{np.mean(by[k]):+5.2f}" if by[k] else "  -  " for k in KINDS))
    per_point.append(" | ".join(line))
print("id  | 4 Nov camera.json: " + " ".join(k.split()[0][:3] + "-" + k.split()[1][:3] for k in KINDS)
      + " | sheets' camera: same")
for line in per_point:
    print(line)
for name in rows:
    print(f"{name:20s} all: " + " | ".join(f"{k} {np.mean(v):+5.2f} ({len(v)})" for k, v in rows[name].items()))
