#!/usr/bin/env python3
"""tops.py's model-free check on every session: the cubes' top centres (the diagonals' crossing of the top face, from
the found edges) against the readings, through one homography. Per point its leftover, pixels and mm, split along the
camera's line of sight and across it (the camera's position from that session's camera.json, used for nothing else).
Then each point left out, the others' homography predicting it; and, for each point, how well the others fit
without it (their own worst leftover): the set that fits best without one point names that point as the odd one.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/tops-sessions.py
"""
import json
from pathlib import Path

import numpy as np

from applib.camera.fit import HANDEDNESS, CameraFit
from applib.camera.model import camera_position
from tops import apply, centre_of, homography, top_face

SESSIONS = {
    "10 Nov (today)": ("config/camera.json", "config/camera-refine-points.json"),
    "9-10 Nov": ("config/camera.json", "config/camera-refine-points-previous.json"),
    "5 Nov": ("../work/config-jackal-1105/camera.json", "../work/config-jackal-1105/camera-refine-points.json"),
    "4 Nov": ("../work/config-jackal-1104/camera.json", "../work/config-jackal-1104/camera-refine-points.json"),
}


def leftovers(h, readings, centres):
    inv = np.linalg.inv(h)
    px = np.linalg.norm(apply(h, readings) - centres, axis=1)
    mm = np.array([HANDEDNESS @ (apply(inv, c[None])[0] - r) for c, r in zip(centres, readings)])
    return px, mm


for name, (camera_file, points_file) in SESSIONS.items():
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    lens = HANDEDNESS @ camera_position(fit.cube_fit().true_to_pixel)[:2]          # desk x, y under the lens
    stored = [p for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    readings = np.array([HANDEDNESS @ np.array((p["x"], p["y"])) for p in stored])
    desk = np.array([(p["x"], p["y"]) for p in stored])
    centres = np.array([centre_of(top_face(p)[0]) for p in stored])
    ids = [p["id"] for p in stored]
    n = len(stored)
    print(f"\n=== {name}: {n} points, the lens over desk ({lens[0]:.0f}, {lens[1]:.0f})")
    if n < 5:
        print("  fewer than five points: a homography goes exactly through any four, nothing to check")
        continue
    h = homography(readings, centres)
    px, mm = leftovers(h, readings, centres)
    print("  one homography through all, each point's leftover: px, mm, along the line of sight / across it")
    for k in range(n):
        sight = (desk[k] - lens) / np.linalg.norm(desk[k] - lens)
        across = np.array((-sight[1], sight[0]))
        print(f"    #{ids[k]:2} ({desk[k][0]:7.1f}, {desk[k][1]:7.1f}) {px[k]:5.2f} px  {np.linalg.norm(mm[k]):5.2f} mm"
              f"  along {mm[k] @ sight:+5.2f}  across {mm[k] @ across:+5.2f}   {stored[k]['taken']}"
              f" {stored[k].get('cube') or 'red'}")
    print("  each point left out: the others' homography predicting it (mm), and the others' own worst leftover (mm)")
    for k in range(n):
        keep = [j for j in range(n) if j != k]
        hk = homography(readings[keep], centres[keep])
        _, mk = leftovers(hk, readings[keep], centres[keep])
        _, mo = leftovers(hk, readings[k:k + 1], centres[k:k + 1])
        print(f"    without #{ids[k]:2}: predicts it {np.linalg.norm(mo[0]):6.2f} mm; the others fit to "
              f"{np.max(np.linalg.norm(mk, axis=1)):5.2f} mm at worst, {np.median(np.linalg.norm(mk, axis=1)):5.2f} typically")
