#!/usr/bin/env python3
"""One kind of edge pushed 1 px outward on all three CALIBRATE cubes, the camera refitted: how far the lens and the
prediction at point 4 move. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/edge-push.py
"""
import json
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares, cube_corners, project
from applib.camera.sighting import Sighting
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path("/tmp/c1110/camera.json").read_text()))
pts = [RefinePoint(**p).sighting for p in json.loads(Path("/tmp/c1110/camera-refine-points.json").read_text())["points"]]
cubes, p4 = pts[:3], pts[3]
first = fit.cube_fit().true_to_pixel


def camera(sightings):
    on_arm = refine._OnArm(fit.frame_size, sightings, first)
    n = least_squares(on_arm.residuals, refine._numbers(first), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=300, derivatives=on_arm.derivatives)[0]
    return n


def pushed(s, kind, px):
    starts, ends, found = s.edges
    # outward: away from the cube's middle in the picture, through the current camera
    centre = np.array([*HANDEDNESS @ (s.x, s.y)])
    corners = project(first, cube_corners(centre, np.radians(s.turn_deg)))
    middle = corners.mean(axis=0)
    side = corners[ends] - corners[starts]
    normal = np.column_stack((-side[:, 1], side[:, 0])) / np.linalg.norm(side, axis=1)[:, None]
    mid = (corners[starts] + corners[ends]) / 2
    normal *= np.sign(np.sum((mid - middle) * normal, axis=1))[:, None]
    top, base = (starts >= 4) & (ends >= 4), (starts < 4) & (ends < 4)
    pick = {"top": top, "base": base, "uprights": ~top & ~base, "all": np.ones(len(starts), bool)}[kind]
    moved = found + px * normal * pick[:, None]
    return Sighting(s.x, s.y, s.pixel, s.frame_size, s.turn_deg, (starts, ends, moved))


base_n = camera(cubes)
base_p4 = HANDEDNESS @ refine._seen(refine._pinhole(base_n).matrix, p4)
print(f"as found: f {base_n[0]:.0f}, aim ({base_n[1]:.0f}, {base_n[2]:.0f}); point 4 {np.linalg.norm(base_p4 - (p4.x, p4.y)):.2f} mm off")
for kind in ("top", "base", "uprights", "all"):
    for px in (1.0, -1.0):
        n = camera([pushed(s, kind, px) for s in cubes])
        q = HANDEDNESS @ refine._seen(refine._pinhole(n).matrix, p4)
        print(f"  {kind:8} {px:+.0f} px: f {n[0]:5.0f} ({n[0]-base_n[0]:+4.0f}), aim ({n[1]:5.0f}, {n[2]:5.0f}) "
              f"({n[1]-base_n[1]:+4.0f}, {n[2]-base_n[2]:+4.0f}); point 4 moves {np.linalg.norm(q - base_p4):.2f} mm")
