#!/usr/bin/env python3
"""Where the cubes put the principal point, session by session: each session's three CALIBRATE cubes refitted with
SAVE's fit, the principal point free (unheld) and held as the app holds it; then each cube's edges by kind left out
in turn. The principal point belongs to the camera's mode and focus, not to where the camera stands.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/centres.py
"""
import json
from pathlib import Path

import numpy as np

from applib.camera.fit import EDGE_HUBER_PX, CameraFit
from applib.camera.model import least_squares
from applib.camera.sighting import Sighting
from applib.camera.store import RefinePoint
from camera import refine

SESSIONS = {"4 Nov": ("../work/config-jackal-1104/camera.json", "../work/config-jackal-1104/camera-refine-points.json"),
            "5 Nov": ("../work/config-jackal-1105/camera.json", "../work/config-jackal-1105/camera-refine-points.json"),
            "9 Nov": ("/tmp/c1110/camera.json", "/tmp/c1110/camera-refine-points-previous.json"),
            "10 Nov": ("/tmp/c1110/camera.json", "/tmp/c1110/camera-refine-points.json")}


class Unheld(refine._OnArm):
    def held(self, numbers):
        return np.zeros(2)

    def held_rows(self, count):
        return np.zeros((2, count))


def fitted(cls, frame_size, sightings, first):
    on_arm = cls(frame_size, sightings, first)
    return least_squares(on_arm.residuals, refine._numbers(first), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                         iterations=300, derivatives=on_arm.derivatives)[0]


def without(s: Sighting, kind: str, turn_corners) -> Sighting:
    starts, ends, found = s.edges
    top = (starts >= 4) & (ends >= 4)
    base = (starts < 4) & (ends < 4)
    upright = ~top & ~base
    drop = {"top": top, "base": base, "uprights": upright}[kind]
    return Sighting(s.x, s.y, s.pixel, s.frame_size, s.turn_deg, (starts[~drop], ends[~drop], found[~drop]))


for name, (camera_file, points_file) in SESSIONS.items():
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [p for p in json.loads(Path(points_file).read_text())["points"] if p.get("cube")]
    cubes = [RefinePoint(**p).sighting for p in stored]
    first = fit.cube_fit().true_to_pixel
    held = fitted(refine._OnArm, fit.frame_size, cubes, first)
    free = fitted(Unheld, fit.frame_size, cubes, refine._pinhole(held).matrix)
    print(f"{name}: held f {held[0]:.0f} centre ({held[1]:.0f}, {held[2]:.0f});  unheld f {free[0]:.0f} centre "
          f"({free[1]:.0f}, {free[2]:.0f}); camera at {np.round(refine._pinhole(free).position * [-1, 1, 1], 0).tolist()}")
    for index, colour in enumerate(p["cube"] for p in stored):
        for kind in ("top", "base", "uprights"):
            trial = list(cubes)
            trial[index] = without(cubes[index], kind, None)
            n = fitted(Unheld, fit.frame_size, trial, refine._pinhole(free).matrix)
            print(f"   {colour:5} without its {kind:8}: f {n[0]:5.0f} centre ({n[1]:5.0f}, {n[2]:5.0f})")
