#!/usr/bin/env python3
"""The 17:30 CALIBRATE (config/: its frame and the three readings, taken without the turn fix) run again from
scratch with the readings made true with the turn fix as it was, with only its big part, and as recorded (none).
The same servo positions each time; only how they are made true differs. Nothing written.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/recalibrate-now.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np
from maxarm import raw
from applib.camera.fit import fit_camera

CONFIG = Path("config")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
cubes = json.loads((CONFIG / "camera-points.json").read_text())["cubes"]
assert raw.WAVE_SIN_DEG == 0.0 and raw.WAVE_COS_DEG == 0.0          # the readings were made true without it


def remade(p, turn):
    rx, ry, _ = raw.raw_position((p["x"], p["y"], p["z"] + raw.DESK_Z_MM))     # the servos' own (no turn fix now)
    r, b = math.hypot(rx, ry), math.atan2(rx, -ry)
    r += raw._further_out(max(p["z"], 0.0))
    b += math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return r * math.sin(b), -r * math.cos(b)


for name, turn in (("as recorded, no turn fix", (0.0, 0.0)), ("turn fix as it was (-1.36, +0.33)", (-1.36, 0.33)),
                   ("big part only (-1.36, 0)", (-1.36, 0.0))):
    points = {c: remade(p, turn) for c, p in cubes.items()}
    fit = fit_camera(frame, points)
    print(f"{name:36}: edges {fit.edge_px[0]:.2f} px typically, {fit.edge_px[1]:.2f} at 90%; cubes from their "
          f"readings " + ", ".join(f"{c} {fit.cube_off_mm(c):.2f}" for c in fit.cubes) + f" mm; focal {fit.focal:.0f}")
