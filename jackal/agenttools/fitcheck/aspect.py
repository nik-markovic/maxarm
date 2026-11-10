#!/usr/bin/env python3
"""Pixels not square: every point's edges (and pixel) scaled in picture y by s before the square-pixel fit -- the
same as a camera whose pixels are s times taller than wide. The three cubes' camera predicting point 4; all four
points' camera: its misses and the aim. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/aspect.py camera.json points.json
"""
import json, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.sighting import Sighting
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
pts = [RefinePoint(**p).sighting for p in json.loads(Path(sys.argv[2]).read_text())["points"]]
first = fit.cube_fit().true_to_pixel
middle_y = fit.frame_size[1] / 2


def scaled(s, k):
    st, en, f = s.edges
    f = f.copy(); f[:, 1] = middle_y + (f[:, 1] - middle_y) * k
    return Sighting(s.x, s.y, (s.pixel[0], middle_y + (s.pixel[1] - middle_y) * k), s.frame_size, s.turn_deg, (st, en, f))


def camera(sightings, start):
    on_arm = refine._OnArm(fit.frame_size, sightings, start)
    return least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                         iterations=300, derivatives=on_arm.derivatives)[0]


for k in (0.98, 0.99, 1.0, 1.01, 1.02, 1.03):
    s = [scaled(p, k) for p in pts]
    n3 = camera(s[:3], first)
    m3 = refine._pinhole(n3).matrix
    miss4 = np.linalg.norm(HANDEDNESS @ refine._seen(m3, s[3]) - (s[3].x, s[3].y))
    n4 = camera(s, m3)
    m4 = refine._pinhole(n4).matrix
    misses = [np.linalg.norm(HANDEDNESS @ refine._seen(m4, p) - (p.x, p.y)) for p in s]
    print(f"pixels {k:.2f} tall: 3 cubes -> point 4 {miss4:4.2f} mm (f {n3[0]:.0f}, aim {n3[1]:.0f},{n3[2]:.0f}); "
          f"4 points: misses {' '.join(f'{v:4.2f}' for v in misses)} (aim {n4[1]:.0f},{n4[2]:.0f})")
