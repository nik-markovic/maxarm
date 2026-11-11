#!/usr/bin/env python3
"""Today's calibration done again as the app does it, with the turn fix as it is, its small part off, or off
altogether -- the reach (distance) formula kept. Each reading, camera page and Fine Tune alike, is taken back to
the servos' own numbers (raw.py, exact inverse) and made true again that way. CALIBRATE runs from scratch on its
frame (config/camera-frame.png, 10:31: fit.fit_camera, every step), then the Fine Tune page's table as it shows it
(tests/refine.describe) with 4, 5 and 6 points. No arm, no app; nothing written.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/recalibrate.py
"""
import dataclasses, json, math
from pathlib import Path

import cv2
import numpy as np

from maxarm import raw
from applib.camera import store
from applib.camera.fit import fit_camera
from applib.camera.store import CubePoint, RefinePoint
from tests import refine as table

CONFIG = Path("config")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
recorded = {colour: CubePoint(**{k: v for k, v in point.items()})
            for colour, point in json.loads((CONFIG / "camera-points.json").read_text())["cubes"].items()}
stored = [RefinePoint(**p) for p in json.loads((CONFIG / "camera-refine-points.json").read_text())["points"]]


def remade(x, y, z, turn):
    """A reading (desk = true x, y; z over the desk) made true with `turn` (degrees: big part, small part)."""
    rx, ry, _ = raw.raw_position((x, y, z + raw.DESK_Z_MM))
    h = max(z, 0.0)
    r, b = math.hypot(rx, ry), math.atan2(rx, -ry)
    r += raw._further_out(h)                                   # the reach (distance) formula, as it is
    b += math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return round(r * math.sin(b), 2) + 0.0, round(-r * math.cos(b), 2) + 0.0


for name, turn in (("turn fix as it is (-1.36, +0.33)", (-1.36, 0.33)), ("small part off (-1.36, 0)", (-1.36, 0.0)),
                   ("turn fix off", (0.0, 0.0))):
    cubes = {c: dataclasses.replace(p, **dict(zip("xy", remade(p.x, p.y, p.z, turn)))) for c, p in recorded.items()}
    fit = fit_camera(frame, {c: (p.x, p.y) for c, p in cubes.items()})
    first = store.cube_refine_points(fit, cubes, "10 Nov 10:31")
    rest = [dataclasses.replace(p, **dict(zip("xy", remade(p.x, p.y, p.z, turn)))) for p in stored[3:]]
    points = first + rest
    lens = dict(zip(("f", "cx", "cy"), np.round([fit.focal, *cv2.decomposeProjectionMatrix(fit.true_to_pixel)[0][:2, 2]
                                                  / cv2.decomposeProjectionMatrix(fit.true_to_pixel)[0][2, 2]], 0)))
    print(f"\n=== {name}: CALIBRATE again on its frame: focal {lens['f']:.0f} px, aim ({lens['cx']:.0f}, "
          f"{lens['cy']:.0f}); its cubes' own misses " + " ".join(
              f"{c} {fit.cube_off_mm(c):.2f}" for c in fit.cubes))
    for count in (4, 5, 6):
        shown = table.describe(fit, points[:count])
        cells = []
        for row in shown["points"]:
            cells.append(f"#{row['id']} {row['off']['calibrate']:4.2f}/{row['off']['after'] if row['off']['after'] is not None else '-':>4}"
                         f" Z {row['height']['calibrate'] if row['height']['calibrate'] is not None else '-':>5}")
        print(f"  {count} points (XY at CALIBRATE / after SAVE, Z at CALIBRATE): " + "  ".join(cells)
              + (f"  [{shown['problem']}]" if shown["problem"] else ""))
