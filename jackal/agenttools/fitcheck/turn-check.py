#!/usr/bin/env python3
"""Is the bearing correction (raw.py's turn term) what today's six points want, and is the test fair? Each point's
servo reading is taken back to the firmware's own (raw.py, exact inverse), made true again with another turn term
A sin 2b + B (cos 2b - 1), and the camera fitted afresh to all six cubes' edges at those positions. The edges are
the frame's (research.py: they do not follow the camera they were searched through). Fitted from three starting
cameras -- CALIBRATE's, one with the focal length 8% shorter, one 8% longer -- to see that the fit lands the same.
With the blue cube 40 mm and with its top at 41.5.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/turn-check.py
"""
import dataclasses, itertools, json, math, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
from camera import refine
from pretend import remade
import blue as b

fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = [RefinePoint(**p) for p in json.loads(Path("config/camera-refine-points.json").read_text())["points"]]
alone = fit.cube_fit().true_to_pixel


def start_with_focal(scale):
    n = refine._numbers(alone)
    n[0] *= scale
    n[6:9] *= scale                       # the camera moved back along its axis, the picture kept about the same
    return refine._pinhole(n).matrix


starts = {"CALIBRATE's": alone, "focal -8%": start_with_focal(0.92), "focal +8%": start_with_focal(1.08)}
for top in (40.0, 41.5):
    sizes = [(40.0, top) if p.cube == "blue" else (40.0, 40.0) for p in stored]
    print(f"\n=== blue's top {top} mm: edge cost of all six at their readings, by the turn term")
    for a, bb in ((-1.36, 0.33), (0.0, 0.0), (-1.0, 0.33), (-1.75, 0.33), (-1.5, 0.0), (-1.5, 0.65), (-2.0, 0.33)):
        sights = [dataclasses.replace(p.sighting, x=x, y=y) for p in stored for x, y in [remade(p, (a, bb, 0.0))]]
        costs = []
        for name, first in starts.items():
            m, cost, n = b.camera(fit.frame_size, sights, sizes, first)
            costs.append((cost, n[0]))
        m, cost, n = b.camera(fit.frame_size, sights, sizes, alone)
        offs = [float(np.linalg.norm(b.placed_miss(m, s, z))) for s, z in zip(sights, sizes)]
        preds = []
        for three in itertools.combinations(range(6), 3):
            mt, _, _ = b.camera(fit.frame_size, [sights[k] for k in three], [sizes[k] for k in three], alone)
            preds += [float(np.linalg.norm(b.placed_miss(mt, sights[k], sizes[k]))) for k in range(6) if k not in three]
        label = "library" if (a, bb) == (-1.36, 0.33) else "none" if (a, bb) == (0.0, 0.0) else ""
        print(f"  A {a:+5.2f} B {bb:+5.2f} {label:8}: cost from the three starts "
              + " / ".join(f"{c:7.1f} (f {f:4.0f})" for c, f in costs)
              + f" | misses " + " ".join(f"{o:4.2f}" for o in offs)
              + f" | every three: median {np.median(preds):4.2f}, worst {max(preds):4.2f}")
