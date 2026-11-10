#!/usr/bin/env python3
"""The 3-D fit (all points at their readings, every three predicting the rest, CALIBRATE's three predicting the
others) with only some of each cube's edges: all; the top's four edges and the uprights (no base); the top's
four only. The blue cube's top at 40 or 41.5 mm.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/tops-3d.py
"""
import dataclasses, itertools, json, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
import blue as b

KEEP = {"all edges": lambda a, c: True, "no base edges": lambda a, c: max(a, c) >= 4,
        "top edges only": lambda a, c: min(a, c) >= 4}


def only(sight, keep):
    starts, ends, found = sight.edges
    sel = np.array([keep(a, c) for a, c in zip(starts, ends)])
    return dataclasses.replace(sight, edges=(starts[sel], ends[sel], found[sel]))


for name in ("10 Nov", "5 Nov"):
    camera_file, points_file = b.SESSIONS[name]
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    start = fit.cube_fit().true_to_pixel
    calib = [k for k, p in enumerate(stored) if p.cube]
    print(f"\n=== {name}")
    for label, keep in KEEP.items():
        sights = [only(p.sighting, keep) for p in stored]
        for top in (40.0, 41.5):
            sizes = [(40.0, top) if p.cube == "blue" else (40.0, 40.0) for p in stored]
            m6, _, n6 = b.camera(fit.frame_size, sights, sizes, start)
            all6 = [float(np.linalg.norm(b.placed_miss(m6, s, z))) for s, z in zip(sights, sizes)]
            m3, _, n3 = b.camera(fit.frame_size, [sights[k] for k in calib], [sizes[k] for k in calib], start)
            cal = [float(np.linalg.norm(b.placed_miss(m3, sights[k], sizes[k]))) for k in range(len(stored))
                   if k not in calib]
            preds = []
            for three in itertools.combinations(range(len(stored)), 3):
                try:
                    mt, _, _ = b.camera(fit.frame_size, [sights[k] for k in three], [sizes[k] for k in three], start)
                except np.linalg.LinAlgError:
                    continue
                preds += [float(np.linalg.norm(b.placed_miss(mt, sights[k], sizes[k])))
                          for k in range(len(stored)) if k not in three]
            print(f"  {label:15} blue {top:4.1f}: all-points camera misses " + " ".join(f"{v:4.2f}" for v in all6)
                  + f" (f {n6[0]:.0f}, centre {n6[1]:.0f}, {n6[2]:.0f}) | CALIBRATE's three -> "
                  + " ".join(f"{v:4.2f}" for v in cal) + f" (f {n3[0]:.0f}, centre {n3[1]:.0f}, {n3[2]:.0f})"
                  + f" | every three: median {np.median(preds):4.2f}, 90% {np.percentile(preds, 90):4.2f}, "
                    f"worst {max(preds):4.2f}")
