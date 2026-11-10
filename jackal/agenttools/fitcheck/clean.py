#!/usr/bin/env python3
"""Two things together, per session: the blue cube 40 mm or 41.5 mm tall, and each cube's ragged edges -- an edge
whose found points do not lie on one straight line within MAX_RMS px, so not only the cube's edge -- kept or
dropped. CALIBRATE's three cubes refitted so, each other point's miss through that camera (the page's 'XY at
CALIBRATE'); CALIBRATE's own three placed by their edges through it; and every three predicting the rest.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/clean.py
"""
import dataclasses, itertools, json, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
from tops import line

sys.argv = sys.argv[:1]
import blue as b

MAX_RMS = 1.0


def cleaned(sight):
    starts, ends, found = sight.edges
    keep = np.ones(len(found), bool)
    for a, c in set(zip(starts.tolist(), ends.tolist())):
        sel = (starts == a) & (ends == c)
        if sel.sum() < 5 or line(found[sel])[2] > MAX_RMS:
            keep &= ~sel
    return dataclasses.replace(sight, edges=(starts[keep], ends[keep], found[keep])), int((~keep).sum())


for name, (camera_file, points_file) in b.SESSIONS.items():
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    start = fit.cube_fit().true_to_pixel
    calib = [k for k, p in enumerate(stored) if p.cube]
    print(f"\n=== {name}")
    for drop in (False, True):
        sights = [p.sighting for p in stored]
        dropped = [0] * len(sights)
        if drop:
            sights, dropped = zip(*[cleaned(s) for s in sights])
        for top in (40.0, 41.5):
            sizes = [(40.0, top) if p.cube == "blue" else (40.0, 40.0) for p in stored]
            m3, _, n3 = b.camera(fit.frame_size, [sights[k] for k in calib], [sizes[k] for k in calib], start)
            offs = [float(np.linalg.norm(b.placed_miss(m3, sights[k], sizes[k]))) for k in range(len(stored))]
            preds = []
            triples = list(itertools.combinations(range(len(stored)), 3))
            for three in triples[::max(1, len(triples) // 40)]:
                try:
                    mt, _, _ = b.camera(fit.frame_size, [sights[k] for k in three], [sizes[k] for k in three], start)
                except np.linalg.LinAlgError:
                    continue
                preds += [float(np.linalg.norm(b.placed_miss(mt, sights[k], sizes[k])))
                          for k in range(len(stored)) if k not in three]
            label = f"{'ragged edges dropped' if drop else 'all edges':20} blue {top:4.1f}"
            print(f"  {label}: CALIBRATE f {n3[0]:.0f} centre ({n3[1]:.0f}, {n3[2]:.0f}); its own three "
                  + " ".join(f"#{stored[k].id}:{offs[k]:4.2f}" for k in calib) + " | the others "
                  + " ".join(f"#{stored[k].id}:{offs[k]:4.2f}" for k in range(len(stored)) if k not in calib)
                  + (f" | every three: median {np.median(preds):4.2f}, 90% {np.percentile(preds, 90):4.2f}, "
                     f"worst {max(preds):4.2f}" if preds else ""))
        if drop:
            print("    dropped edge points: " + " ".join(f"#{p.id}:{d}" for p, d in zip(stored, dropped)))
