#!/usr/bin/env python3
"""Can the data do without the turn fix? Every turn fix off, and in its place one other thing that could make the
same pattern, free: the base's angles scaled; the arm's x and y stretched against each other (or the measuring
tool's); the reach scaled and offset; for the camera, its pixels not square. Two data sets: GRID-PAPER-jackal's 25
desk points (the arm's readings against the paper, laid on by a turn and shift) and today's six cubes (all at their
readings, the camera refitted to their edges). The library's turn fix and its big part alone for comparison.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/nullify.py
"""
import dataclasses, json, math, sys
from pathlib import Path
import numpy as np

sys.argv = sys.argv[:1]
from maxarm import raw
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
import blue as b

ns = {}
exec(compile(Path("agenttools/fitcheck/paper-now.py").read_text().split('print(f"{len(rows)}')[0], "paper-now", "exec"), ns)
rigid, firmware, paper = ns["rigid"], ns["firmware"], ns["paper"]


def remake(points, turn=(0.0, 0.0), angle_scale=0.0, stretch=0.0, reach_scale=1.0, reach_add=0.0):
    """Firmware x, y, z rows (true z from raw.py) made true: today's reach term, then the alternative."""
    out = []
    for x, y, z in points:
        h = max(raw.true_position((x, y, z))[2] - raw.DESK_Z_MM, 0.0)
        r, bb = math.hypot(x, y), math.atan2(x, -y)
        r = reach_scale * (r + raw._further_out(h)) + reach_add
        bb = bb * (1.0 + angle_scale) + math.radians(turn[0] * math.sin(2 * bb) + turn[1] * (math.cos(2 * bb) - 1))
        px, py = r * math.sin(bb), -r * math.cos(bb)
        out.append((px * (1.0 + stretch), py * (1.0 - stretch)))
    return np.array(out)


def paper_fit(xy):
    left, _ = rigid(xy, paper)
    d = np.linalg.norm(left, axis=1)
    return float(np.median(d)), float(d.max())


def best(score, grid):
    return min((score(v), v) for v in grid)


print("Grid paper, 25 desk points (3 Nov), each model laid on the paper by a turn and shift: typical / worst mm")
models = {
    "library turn fix (-1.36, +0.33)": lambda v: remake(firmware, turn=(-1.36, 0.33)),
    "big part alone (-1.36, 0)": lambda v: remake(firmware, turn=(-1.36, 0.0)),
    "no turn fix": lambda v: remake(firmware),
}
for name, make in models.items():
    t, w = paper_fit(make(None))
    print(f"  {name:52} {t:4.2f} / {w:4.2f}")
alternatives = {
    "no turn fix, the base's angles scaled by": (lambda v: remake(firmware, angle_scale=v), np.arange(-0.08, 0.021, 0.0025)),
    "no turn fix, x against y stretched by": (lambda v: remake(firmware, stretch=v), np.arange(-0.06, 0.061, 0.0025)),
    "no turn fix, the reach scaled by": (lambda v: remake(firmware, reach_scale=1 + v), np.arange(-0.06, 0.061, 0.0025)),
    "no turn fix, the reach offset by (mm)": (lambda v: remake(firmware, reach_add=v), np.arange(-12.0, 12.1, 0.5)),
}
for name, (make, grid) in alternatives.items():
    (t, v) = best(lambda v: paper_fit(make(v))[0], grid)
    print(f"  {name:43} {v:+8.4f}: {t:4.2f} / {paper_fit(make(v))[1]:4.2f}")

print("\nToday's six cubes, all at their readings, the camera refitted to their edges: edge cost (px^2)")
fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = [RefinePoint(**p) for p in json.loads(Path("config/camera-refine-points.json").read_text())["points"]]
fw = [raw.raw_position((p.x, p.y, p.z + raw.DESK_Z_MM)) for p in stored]
alone = fit.cube_fit().true_to_pixel
sizes = [(40.0, 40.0)] * 6


def cubes_cost(xy, aspect=1.0):
    sights = []
    for p, (x, y) in zip(stored, xy):
        s = dataclasses.replace(p.sighting, x=float(x), y=float(y))
        if aspect != 1.0:
            starts, ends, found = s.edges
            found = found.copy()
            found[:, 1] = fit.frame_size[1] / 2 + (found[:, 1] - fit.frame_size[1] / 2) / aspect
            s = dataclasses.replace(s, edges=(starts, ends, found))
        sights.append(s)
    return b.camera(fit.frame_size, sights, sizes, alone)[1]


for name, make in (("library turn fix (-1.36, +0.33)", lambda: remake(fw, turn=(-1.36, 0.33))),
                   ("big part alone (-1.5, 0)", lambda: remake(fw, turn=(-1.5, 0.0))),
                   ("no turn fix", lambda: remake(fw))):
    print(f"  {name:52} {cubes_cost(make()):7.0f}")
for name, make, grid in (("no turn fix, the base's angles scaled by", lambda v: remake(fw, angle_scale=v),
                          np.arange(-0.05, 0.011, 0.005)),
                         ("no turn fix, x against y stretched by", lambda v: remake(fw, stretch=v),
                          np.arange(-0.05, 0.051, 0.005)),
                         ("no turn fix, the reach scaled by", lambda v: remake(fw, reach_scale=1 + v),
                          np.arange(-0.04, 0.041, 0.005))):
    c, v = best(lambda v: cubes_cost(make(v)), grid)
    print(f"  {name:43} {v:+8.4f}: {c:7.0f}")
c, v = best(lambda v: cubes_cost(remake(fw), aspect=v), np.arange(0.96, 1.041, 0.005))
print(f"  {'no turn fix, the pixels taller than wide by':43} {v:8.4f}: {c:7.0f}   (the owner's jar lid: 672/673)")
