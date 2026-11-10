#!/usr/bin/env python3
"""Pretend calibrations from three of the recorded points, and what the cubes' edges say about the arm's turn
correction (raw.py's bearing wave, from gecko). No arm, no app: reads config/ as it is.

Every point's true x, y is taken back to the firmware's (raw.py, exact inverse) and made true again with another
bearing wave A sin 2b + B (cos 2b - 1) degrees and another reach term at cube top (+dR mm); the library's is
A -1.36, B +0.33, dR 0.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/pretend.py [refine-points.json]
"""
import dataclasses, itertools, json, math, sys
from pathlib import Path

import numpy as np

from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares, _huber_cost
from applib.camera.store import RefinePoint
from camera import refine
from maxarm import raw

CONFIG = Path("config")
POINTS = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG / "camera-refine-points.json"
LIBRARY = (-1.36, 0.33, 0.0)


def load(camera_file, points_file):
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"]]
    return fit, [p for p in stored if p.sighting.edges is not None]


def remade(point, model):
    """The point's reading made true with another bearing wave and reach term."""
    a, b, dr = model
    rx, ry, _ = raw.raw_position((point.x, point.y, point.z + raw.DESK_Z_MM))
    r, bearing = math.hypot(rx, ry), math.atan2(rx, -ry)
    r += raw._further_out(max(point.z, 0.0)) + dr
    bearing += math.radians(a * math.sin(2 * bearing) + b * (math.cos(2 * bearing) - 1))
    return r * math.sin(bearing), -r * math.cos(bearing)


def sightings(stored, model):
    return [dataclasses.replace(p.sighting, x=x, y=y) for p in stored for x, y in [remade(p, model)]]


def fit_camera(frame_size, chosen, start, iterations=200):
    on_arm = refine._OnArm(frame_size, chosen, start)
    numbers, residuals, _ = least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS,
                                          huber=EDGE_HUBER_PX, iterations=iterations, derivatives=on_arm.derivatives)
    return numbers, _huber_cost(residuals[:-2], EDGE_HUBER_PX), len(residuals) - 2


def miss(matrix, s):
    """The cube placed by its own edges through the camera, against the reading: desk mm, (dx, dy)."""
    return HANDEDNESS @ refine._seen(matrix, s) - (s.x, s.y)


def polar(d, s):
    """A miss split into out from the base and round it (towards +bearing), mm."""
    r = math.hypot(s.x, s.y)
    out = np.array((s.x, s.y)) / r
    round_ = np.array((-out[1], out[0]))                  # +bearing: towards +x from straight out (-y)
    return float(d @ out), float(d @ round_)


def describe(numbers):
    p = refine._pinhole(numbers)
    pos = HANDEDNESS @ p.position[:2]
    return (f"f {numbers[0]:6.1f}  centre ({numbers[1]:6.1f}, {numbers[2]:5.1f})  "
            f"camera at desk ({pos[0]:6.1f}, {pos[1]:6.1f}) {p.position[2]:5.1f} up")


def main():
    fit, stored = load(CONFIG / "camera.json", POINTS)
    start = fit.cube_fit().true_to_pixel
    n = len(stored)
    print(f"{POINTS}: {n} points")
    for p in stored:
        print(f"  #{p.id} ({p.x:7.2f}, {p.y:7.2f}) z {p.z:5.2f}")

    models = [LIBRARY, (0.0, 0.0, 0.0)]
    # The edges' own verdict: all points pinned, the wave's sin 2b amplitude swept, cos term held at the library's.
    print("\nAll points pinned, the camera fitted to all their edges; edge cost (Huber, px^2) by the wave's A:")
    sweep = []
    for a in np.arange(-2.5, 0.51, 0.25):
        model = (float(a), LIBRARY[1], 0.0)
        s = sightings(stored, model)
        numbers, cost, count = fit_camera(fit.frame_size, s, start)
        m = refine._pinhole(numbers).matrix
        offs = [float(np.linalg.norm(miss(m, x))) for x in s]
        sweep.append((cost, a))
        print(f"  A {a:+5.2f}: cost {cost:8.2f} over {count} edge points; misses "
              + " ".join(f"{o:4.2f}" for o in offs) + f"   {describe(numbers)}")
    best_a = min(sweep)[1]
    models.append((float(best_a), LIBRARY[1], 0.0))

    for model in models:
        s = sightings(stored, model)
        numbers, cost, _ = fit_camera(fit.frame_size, s, start)
        m = refine._pinhole(numbers).matrix
        print(f"\n=== wave A {model[0]:+.2f} sin 2b, B {model[1]:+.2f}, reach {model[2]:+.2f}: all {n} pinned, "
              f"cost {cost:.2f}; {describe(numbers)}")
        for k, x in enumerate(s):
            d = miss(m, x)
            o, r = polar(d, x)
            print(f"  #{stored[k].id} miss {np.linalg.norm(d):4.2f} mm: out {o:+5.2f}, round {r:+5.2f}")
        rows = []
        for triple in itertools.combinations(range(n), 3):
            chosen = [s[k] for k in triple]
            try:
                numbers, _, _ = fit_camera(fit.frame_size, chosen, start)
            except np.linalg.LinAlgError:
                continue
            m = refine._pinhole(numbers).matrix
            rest = {k: float(np.linalg.norm(miss(m, s[k]))) for k in range(n) if k not in triple}
            rows.append((triple, rest, numbers))
        print(f"  every three predicting the rest ({len(rows)} triples):")
        for triple, rest, numbers in rows:
            names = "+".join(str(stored[k].id) for k in triple)
            print(f"    {names:7} -> " + "  ".join(f"#{stored[k].id} {v:4.2f}" for k, v in rest.items())
                  + f"   {describe(numbers)}")
        every = [v for _, rest, _ in rows for v in rest.values()]
        print(f"  predicted: median {np.median(every):.2f}, 90% {np.percentile(every, 90):.2f}, worst {max(every):.2f}")


if __name__ == "__main__":
    main()
