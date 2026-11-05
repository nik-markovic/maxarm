#!/usr/bin/env python3
"""One camera dimension: the camera fitted to the points' cube edges alone (40 mm cubes, each placed freely) with
its pixels' aspect fy/fx held at a value, or a bend along one frame axis only, x' = x (1 + k x^2) in the lens's
units; then the arm's readings laid on the cubes by a turn and a shift. Both rigs share the camera and its mode.

    one-dimension.py [config_dir ...]       # default the 1104 copy and impala's
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import edges, store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from applib.camera.model import CUBE_MM, cube_corners  # noqa: E402
from applib.camera.sighting import start  # noqa: E402
from camera import refine  # noqa: E402

NAMES = ("focal", "aspect", "cx", "cy", "rx", "ry", "rz", "tx", "ty", "tz", "kx", "ky")


def project(camera, xyz):
    focal, aspect, cx, cy, rx, ry, rz, tx, ty, tz, kx, ky = camera
    seen = xyz @ cv2.Rodrigues(np.array([rx, ry, rz]))[0].T + (tx, ty, tz)
    u, v = seen[:, 0] / seen[:, 2], seen[:, 1] / seen[:, 2]
    return np.column_stack((focal * u * (1 + kx * u * u) + cx, focal * aspect * v * (1 + ky * v * v) + cy))


def solve(residuals, x, free, iterations=40):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = residuals(x)
        w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for c, i in enumerate(free):
            h = 1e-6 * max(1.0, abs(x[i]))
            xx = x.copy(); xx[i] += h
            J[:, c] = (residuals(xx) - r) / h
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -(J.T @ (w * r)), rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return x


class Rig:
    def __init__(self, config: Path):
        fit = store.load_camera(config)
        stored = [p for p in store.load_refine_points(config, fit) if p.edges]
        self.name, self.ids = config.name, [p.id for p in stored]
        self.points = [p.sighting for p in stored]
        self.arm = np.array([(p.x, p.y) for p in self.points])
        self.width, self.height = fit.frame_size
        self.first = refine._numbers(fit.true_to_pixel)
        self.starts = [start(fit.true_to_pixel, p)[:3] for p in self.points]

    def fit(self, held: dict):
        camera = np.array([self.first[0], 1.0, *self.first[1:9], 0.0, 0.0])
        for name, value in held.items():
            camera[NAMES.index(name)] = value
        x = np.concatenate([camera, np.concatenate(self.starts)])
        free = [i for i, n in enumerate(NAMES) if n not in held and n not in ("aspect", "kx", "ky")]
        free += list(range(12, 12 + 3 * len(self.points)))

        def residuals(x):
            parts = [edges.signed_distances(project(x[:12], cube_corners(x[12 + 3 * i: 14 + 3 * i],
                                                                          x[14 + 3 * i])), *p.edges)
                     for i, p in enumerate(self.points)]
            parts.append(np.array([x[2] - self.width / 2, x[3] - self.height / 2]) / (0.1 * self.width))
            return np.concatenate(parts)

        x = solve(residuals, x, free)
        r = residuals(x)[:-2]
        cubes = x[12:].reshape(-1, 3)[:, :2]
        return x[:12], cubes, float(np.sqrt(np.mean(np.minimum(np.abs(r), 3) ** 2)))

    def arm_misses(self, cubes):
        turn, shift, mirror = link_arm(self.arm, cubes)
        back = np.linalg.solve(turn @ mirror, (cubes - shift).T).T
        rows = []
        for a, c in zip(self.arm, back):
            b = math.atan2(a[0], -a[1])
            d = c - a
            rows.append((d @ (math.sin(b), -math.cos(b)), d @ (math.cos(b), math.sin(b))))
        lo = []
        for i in range(len(self.arm)):
            rest = [j for j in range(len(self.arm)) if j != i]
            t, s, m = link_arm(self.arm[rest], cubes[rest])
            lo.append(float(np.linalg.norm(np.linalg.solve(t @ m, cubes[i] - s) - self.arm[i])))
        return np.array(rows), np.array(lo)


def main():
    configs = [Path(a) for a in sys.argv[1:]] or [Path("/tmp/c1104"), Path("/tmp/cimp")]
    rigs = [Rig(c) for c in configs]
    trials = [("square pixels (the app's)", {})]
    trials += [(f"aspect fy/fx {a:.4f}", {"aspect": a}) for a in (0.97, 0.98, 0.99, 1.01, 1.0222, 1.03, 1.04)]
    trials += [(f"bend along frame x, k {k:+.2f}", {"kx": k}) for k in (-0.10, -0.05, 0.05, 0.10)]
    trials += [(f"bend along frame y, k {k:+.2f}", {"ky": k}) for k in (-0.10, -0.05, 0.05, 0.10)]
    for name, held in trials:
        cells = []
        for rig in rigs:
            camera, cubes, edge = rig.fit(held)
            rows, lo = rig.arm_misses(cubes)
            cells.append(f"{rig.name}: edges {edge:.3f} px, f {camera[0]:.0f}, c ({camera[2]:.0f}, {camera[3]:.0f}); "
                         f"left out {np.median(lo):.2f} typ {lo.max():.2f} worst, mean out {rows[:, 0].mean():+.2f}")
        print(f"{name:30s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
