#!/usr/bin/env python3
"""Is the blue (or green) cube a different size from the red one? Every session's points, all at their readings,
the camera fitted to all their edges (refine._OnArm), with the blue and green cubes' heights over the desk -- and
then their whole size -- held at other values than 40 mm; the red cube stays 40. Edge cost and each point's miss
(its cube placed by its own edges through that camera, the same size, against the reading); then every three points
pretending to calibrate, predicting the rest.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/blue.py
"""
import itertools, json, math
from pathlib import Path

import numpy as np

from applib.camera import edges as edge_mod
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import _huber_cost, least_squares, project
from applib.camera.store import RefinePoint
from applib.camera.sighting import STEPS as CUBE_STEPS
from camera import refine

SESSIONS = {
    "10 Nov": ("config/camera.json", "config/camera-refine-points.json"),
    "9-10 Nov": ("config/camera.json", "config/camera-refine-points-previous.json"),
    "5 Nov": ("../work/config-jackal-1105/camera.json", "../work/config-jackal-1105/camera-refine-points.json"),
    "4 Nov": ("../work/config-jackal-1104/camera.json", "../work/config-jackal-1104/camera-refine-points.json"),
}
SQUARE = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))


def corners(centre, turn, side, height):
    c, s = math.cos(turn), math.sin(turn)
    return np.array([(centre[0] + side * (c * dx - s * dy), centre[1] + side * (s * dx + c * dy), z)
                     for z in (0.0, height) for dx, dy in SQUARE])


def misses(matrix, sight, centre, turn, side, height):
    starts, ends, found = sight.edges
    return edge_mod.signed_distances(project(matrix, corners(centre, turn, side, height)), starts, ends, found)


class Sized(refine._OnArm):
    """refine._OnArm with each point's cube its own size: (side, height) per point."""

    def __init__(self, frame_size, points, first, sizes):
        self.sizes = list(sizes)
        super().__init__(frame_size, points, first)

    def blocks(self, numbers):
        self._turn(numbers)
        return [misses(self._view(numbers, p), p, self.centres[k], self.turns[k], *self.sizes[k])
                for k, p in enumerate(self.points)]

    def rows(self, numbers):
        self._turn(numbers)
        out = []
        for k, p in enumerate(self.points):
            def edge_misses(camera_numbers, turn, k=k, p=p):
                return misses(self._view(camera_numbers, p), p, self.centres[k], turn, *self.sizes[k])
            now = edge_misses(numbers, self.turns[k])
            w = np.where(np.abs(now) <= EDGE_HUBER_PX, 1.0, EDGE_HUBER_PX / np.maximum(np.abs(now), 1e-12))[:, None]
            by_turn = ((edge_misses(numbers, self.turns[k] + CUBE_STEPS[2]) - now) / CUBE_STEPS[2])[:, None]
            by_camera = refine._differences(lambda t: edge_misses(t, self.turns[k]), numbers, refine.CAMERA_STEPS)
            moves = -np.linalg.lstsq(by_turn.T @ (w * by_turn), by_turn.T @ (w * by_camera), rcond=None)[0]
            out.append(by_camera + by_turn @ moves)
        return out

    def _turn(self, numbers):
        step = CUBE_STEPS[2]
        for k, p in enumerate(self.points):
            matrix = self._view(numbers, p)
            turn = self.turns[k]
            for _ in range(refine.TURN_STEPS):
                now = misses(matrix, p, self.centres[k], turn, *self.sizes[k])
                slope = (misses(matrix, p, self.centres[k], turn + step, *self.sizes[k]) - now) / step
                w = np.where(np.abs(now) <= EDGE_HUBER_PX, 1.0, EDGE_HUBER_PX / np.maximum(np.abs(now), 1e-12))
                turn -= float((w * slope) @ now / max(float((w * slope) @ slope), 1e-12))
            self.turns[k] = turn


def camera(frame_size, sights, sizes, start):
    model = Sized(frame_size, sights, start, sizes)
    numbers, r, _ = least_squares(model.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                                  iterations=200, derivatives=model.derivatives)
    return refine._pinhole(numbers).matrix, _huber_cost(r[:-2], EDGE_HUBER_PX), numbers


def placed_miss(matrix, sight, size):
    """The cube, of its own size, placed by its own edges through the camera: against the reading, desk mm."""
    reading = HANDEDNESS @ np.array((sight.x, sight.y))
    start = np.array([*refine.on_desk(matrix, sight), math.radians(sight.turn_deg)])
    put = least_squares(lambda t: misses(matrix, sight, t[:2], t[2], *size), start, CUBE_STEPS[:3],
                        huber=EDGE_HUBER_PX, iterations=30)[0]
    return HANDEDNESS @ (put[:2] - reading)


def run(name, camera_file, points_file, blue, green, label):
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    sights = [p.sighting for p in stored]
    sizes = [blue if p.cube == "blue" else green if p.cube == "green" else (40.0, 40.0) for p in stored]
    start = fit.cube_fit().true_to_pixel
    matrix, cost, numbers = camera(fit.frame_size, sights, sizes, start)
    offs = [float(np.linalg.norm(placed_miss(matrix, s, z))) for s, z in zip(sights, sizes)]
    line = (f"  {label:34} cost {cost:8.1f}  misses " + " ".join(f"#{p.id}:{o:4.2f}" for p, o in zip(stored, offs))
            + f"  f {numbers[0]:.0f}")
    preds = []
    if len(stored) >= 4:
        triples = list(itertools.combinations(range(len(stored)), 3))
        for three in triples[::max(1, len(triples) // 40)]:
            try:
                m3, _, _ = camera(fit.frame_size, [sights[k] for k in three], [sizes[k] for k in three], start)
            except np.linalg.LinAlgError:
                continue
            preds += [float(np.linalg.norm(placed_miss(m3, sights[k], sizes[k])))
                      for k in range(len(stored)) if k not in three]
        line += (f"\n  {'':34} every three predicting the rest: median {np.median(preds):4.2f}, "
                 f"90% {np.percentile(preds, 90):4.2f}, worst {max(preds):4.2f} ({len(preds)})")
    print(line)
    return cost


if __name__ == "__main__":
  for name, files in SESSIONS.items():
      print(f"\n=== {name}")
      run(name, *files, (40.0, 40.0), (40.0, 40.0), "all cubes 40 mm")
      for dz in (0.5, 1.0, 1.5, 2.0):
          run(name, *files, (40.0, 40.0 + dz), (40.0, 40.0), f"blue's top {dz:+.1f} mm")
      for scale in (1.0125, 1.025, 1.0375):
          run(name, *files, (40.0 * scale, 40.0 * scale), (40.0, 40.0), f"blue a {40 * scale:.1f} mm cube")
      for dz in (-0.5, 0.5, 1.0):
          run(name, *files, (40.0, 40.0), (40.0, 40.0 + dz), f"green's top {dz:+.1f} mm")
