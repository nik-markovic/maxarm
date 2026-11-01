#!/usr/bin/env python3
"""Where two calibrations disagree about a 40 mm cube: drawn through one, fitted through the other, height free.

    .venv/bin/python impala/agenttools/height/two-cameras.py [camera.json]

The refined camera of the file draws a cube on the desk every 40 mm in reach (the edges as found points); the
three cubes' own camera then fits it, base on the desk, x, y, turn and height free. How tall it looks there, and
how far its base moved, say how much the refinement changed the camera's sense of height.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera.fit import EDGE_HUBER_PX, CameraFit                                          # noqa: E402
from camera.model import CUBE_MM, camera_position, least_squares, project, visible_edges  # noqa: E402
from camera import edges                                                                 # noqa: E402
from maxarm.path import REACH_MAX_MM, REACH_MIN_MM                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module                                                      # noqa: E402
box = import_module("cube-height").box


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config" / "camera-previous.json"
    refined = CameraFit.from_json(json.loads(path.read_text()))
    alone = refined.cube_fit()
    draw, fit = refined.true_to_pixel, alone.true_to_pixel
    width, height = refined.frame_size
    rows = []
    for x in range(-280, 281, 40):
        for y in range(-280, 281, 40):
            if not REACH_MIN_MM + 20 <= math.hypot(x, y) <= REACH_MAX_MM - 20 or y > 40:
                continue
            centre = refined.desk_to_true @ (x, y)
            corners = project(draw, box(centre, 0.4, CUBE_MM))
            if corners.min() < 0 or corners[:, 0].max() >= width or corners[:, 1].max() >= height:
                continue
            seen = visible_edges(centre, 0.4, camera_position(draw))
            starts, ends, targets = [], [], []
            for edge in seen:
                for share in np.linspace(0.12, 0.88, 24):
                    starts.append(edge.start), ends.append(edge.end)
                    targets.append(corners[edge.start] + share * (corners[edge.end] - corners[edge.start]))
            starts, ends, targets = np.array(starts), np.array(ends), np.array(targets)
            q = least_squares(lambda q: edges.signed_distances(project(fit, box(q[:2], q[2], q[3])), starts, ends,
                                                               targets),
                              np.array([*centre, 0.4, CUBE_MM]), np.array([0.01, 0.01, 1e-5, 0.01]),
                              huber=EDGE_HUBER_PX, iterations=60)[0]
            rows.append((x, y, q[3] - CUBE_MM, math.dist(q[:2], centre)))
    rows = np.array(rows)
    print(f"{len(rows)} places; height seen through the cubes' camera minus 40: "
          f"{rows[:, 2].min():+.2f} to {rows[:, 2].max():+.2f} mm; base moved median {np.median(rows[:, 3]):.2f}, "
          f"worst {rows[:, 3].max():.2f} mm")
    for x, y, dz, moved in rows[np.argsort(-np.abs(rows[:, 2]))][:6]:
        print(f"  ({x:+4.0f}, {y:+4.0f}): {dz:+.2f} mm tall, base {moved:.2f} mm")


if __name__ == "__main__":
    main()
