#!/usr/bin/env python3
"""SAVE CALIBRATION on drawn scenes, where the camera is known: the new fit (edges, height) against the old (the
pixel under each cube's centre alone).

    .venv/bin/python impala/agenttools/height/synthetic-save.py [arm error mm] [seed]

The drawing's camera made a little wrong, as CALIBRATE's might be (focal length, where it stands, which way it
looks); then the red cube drawn at 15 places, found through that camera (RECORD POINT), the arm's point off by the
arm error. Says how far each calibration puts the desk from the drawing's own camera, and how tall it sees a 40 mm
cube across the desk. It is a drawing: it says whether the fit finds a camera that is there to find, nothing about
a real camera.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(ROOT / "agenttools")]

import cube_scene                                                          # noqa: E402
from camera import check, store                                            # noqa: E402
from camera.fit import HANDEDNESS, CameraFit                               # noqa: E402
from camera.locate import find_red_cube                                    # noqa: E402
from camera.refine import _numbers, _pinhole, off_by, refine               # noqa: E402
from camera.sighting import Sighting                                       # noqa: E402

SPOTS = [(-200, -120), (-150, -40), (-120, -200), (-60, -140), (-30, -240), (0, -110), (40, -180), (70, -250),
         (110, -120), (130, -200), (170, -60), (200, -150), (-220, 20), (150, 30), (60, -60)]


def main():
    error = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    scene = cube_scene.render()
    truth = CameraFit(scene.true_to_pixel, HANDEDNESS.copy(), {}, cube_scene.SIZE, (0.0, 0.0), 0)
    numbers = _numbers(scene.true_to_pixel)
    numbers[0] *= 0.985
    numbers[3:6] += (0.002, -0.0015, 0.001)
    numbers[6:9] += (2.0, -1.0, 3.0)
    cubes_seen = {colour: {"x": x, "y": y, "turn_deg": 0.0, "pixel": [0.0, 0.0]}
                  for colour, (x, y) in scene.points.items()}
    calibrated = CameraFit(_pinhole(numbers).matrix, HANDEDNESS.copy(), cubes_seen, cube_scene.SIZE, (0.0, 0.0), 0)
    cubes = []
    reds = []
    for index, spot in enumerate(SPOTS):
        cube_scene.TURNS_DEG["red"] = float(rng.uniform(0, 90))
        frame = cube_scene.render({"red": spot}, decoy=False, seed=index).frame
        found = find_red_cube(frame, calibrated)
        arm = (spot[0] + rng.normal(0, error), spot[1] + rng.normal(0, error))
        point = store.RefinePoint(index + 4, *arm, 42.0, [*spot, 42.0], list(found.pixel), list(cube_scene.SIZE),
                                  "drawn", turn_deg=found.true_turn_deg, edges=store.edges_to_json(found.edges))
        reds.append(point.sighting)
    points = cubes + reds
    old_points = [Sighting(p.x, p.y, p.pixel, p.frame_size) for p in points]
    for name, fit in (("start (CALIBRATE's)", calibrated), ("old SAVE", refine(calibrated, old_points))):
        report(name, fit, truth, points)
    started = time.monotonic()
    new = refine(calibrated, points)
    report(f"new SAVE ({time.monotonic() - started:.1f} s)", new, truth, points)


def report(name, fit, truth, points):
    typical, worst = check.difference_mm(truth, fit)
    offs = [off_by(fit, point) for point in points]
    across = [off.across for off in offs]
    heights = [off.height for off in offs]
    tall = height_sense(fit, truth)
    print(f"{name:22s} desk from the drawing's camera {typical:.2f} / {worst:.2f} mm; focal {fit.focal:.1f} "
          f"(drawn {truth.focal:.1f}); points across {np.median(across):.2f} / {max(across):.2f}, tops "
          f"{np.median(np.abs(heights)):.2f} / {max(np.abs(heights)):.2f} mm; a 40 mm cube seen "
          f"{tall.min():+.2f} to {tall.max():+.2f} mm")


def height_sense(fit, truth):
    """A cube drawn by the drawing's camera at places across the desk, measured through `fit`: its top from 40."""
    from camera.model import camera_position, project, visible_edges, cube_corners
    rows = []
    for x in range(-240, 241, 60):
        for y in range(-260, 41, 60):
            if not 120 <= math.hypot(x, y) <= 250:
                continue
            centre = truth.desk_to_true @ (x, y)
            corners = project(truth.true_to_pixel, cube_corners(centre, 0.4))
            if corners.min() < 0 or corners[:, 0].max() >= truth.frame_size[0] or \
                    corners[:, 1].max() >= truth.frame_size[1]:
                continue
            seen = visible_edges(centre, 0.4, camera_position(truth.true_to_pixel))
            starts = np.repeat([edge.start for edge in seen], 24)
            ends = np.repeat([edge.end for edge in seen], 24)
            shares = np.tile(np.linspace(0.12, 0.88, 24), len(seen))
            found = corners[starts] + shares[:, None] * (corners[ends] - corners[starts])
            pixel = project(truth.true_to_pixel, np.array([[*centre, 0.0]]))[0]
            point = Sighting(x, y, tuple(pixel), truth.frame_size, math.degrees(0.4), (starts, ends, found))
            rows.append(off_by(fit, point).height)
    return np.array(rows)


if __name__ == "__main__":
    main()
