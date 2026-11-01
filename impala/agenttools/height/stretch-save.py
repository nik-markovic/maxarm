#!/usr/bin/env python3
"""The new CALIBRATE and SAVE on a drawn scene whose arm measures differently from the camera: its x, y stretched
(the drawing's `stretch`), each reading off by a little, one point recorded off the cube's centre.

    .venv/bin/python impala/agenttools/height/stretch-save.py [reading error mm] [seed]

CALIBRATE on the three cubes drawn; then the red cube drawn at 15 places and found through CALIBRATE's camera
(RECORD POINT); SAVE from a start made a little wrong, as a real CALIBRATE's might be. Says how far each puts the
desk from the drawing's, the stretch each finds, each point's XY and Z after SAVE. A drawing says whether the fit
finds what is there to find, nothing about a real camera.
"""

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(ROOT / "agenttools")]

import cube_scene                                                          # noqa: E402
from camera import check, store                                            # noqa: E402
from camera.fit import HANDEDNESS, CameraFit, fit_camera, stretch_of       # noqa: E402
from camera.locate import find_red_cube                                    # noqa: E402
from camera.refine import _numbers, _pinhole, off_by, refine               # noqa: E402

STRETCH = ((1.02, 0.004), (0.004, 0.99))      # the arm's x, y to true mm: 2% and 1% apart, a little skewed
SPOTS = [(-200, -120), (-150, -40), (-120, -200), (-60, -140), (-30, -240), (0, -110), (40, -180), (70, -250),
         (110, -120), (130, -200), (170, -60), (200, -150), (-220, 20), (150, 30), (60, -60)]
OFF_CENTRE = (6, (2.4, -1.8))                 # the seventh red point recorded this far off the cube's centre, mm


def main():
    error = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    scene = cube_scene.render(stretch=STRETCH)
    truth = CameraFit(scene.true_to_pixel, np.array(STRETCH) @ HANDEDNESS, {}, cube_scene.SIZE, (0.0, 0.0), 0)
    drawn = stretch_of(truth.desk_to_true)
    print(f"drawn: focal {truth.focal:.1f}, the arm stretched {drawn[0]:.4f} to {drawn[1]:.4f}, {drawn[2]:.2f} deg")
    started = time.monotonic()
    fit = fit_camera(scene.frame, scene.points)
    report(f"CALIBRATE ({time.monotonic() - started:.1f} s)", fit, truth)
    print("  the three cubes from the arm's points: " + ", ".join(f"{colour} {fit.cube_off_mm(colour):.3f}"
                                                                for colour in fit.cubes) + " mm")
    cubes = store.cube_refine_points(fit, {}, "drawn")
    reds = []
    for index, spot in enumerate(SPOTS):
        cube_scene.TURNS_DEG["red"] = float(rng.uniform(0, 90))
        frame = cube_scene.render({"red": spot}, stretch=STRETCH, decoy=False, seed=index).frame
        try:
            found = find_red_cube(frame, fit)
        except Exception as reason:                                            # noqa: BLE001
            print(f"  red at {spot}: not found ({str(reason)[:60]})")
            continue
        arm = np.array(spot, float) + rng.normal(0, error, 2)
        if index == OFF_CENTRE[0]:
            arm += OFF_CENTRE[1]
        reds.append(store.RefinePoint(index + 4, float(arm[0]), float(arm[1]), 42.0, [*spot, 42.0],
                                      list(found.pixel), list(cube_scene.SIZE), "drawn",
                                      turn_deg=found.true_turn_deg, edges=store.edges_to_json(found.edges)))
    points = cubes + reds
    numbers = _numbers(fit.true_to_pixel)
    numbers[0] *= 0.985
    numbers[3:6] += (0.002, -0.0015, 0.001)
    numbers[6:9] += (2.0, -1.0, 3.0)
    start = dataclasses.replace(fit, true_to_pixel=_pinhole(numbers).matrix)
    report("the start, made wrong", start, truth)
    started = time.monotonic()
    saved = refine(start, [point.sighting for point in points])
    report(f"SAVE ({time.monotonic() - started:.1f} s)", saved, truth)
    print(f"  {saved.refined}")
    print("  each point after SAVE, XY and Z, mm (point 10 recorded "
          f"{math.hypot(*OFF_CENTRE[1]):.1f} mm off centre):")
    rows = [(point.id, off_by(saved, point.sighting)) for point in points]
    print("    " + "  ".join(f"{number}: {off.across:.2f} {off.height:+.2f}" for number, off in rows))


def report(name, fit, truth):
    typical, worst = check.difference_mm(truth, fit)
    low, high, skew = stretch_of(fit.desk_to_true)
    print(f"{name:24s} desk from the drawing's {typical:.2f} / {worst:.2f} mm; focal {fit.focal:.1f}; the arm "
          f"stretched {low:.4f} to {high:.4f}, {skew:.2f} deg")


if __name__ == "__main__":
    main()
