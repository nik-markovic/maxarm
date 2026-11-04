#!/usr/bin/env python3
"""The changed calibration code on 3 Nov's data, no arm: CALIBRATE's fit run again on config/camera-frame.png and
camera-points.json (the arm laid by a turn and a shift now), and the 24 desk cups of the grid-paper run predicted
through it from the arm's readings with raw.py as it is now (the way-down term back) -- judged in the paper's mm
(analyse.py). Then the Fine Tune page's description of the points, to see it run.

    check-tie.py
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
sys.path[1:1] = [str(JACKAL / "src" / "calibration"), str(JACKAL / "src")]
from applib.camera import store  # noqa: E402
from applib.camera.fit import fit_camera, stretch_of  # noqa: E402
from maxarm import raw  # noqa: E402
from maxarm.spaces import Space, Spaces  # noqa: E402

spec = importlib.util.spec_from_file_location("analyse", HERE / "analyse.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)


def main() -> None:
    config = JACKAL / "config"
    frame = cv2.imread(str(config / "camera-frame.png"))
    cubes = json.loads((config / "camera-points.json").read_text())["cubes"]
    fit = fit_camera(frame, {colour: (cube["x"], cube["y"]) for colour, cube in cubes.items()})
    low, high, skew = stretch_of(fit.measured_stretch)
    print("CALIBRATE again:", store.describe_fit(fit))
    print(f"  desk_to_true {np.round(fit.desk_to_true, 4).tolist()}; measured stretch {low:.4f}-{high:.4f}, "
          f"skew {skew:.2f} deg")
    # the 24 desk cups: their pixels (rim through the camera of the run) against the new calibration's prediction
    P_run, _, run = analyse.load()
    Hi, _ = analyse.paper(P_run, run["paper"])
    spaces = Spaces(config)
    rows = json.loads((HERE / "analysis.json").read_text())["points"]
    saved = store.load_camera(config)
    misses = {"saved camera.json, raw.py of the run": [], "CALIBRATE again, raw.py now": []}
    for point in rows:
        if math.hypot(*point["desk"][:2]) < 109:
            continue
        cup = analyse.to_paper(Hi, analyse.project(P_run, np.array([[*point["rim_desk"], 0.0]]))[0])
        now = spaces.convert(raw.true_position(tuple(point["raw"])), Space.ARM, Space.DESK)
        for name, matrix, desk in (("saved camera.json, raw.py of the run", saved.desk_to_pixel, point["desk"]),
                                   ("CALIBRATE again, raw.py now", fit.desk_to_pixel, now)):
            pixel = analyse.project(matrix, np.array([[desk[0], desk[1], 0.0]]))[0]
            misses[name].append(np.linalg.norm(analyse.to_paper(Hi, pixel) - cup))
    for name, values in misses.items():
        values = np.array(values)
        print(f"  the 24 desk cups, {name}: {np.median(values):.2f} mm typically, {values.max():.2f} worst")
    # the Fine Tune page's points, as it would describe them
    from tests import refine as tests_refine                       # noqa: E402
    points = store.load_refine_points(config, fit)
    described = tests_refine.describe(fit, points)
    for row in described["points"]:
        print(f"  point {row['id']}: off at CALIBRATE {row['off']['calibrate']}, after SAVE {row['off']['after']}, "
              f"put {row['put']}")
    print("  SAVE:", described["problem"] or described["fit"])


if __name__ == "__main__":
    main()
