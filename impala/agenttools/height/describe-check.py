#!/usr/bin/env python3
"""The Fine Tune table as the page gets it (tests/refine.py describe()), for the owner's three cubes and for the
drawn scene of stretch-save.py with its red points; written as JSON for render-check.mjs. Nothing of config/ is
written.

    .venv/bin/python impala/agenttools/height/describe-check.py OUT-DIR
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(ROOT / "agenttools")]

import cube_scene                                                            # noqa: E402
from camera import store                                                     # noqa: E402
from camera.fit import CameraFit, fit_camera                                 # noqa: E402
from camera.locate import find_red_cube                                      # noqa: E402
from tests import refine as table                                            # noqa: E402

spec = importlib.util.spec_from_file_location("stretch_save", HERE / "stretch-save.py")
drawn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawn)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    config = ROOT / "config"
    fit = CameraFit.from_json(json.loads((config / "camera.json").read_text()))
    points = [store.RefinePoint(**point) for point in
              json.loads((config / "camera-refine-points.json").read_text())["points"]]
    owner = table.describe(fit, points)
    print("owner's three, through camera.json as saved (turn and shift):", [row["off"] for row in owner["points"]],
          owner["problem"])
    again = fit_camera(cubes_frame(config), {p.cube: (p.x, p.y) for p in points})
    owner_new = table.describe(again, store.cube_refine_points(again, {}, "now"))
    print("owner's three, through CALIBRATE again:", [row["off"] for row in owner_new["points"]], owner_new["problem"])
    (out / "owner.json").write_text(json.dumps(owner_new))
    rng = np.random.default_rng(1)
    scene = cube_scene.render(stretch=drawn.STRETCH)
    fit = fit_camera(scene.frame, scene.points)
    rows = store.cube_refine_points(fit, {}, "drawn")
    for index, spot in enumerate(drawn.SPOTS[:8]):
        cube_scene.TURNS_DEG["red"] = float(rng.uniform(0, 90))
        frame = cube_scene.render({"red": spot}, stretch=drawn.STRETCH, decoy=False, seed=index).frame
        try:
            found = find_red_cube(frame, fit)
        except Exception:                                                     # noqa: BLE001
            continue
        arm = np.array(spot, float) + rng.normal(0, 0.4, 2)
        rows.append(store.RefinePoint(len(rows) + 1, float(arm[0]), float(arm[1]), 42.0, [*spot, 42.0],
                                      list(found.pixel), list(cube_scene.SIZE), "drawn",
                                      turn_deg=found.true_turn_deg, edges=store.edges_to_json(found.edges)))
    for count in (3, 4, len(rows)):
        described = table.describe(fit, rows[:count])
        print(f"drawn, {count} points: problem {described['problem']}; fit {described['fit']}")
        print("   ", [(row["id"], row["off"], row["height"]) for row in described["points"]])
        (out / f"drawn-{count}.json").write_text(json.dumps(described))


def cubes_frame(config):
    import cv2
    return cv2.imread(str(config / "camera-frame.png"))


if __name__ == "__main__":
    main()
