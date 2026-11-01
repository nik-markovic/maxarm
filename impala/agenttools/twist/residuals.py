#!/usr/bin/env python3
"""Fine Tune's points as SAVE would fit them: for each, where the arm said, where the camera sees the cube, and the
miss split along the arm's reach (radial, + further out) and round its base (tangential, + towards +x side turning
clockwise seen from above, i.e. increasing bearing). Nothing is written but twist-<name>.json beside this file.

    .venv/bin/python impala/agenttools/twist/residuals.py config-dir name
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                            # noqa: E402
from camera.fit import stretch_of                                   # noqa: E402
from camera.refine import off_by, refine                            # noqa: E402


def main():
    config, name = Path(sys.argv[1]), sys.argv[2]
    fit = store.load_camera(config)
    points = store.load_refine_points(config)
    saved = refine(fit, [point.sighting for point in points])
    low, high, skew = stretch_of(saved.desk_to_true)
    print(f"{name}: {len(points)} points; stretch {low:.4f}-{high:.4f}, skew {skew:.2f} deg")
    rows = []
    for point in points:
        off = off_by(saved, point.sighting)
        arm = np.array([point.x, point.y])
        miss = off.seen - arm                       # where the camera sees it, less where the arm says
        reach, bearing = math.hypot(*arm), math.atan2(arm[0], -arm[1])
        out = np.array([math.sin(bearing), -math.cos(bearing)])
        round_ = np.array([math.cos(bearing), math.sin(bearing)])
        rows.append({"id": point.id, "arm": arm.tolist(), "seen": off.seen.tolist(), "pixel": list(point.sighting.pixel),
                     "z": point.z, "height": off.height, "reach": reach, "bearing": math.degrees(bearing),
                     "miss": miss.tolist(), "radial": float(miss @ out), "tangential": float(miss @ round_)})
        print(f"  {point.id:3d} arm ({point.x:7.1f},{point.y:7.1f}) reach {reach:5.1f} bearing {math.degrees(bearing):+6.1f}"
              f" px ({point.sighting.pixel[0]:6.0f},{point.sighting.pixel[1]:6.0f}) off {off.across:4.2f}"
              f"  radial {rows[-1]['radial']:+5.2f} round {rows[-1]['tangential']:+5.2f}  height {off.height:+5.2f}")
    across = [math.hypot(*row["miss"]) for row in rows]
    print(f"  median {np.median(across):.2f}, rms {math.sqrt(np.mean(np.square(across))):.2f}, max {max(across):.2f}; "
          f"green (<=1) {sum(a <= 1.0 for a in across)} of {len(across)}")
    camera = np.linalg.inv(saved.true_to_pixel[:, [0, 1, 2]]) @ -saved.true_to_pixel[:, 3]
    (HERE / f"twist-{name}.json").write_text(json.dumps({"rows": rows, "camera_true": camera.tolist(),
                                                         "desk_to_true": saved.desk_to_true.tolist(),
                                                         "true_to_pixel": saved.true_to_pixel.tolist()}, indent=1))


if __name__ == "__main__":
    main()
