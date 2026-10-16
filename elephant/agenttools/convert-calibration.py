#!/usr/bin/env python3
"""Turn dolphin's calibration.json into elephant's desk.json and camera.json, and prove nothing was lost.

    ../.venv/bin/python agenttools/convert-calibration.py ../dolphin/config/calibration.json config

dolphin's file holds the camera (`desk_camera`, 3x4, true desk millimetres to
pixels), the arm's x/y correction (`desk_to_arm`, 3x3 affine) and condor's desk
plane (`aacs_zero_plane`, board z of the bare desk at x, y), plus cube points
and two derived plane transforms. elephant keeps two matrices:

- `desk.json`, `desk_to_arm` 4x4: the x/y correction and the desk plane, composed.
- `camera.json`, `desk_to_pixel` 3x4: the camera, unchanged.

The check runs a grid of pixels at several heights through dolphin's own
`DeskMapping.pixel_to_board()` and through the two new matrices, and fails if
they differ by more than a micrometre. One-off; the files' `about` says where
they came from once this is gone.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "dolphin" / "calibration"))

from maxarm.spaces import Camera, Spaces             # noqa: E402
from mapping import DeskMapping                      # noqa: E402  (dolphin's, read only)

DESK_ABOUT = ("desk space to arm space: arm [x, y, z, 1] = desk_to_arm @ desk [x, y, z, 1]. "
              "Desk space is millimetres on the desk, z the height above its surface (0: the cup "
              "snug on it). This one is condor's three measured desk heights (the tilt) composed "
              "with dolphin's grid-paper x/y correction (the arm's x travels ~7% short); the tilt "
              "calibration replaces it.")
CAMERA_ABOUT = ("desk space to camera pixels: [u, v, w] = desk_to_pixel @ [x, y, z, 1], "
                "px = u / w, py = v / w. dolphin's grid-paper camera: square pixels, true "
                "millimetres, 2592x1944 frames.")
TOLERANCE_MM = 1e-3


def desk_to_arm(calibration: dict) -> np.ndarray:
    xy = np.array(calibration["desk_to_arm"])
    plane = calibration["aacs_zero_plane"]
    slope = np.array([plane["dz_dx"], plane["dz_dy"]])
    matrix = np.eye(4)
    matrix[:2, :2], matrix[:2, 3] = xy[:2, :2], xy[:2, 2]
    # board z = height + z0 + slope . (arm x, y), and arm x, y is the correction of desk x, y.
    matrix[2, :2] = slope @ xy[:2, :2]
    matrix[2, 3] = plane["z0"] + slope @ xy[:2, 2]
    return matrix


def worst_difference_mm(old: DeskMapping, config_dir: Path) -> float:
    spaces = Spaces(config_dir)
    camera = Camera.load(config_dir / "camera.json")
    worst = 0.0
    for px in np.linspace(200, 2400, 12):
        for py in np.linspace(200, 1750, 9):
            for height in (0.0, 4.0, 17.0, 40.0):
                before = old.pixel_to_board((px, py), height, seen_at_mm=height)
                after = spaces.arm.from_desk(camera.to_desk((px, py, height)))
                worst = max(worst, float(np.linalg.norm(np.subtract(before, after))))
    return worst


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    calibration = json.loads(source.read_text())
    target.mkdir(parents=True, exist_ok=True)
    (target / "desk.json").write_text(json.dumps(
        {"about": DESK_ABOUT, "desk_to_arm": desk_to_arm(calibration).tolist()}, indent=2) + "\n")
    (target / "camera.json").write_text(json.dumps(
        {"about": CAMERA_ABOUT, "desk_to_pixel": calibration["desk_camera"]}, indent=2) + "\n")
    worst = worst_difference_mm(DeskMapping.load(source), target)
    print(f"{source} -> {target}/desk.json, camera.json: worst difference from dolphin's "
          f"pixel_to_board {worst:.2e} mm over 432 points")
    return 0 if worst < TOLERANCE_MM else 1


if __name__ == "__main__":
    sys.exit(main())
