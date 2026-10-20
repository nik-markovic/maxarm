#!/usr/bin/env python3
"""How far the shoulder and elbow servos really turn per step, from desk touch points.

    ../.venv/bin/python agenttools/fit-servo-scale.py                     # config/desk-points.json
    ../.venv/bin/python agenttools/fit-servo-scale.py path/to/points.json

The firmware assumes 0.24 degrees per step, counted from the centre step 500, where the arm is at
its reset pose. If the servos really turn `scale` times that, a flat desk reads lower the further
the arm reaches. This finds the one scale that makes every touch point the same real height, then
checks it on what it was not fitted to: the owner's reset-pose and card measurements
(work/KINEMATICS.md). No arm needed; nothing moves.
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from maxarm import geometry                                  # noqa: E402

# The owner's measurements, 2026-10-19 (work/KINEMATICS.md sections 1.1 and 3).
WRIST_TO_TIP_MM = 63.0
HOME_WRIST_MM = 223.0
DESK_SQUASH_MM = 0.75            # desk touches were pressed 0.5-1 mm harder than card touches
CARDS = [(53.98, y, z) for y, z in ((-260, 98), (-220, 98), (-200, 98), (-180, 100), (-160, 101),
                                     (-140, 103), (-120, 104), (-100, 104), (-80, 105))]
CARDS += [(85.60, -80, 135), (85.60, -260, 132)]


def elevations(x: float, y: float, z: float) -> Tuple[float, float]:
    """The firmware's upper-arm and forearm angles above level, in radians."""
    _, angle2, angle3 = geometry.inverse((x, y, z))
    return math.radians(180.0 - angle2), math.radians(-angle3)


def lift(elev: np.ndarray, scale: float) -> np.ndarray:
    """Wrist height above the shoulder joint when the servos turn `scale` times what the firmware assumes."""
    upper = math.pi / 2 + scale * (elev[:, 0] - math.pi / 2)
    return geometry.L2 * np.sin(upper) + geometry.L3 * np.sin(scale * elev[:, 1])


def best_scale(elev: np.ndarray) -> Tuple[float, float]:
    """The scale that leaves the touch points' real heights least spread, and its standard error."""
    spread = lambda s: float(np.std(lift(elev, s)))
    grid = np.arange(0.85, 1.10, 0.0005)
    s = float(grid[np.argmin([spread(v) for v in grid])])
    for step in (1e-4, 1e-5, 1e-6):
        s = min((s + k * step for k in range(-10, 11)), key=spread)
    h = 1e-4
    curvature = (spread(s + h) ** 2 - 2 * spread(s) ** 2 + spread(s - h) ** 2) / h ** 2
    return s, math.sqrt(2 * spread(s) ** 2 / (len(elev) - 2) / curvature)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "config" / "desk-points.json"
    points = json.loads(path.read_text())["points"]
    xyz = np.array([(p["x"], p["y"], p["z"]) for p in points])
    elev = np.array([elevations(*p) for p in xyz])
    reach = np.hypot(xyz[:, 0], xyz[:, 1])

    print(f"{len(points)} touch points from {path}")
    print(f"  as the firmware says: real heights spread {np.std(lift(elev, 1.0)):.2f} mm (RMS)")
    scale, error = best_scale(elev)
    heights = lift(elev, scale)
    print(f"  servos turn {scale:.4f} +- {error:.4f} of what the firmware assumes: "
          f"{0.24 * scale:.4f} degrees per step, {240 * scale:.1f} per 1000")
    print(f"  spread left {np.std(heights):.2f} mm, worst {np.max(np.abs(heights - heights.mean())):.2f}")
    for label, part in (("reach under 180", reach < 180), ("reach 180 and over", reach >= 180)):
        s, e = best_scale(elev[part])
        other = lift(elev[~part], s)
        offset = lift(elev[part], s).mean()
        print(f"  fitted on {label} only: {s:.4f} +- {e:.4f}; the other points then sit "
              f"{np.sqrt(np.mean((other - offset) ** 2)):.2f} mm (RMS) off")

    shoulder = WRIST_TO_TIP_MM - DESK_SQUASH_MM - heights.mean()
    print(f"\nnot fitted, predicted: shoulder joint {shoulder:.1f} mm above the desk (owner: about 96); "
          f"wrist at reset {shoulder + geometry.L2:.1f} (owner: {HOME_WRIST_MM:.0f}); forearm level at reset")
    cards = np.array([elevations(0.0, y, z) for _, y, z in CARDS])
    card_tip = shoulder + lift(cards, scale) - WRIST_TO_TIP_MM
    for size in (53.98, 85.60):
        mine = [i for i, card in enumerate(CARDS) if card[0] == size]
        print(f"  card {size} mm tall, {len(mine)} touches: read as "
              + " ".join(f"{card_tip[i]:.1f}" for i in mine))

    true_z = geometry.L0 + heights
    design = np.column_stack((xyz[:, 0], xyz[:, 1], np.ones(len(xyz))))
    (tilt_x, tilt_y, _), *_ = np.linalg.lstsq(design, true_z, rcond=None)
    left = true_z - design @ np.linalg.lstsq(design, true_z, rcond=None)[0]
    print(f"\nthe desk, corrected: tilt {tilt_x * 1000:+.2f} mm/m in x, {tilt_y * 1000:+.2f} in y; "
          f"off the plane {np.sqrt(np.mean(left ** 2)):.2f} mm (RMS), worst {np.max(np.abs(left)):.2f}")


if __name__ == "__main__":
    main()
