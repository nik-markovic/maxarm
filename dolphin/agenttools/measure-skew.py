#!/usr/bin/env python3
"""How square the tiles come out: each fitted as a free parallelogram on its trusted edges.

The check for a calibration. Through a true-millimetre camera every tile is a
17.5 x 20 rectangle, so its fitted skew is zero and its sides are the tile's.
condor's camera gave skews of -2.2 +- 2.1 degrees, up to 5; the grid camera
-0.1 +- 0.6, which is this method's own floor (perfect boxes rendered through
the camera come out within 0.8 degrees, and 0.3 mm large from edge blur).

    ./agenttools/measure-skew.py training/tiles/baseline/tiles-1.png
    ./agenttools/measure-skew.py training/tiles/baseline/tiles-1.png --config config/calibration-condor.json
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import pose                             # noqa: E402
import read                             # noqa: E402
import scene                            # noqa: E402
import topdown                          # noqa: E402
from mapping import DeskMapping         # noqa: E402

STEPS = (0.25, 0.25, np.radians(0.5), 0.25, 0.25, np.radians(0.5))


def parallelogram_score(field: pose.EdgeField, params) -> float:
    """Trusted-edge strength of a tile with its width, height and skew free."""
    x, y, angle, width, height, skew = params
    centre = np.array((x, y))
    baseline = np.array([np.cos(angle), np.sin(angle)])
    up = np.array([-baseline[1], baseline[0]]) + np.tan(skew) * baseline
    corners = np.array([centre + sx * width / 2 * baseline + sy * height / 2 * up
                        for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))])
    camera = field.camera_xyz
    shift = (pose.TILE_THICKNESS_MM / camera[2]) * (camera[:2] - corners)
    points, normals, weights = [], [], []
    along = np.linspace(0.1, 0.9, pose.SAMPLES_PER_EDGE)
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([direction[1], -direction[0]])
        if np.dot(normal, centre - start) > 0:
            normal = -normal
        if np.dot(normal, camera[:2] - (start + end) / 2) <= 0:
            edge, weight = start + np.outer(along, end - start), 1.0
        else:
            s, e = start + shift[index], end + shift[(index + 1) % 4]
            edge, weight = s + np.outer(along, e - s), pose.NEAR_EDGE_WEIGHT
        points.append(edge)
        normals.append(np.repeat(normal[None], len(edge), axis=0))
        weights.append(np.full(len(edge), weight))
    weights = np.concatenate(weights)
    strength = field.strength(np.vstack(points), np.vstack(normals))
    return float((strength * weights).sum() / weights.sum())


def descend(value_of, start, steps, rounds=50):
    best, value, steps = np.array(start, float), value_of(start), np.array(steps, float)
    for _ in range(rounds):
        improved = False
        for axis in range(len(best)):
            for sign in (1, -1):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                trial_value = value_of(trial)
                if trial_value > value:
                    best, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frame", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "calibration.json")
    args = parser.parse_args()
    frame = cv2.imread(str(args.frame))
    mapping = DeskMapping.load(args.config)
    view = topdown.render(frame, mapping, scene.TILE_HEIGHT_MM)
    field = pose.EdgeField(view, mapping.camera_position())
    fits = []
    print(" tile  baseline   width  height  skew")
    try:
        tiles = scene.find_tiles(frame, mapping, read.Reader())
    except scene.UnplacedLetters as refused:
        tiles = refused.tiles           # the placed ones measure the calibration just as well
    for tile in tiles:
        start = (*tile.desk_centre_mm, np.radians(tile.desk_baseline_deg), *pose.TILE_MM, 0.0)
        fitted = descend(lambda p: parallelogram_score(field, p), start, STEPS)
        fits.append(fitted)
        print(f"   {tile.letter}   {tile.desk_baseline_deg:7.1f}   {fitted[3]:5.2f}  {fitted[4]:5.2f}"
              f"  {np.degrees(fitted[5]):+5.1f}")
    fits = np.array(fits)
    skew = np.degrees(fits[:, 5])
    print(f"skew {skew.mean():+.1f} +- {skew.std():.1f} deg (worst {np.abs(skew).max():.1f}); "
          f"width {fits[:, 3].mean():.2f} +- {fits[:, 3].std():.2f}, height {fits[:, 4].mean():.2f} "
          f"+- {fits[:, 4].std():.2f} mm (tile is {pose.TILE_MM[0]} x {pose.TILE_MM[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
