#!/usr/bin/env python3
"""How far the calibrated top-down view is from true millimetres, measured with tiles.

Every tile is a rigid 17.5 x 20 x 4 mm box. The view's ground frame is fitted
with a perspective correction -- a scale along each axis, a shear, and the two
keystone terms by which the near side of the view comes out longer than the far
side -- jointly with every tile's position and turn, scored on the edges that
can be trusted (`tiles/pose.py`). The correction is held still at the tiles'
centroid and does not turn, so it changes shape only.

Then each half of the tiles is used to fit it and the other half to judge it:
fitted with their size left free, do the held-out tiles come out 17.5 x 20?

    ./agenttools/fit-perspective.py files/tiles-1.png
"""

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

# Coordinate-descent steps: linear terms, and keystone per millimetre.
LINEAR_STEP = 0.004
KEYSTONE_STEP = 2e-5


class Correction:
    """true millimetres -> the view's millimetres: m + L (g - m) / (1 + k . (g - m))."""

    def __init__(self, centre, params=(0, 0, 0, 0, 0)):
        self.centre = np.asarray(centre, float)
        self.params = np.asarray(params, float)

    def apply(self, points):
        a, b, d, e, f = self.params
        linear = np.array([[1 + a, b], [b, 1 + d]])
        offset = points - self.centre
        return self.centre + (offset @ linear.T) / (1 + offset @ np.array([e, f]))[:, None]

    def normals(self, points, normals):
        """Normals carried through the correction's local linear part."""
        step = 1e-3
        moved = []
        for column in range(2):
            delta = np.zeros(2)
            delta[column] = step
            moved.append((self.apply(points + delta) - self.apply(points - delta)) / (2 * step))
        # Rows of the Jacobian at each point; normals go by its inverse transpose.
        jacobian = np.stack(moved, axis=-1)
        carried = np.einsum("nji,nj->ni", np.linalg.inv(jacobian), normals)
        return carried / np.linalg.norm(carried, axis=1, keepdims=True)


def outline(field, camera_xy, camera_z, centre, angle, size=pose.TILE_MM):
    """The trusted edges of a tile of `size`, in true millimetres, as pose.EdgeField.outline."""
    width, height = size
    baseline = np.array([np.cos(angle), np.sin(angle)])
    up = np.array([-baseline[1], baseline[0]])
    top = np.array([centre + sx * width / 2 * baseline + sy * height / 2 * up
                    for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))])
    shift = (pose.TILE_THICKNESS_MM / camera_z) * (camera_xy - top)
    points, normals, weights = [], [], []
    t = np.linspace(0, 1, pose.SAMPLES_PER_EDGE)
    for index in range(4):
        start, end = top[index], top[(index + 1) % 4]
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([direction[1], -direction[0]])
        if np.dot(normal, centre - start) > 0:
            normal = -normal
        margin = pose.CORNER_MARGIN_MM / np.linalg.norm(end - start)
        along = margin + t * (1 - 2 * margin)
        if np.dot(normal, camera_xy - centre) <= 0:
            edge, weight = start + np.outer(along, end - start), 1.0
        else:
            s, e = start + shift[index], end + shift[(index + 1) % 4]
            edge, weight = s + np.outer(along, e - s), pose.NEAR_EDGE_WEIGHT
        points.append(edge)
        normals.append(np.repeat(normal[None], len(edge), axis=0))
        weights.append(np.full(len(edge), weight))
    return np.vstack(points), np.vstack(normals), np.concatenate(weights)


def tile_score(field, correction, camera, tile, size=pose.TILE_MM):
    points, normals, weights = outline(field, camera[:2], camera[2], tile[:2], tile[2], size)
    strength = field.strength(correction.apply(points), correction.normals(points, normals))
    return float((strength * weights).sum() / weights.sum())


def descend(score, start, steps, rounds=40):
    best, value, steps = np.array(start, float), score(start), np.array(steps, float)
    for _ in range(rounds):
        improved = False
        for axis in range(len(best)):
            if not steps[axis]:
                continue
            for sign in (1, -1):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                trial_value = score(trial)
                if trial_value > value:
                    best, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
    return best, value


def fit_correction(field, camera, tiles, rounds=6):
    correction = Correction(np.mean([t[:2] for t in tiles], axis=0))
    tiles = [np.array(t, float) for t in tiles]
    for _ in range(rounds):
        tiles = [descend(lambda p: tile_score(field, correction, camera, p), t,
                         (0.25, 0.25, np.radians(0.5)), rounds=12)[0] for t in tiles]

        def total(params):
            trial = Correction(correction.centre, params)
            return sum(tile_score(field, trial, camera, t) for t in tiles)

        params, _ = descend(total, correction.params,
                            (LINEAR_STEP, LINEAR_STEP, LINEAR_STEP, KEYSTONE_STEP, KEYSTONE_STEP),
                            rounds=12)
        correction = Correction(correction.centre, params)
    return correction, tiles


def free_size(field, correction, camera, tile):
    """Refit one tile with its width and height free: (width, height)."""
    def score(p):
        return tile_score(field, correction, camera, p[:3], (p[3], p[4]))
    fitted, _ = descend(score, (*tile, *pose.TILE_MM), (0.25, 0.25, np.radians(0.5), 0.25, 0.25))
    return fitted[3], fitted[4]


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    found = scene.find_tiles(frame, mapping, read.Reader())
    view = topdown.render(frame, mapping, scene.TILE_HEIGHT_MM)
    field = pose.EdgeField(view, mapping.camera_position())
    camera = mapping.camera_position()
    singles = [t for t in found if all(np.hypot(*np.subtract(t.centre_mm, o.centre_mm)) > 21
                                       for o in found if o is not t)]
    starts = [(*t.centre_mm, np.radians(t.baseline_deg)) for t in singles]
    identity = Correction(np.mean([s[:2] for s in starts], axis=0))
    print(f"{len(singles)} single tiles of {len(found)}")

    def sizes(correction, subset):
        return np.array([free_size(field, correction, camera, starts[i]) for i in subset])

    halves = (list(range(0, len(starts), 2)), list(range(1, len(starts), 2)))
    for fit_half, judge_half in (halves, halves[::-1]):
        correction, _ = fit_correction(field, camera, [starts[i] for i in fit_half])
        before, after = sizes(identity, judge_half), sizes(correction, judge_half)
        a, b, d, e, f = correction.params
        print(f"fitted on {len(fit_half)}: scale x {1 + a:.3f} y {1 + d:.3f} shear {b:+.3f} "
              f"keystone ({e * 100:+.4f}, {f * 100:+.4f}) per 100 mm")
        for label, measured in (("as calibrated", before), ("corrected", after)):
            print(f"  held-out {len(judge_half)}, {label:13s}: width {measured[:, 0].mean():5.2f} "
                  f"+-{measured[:, 0].std():.2f}, height {measured[:, 1].mean():5.2f} "
                  f"+-{measured[:, 1].std():.2f}  (tile is 17.5 x 20)")
    correction, _ = fit_correction(field, camera, starts)
    print(f"all tiles: params {np.round(correction.params, 6)} about {np.round(correction.centre, 1)}")
    np.save(ROOT / "files" / "perspective-correction.npy",
            np.concatenate((correction.centre, correction.params)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
