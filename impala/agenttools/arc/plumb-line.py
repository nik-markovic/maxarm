"""Is the lens bending straight lines? The paper's lines, where paper-lines.py found them, in frame pixels: a perfect
lens keeps each straight. Fitted: the one bend of the lens (k1, round the picture's middle) that straightens them
all best. camera.json only tells where on the frame each found point is; the answer does not lean on it. No arm.

python plumb-line.py FRAME-lines.npz [FRAME-lines.npz ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
camera = json.loads((ROOT / "config" / "camera.json").read_text())
P = np.array(camera["true_to_pixel"])
H = P[:, [0, 1, 3]]
width, height = camera["frame_size"]
FOCAL = 1916.0                         # camera.json's, px: only to put k1 on the usual scale
centre = np.array([(width - 1) / 2, (height - 1) / 2])

lines = []
for path in sys.argv[1:]:
    data = np.load(path)
    middle = data["middle"]
    for name in ("a", "b"):
        angle = float(data[f"{name}_angle"])
        normal = np.array([math.cos(angle), math.sin(angle)])
        along = np.array([-normal[1], normal[0]])
        rows = data[f"{name}_rows"]
        for number in np.unique(rows[:, 0]):
            mine = rows[rows[:, 0] == number]
            if len(mine) < 8:
                continue
            true = middle + mine[:, 2:3] * normal + mine[:, 1:2] * along
            mapped = np.column_stack((true, np.ones(len(true)))) @ H.T
            lines.append(mapped[:, :2] / mapped[:, 2:])


def undistort(pixels, k1):
    rel = (pixels - centre) / FOCAL
    r2 = (rel ** 2).sum(axis=1, keepdims=True)
    return centre + FOCAL * rel * (1 + k1 * r2)


def bends(k1):
    """Each line's points off its own straight line, px, after undoing a bend k1."""
    out = []
    for pixels in lines:
        points = undistort(pixels, k1)
        mean = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - mean)
        out.append((points - mean) @ vt[1])
    return out


def cost(k1):
    return float(np.sum([np.sum(off ** 2) for off in bends(k1)]))


ks = np.linspace(-0.3, 0.3, 601)
costs = [cost(k) for k in ks]
best = ks[int(np.argmin(costs))]
for _ in range(40):                      # golden-section polish
    low, high = best - 0.002, best + 0.002
    for _ in range(30):
        a, b = low + 0.382 * (high - low), low + 0.618 * (high - low)
        if cost(a) < cost(b):
            high = b
        else:
            low = a
    best = (low + high) / 2
    break


def report(k1, label):
    offs = bends(k1)
    every = np.concatenate([np.abs(off) for off in offs])
    sag = [np.abs(off).max() for off in offs]
    print(f"{label}: points off their line median {np.median(every):.3f} px, 95% {np.percentile(every, 95):.3f}; "
          f"each line's worst: median {np.median(sag):.3f} px, max {np.max(sag):.3f}")


print(f"{len(lines)} lines, {sum(len(line) for line in lines)} points")
report(0.0, "as found (no lens bend)")
report(best, f"lens bend k1 = {best:+.4f} undone")
# What that bend does on the desk: how much the picture's scale changes from the middle to where the sheets are.
for r_px in (300, 600, 900, 1200):
    rel = r_px / FOCAL
    print(f"  k1 {best:+.4f}: at {r_px} px from the picture's middle, sizes along the radius change "
          f"{100 * 3 * best * rel ** 2:+.2f}%, across it {100 * best * rel ** 2:+.2f}%")
