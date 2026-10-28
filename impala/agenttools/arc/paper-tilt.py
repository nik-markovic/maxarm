"""Does one error of the camera's view explain how the paper's lines run? paper-lines.py's line places fitted, both
directions at once, by a perfect grid seen through: (0) the camera as it is; (1) the camera with the desk stretched
one way and skewed; (2) also tilted -- the error a camera turned wrong, or a sheet lying tilted, makes. A sheet that
bows makes neither. No arm.

python paper-tilt.py FRAME-lines.npz [pitch_mm, else fitted]
"""
import math
import sys

import numpy as np

data = np.load(sys.argv[1])
fixed_pitch = float(sys.argv[2]) if len(sys.argv) > 2 else None
middle = data["middle"]
points, numbers, family = [], [], []
for index, name in enumerate(("a", "b")):
    angle = float(data[f"{name}_angle"])
    normal = np.array([math.cos(angle), math.sin(angle)])
    along = np.array([-normal[1], normal[0]])
    rows = data[f"{name}_rows"]
    points.append(rows[:, 2:3] * normal + rows[:, 1:2] * along)        # relative to the sheet's middle, true mm
    numbers.append(rows[:, 0])
    family.append(np.full(len(rows), index))
points, numbers, family = np.vstack(points), np.concatenate(numbers), np.concatenate(family)
angle_a = float(data["a_angle"])


def residuals(params, kind):
    phi, c_a, c_b, pitch = params[:4]
    if fixed_pitch:
        pitch = fixed_pitch
    e = s = g = h = 0.0
    if kind >= 1:
        e, s = params[4:6]
    if kind >= 2:
        g, h = params[6:8] * 1e-3
    x, y = points[:, 0], points[:, 1]
    w = 1 + g * x + h * y
    px = ((1 + e) * x + s * y) / w
    py = (s * x + (1 - e) * y) / w
    out = np.empty(len(points))
    for index, offset in ((0, c_a), (1, c_b)):
        theta = phi + index * math.pi / 2
        mine = family == index
        out[mine] = px[mine] * math.cos(theta) + py[mine] * math.sin(theta) - offset - numbers[mine] * pitch
    return out, (px, py)


def solve(kind, start):
    params = start.copy()
    count = 4 + 2 * kind
    keep = np.ones(len(points), bool)
    for round_ in range(6):
        for _ in range(30):
            r, _ = residuals(params, kind)
            jac = np.empty((len(points), count))
            for j in range(count):
                step = np.zeros_like(params)
                step[j] = 1e-6
                jac[:, j] = (residuals(params + step, kind)[0] - r) / 1e-6
            delta, *_ = np.linalg.lstsq(jac[keep], -r[keep], rcond=None)
            params[:count] += delta
            if np.abs(delta).max() < 1e-10:
                break
        r, _ = residuals(params, kind)
        keep = np.abs(r) < max(4 * np.median(np.abs(r[keep])), 0.05)
    return params, r, keep


# Start: the lines' own offsets from a straight fit, per direction.
start = np.zeros(8)
start[0] = angle_a
for index in (0, 1):
    mine = family == index
    theta = angle_a + index * math.pi / 2
    across = points[mine] @ [math.cos(theta), math.sin(theta)]
    design = np.column_stack((np.ones(mine.sum()), numbers[mine]))
    (offset, pitch), *_ = np.linalg.lstsq(design, across, rcond=None)
    start[1 + index] = offset
    start[3] = pitch
names = ("as the camera sees it", "+ stretched one way, skewed", "+ tilted")
previous = None
for kind in (0, 1, 2):
    params, r, keep = solve(kind, start if previous is None else previous)
    previous = params.copy()
    line = (f"({kind}) {names[kind]}: squares {fixed_pitch or params[3]:.4f} mm; left over median "
            f"{np.median(np.abs(r[keep])):.3f} mm, 95% {np.percentile(np.abs(r[keep]), 95):.3f}, "
            f"max {np.abs(r[keep]).max():.3f} ({keep.sum()} of {len(r)})")
    if kind >= 1:
        e, s = params[4:6]
        line += f"; stretch {200 * e:+.2f}% one way against the other, skew {math.degrees(2 * s):+.3f} deg"
    if kind >= 2:
        g, h = params[6:8] * 1e-3
        _, (px, py) = residuals(params, kind)
        _, (qx, qy) = residuals(np.concatenate((params[:6], [0, 0])), kind)
        move = np.hypot(px - qx, py - qy)
        line += (f"; the tilt moves the sheet's points by up to {move.max():.2f} mm "
                 f"(scale changes {1e4 * math.hypot(g, h):.2f}% per 10 cm, towards {math.degrees(math.atan2(h, g)):.0f} deg)")
    print(line)
