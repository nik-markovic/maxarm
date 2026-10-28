"""The base's visible suction cups, each found as the circle on the desk whose outline best follows the light rim.

python cup-match.py still.png out-prefix [config-dir]
A circle on the desk (centre, radius; true mm, through the camera alone) is scored by the light-line response
along its outline, over the part of the rim the base does not hide. The best circle is searched over +-8 mm and
radii 20-27 mm, then polished. Then: the square the cups' centres make, and its centre against desk (0, 0), where
the pages centre the reach arcs. No arm.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera.fit import CameraFit  # noqa: E402
from camera.model import project  # noqa: E402

still, prefix = sys.argv[1], sys.argv[2]
config = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "config"
fit = CameraFit.from_json(json.loads((config / "camera.json").read_text()))
image = cv2.imread(still)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
ridge = np.maximum(-cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, ksize=3), 0)
P, D2T = fit.true_to_pixel, fit.desk_to_true
camera_foot = fit.camera_position[:2]


def sample(img, pts):
    return cv2.remap(img, pts[:, 0].astype(np.float32).reshape(-1, 1), pts[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()


def outline(c, r, phis):
    return project(P, np.column_stack((c[0] + r * np.cos(phis), c[1] + r * np.sin(phis), np.zeros(len(phis)))))


def visible_phis(c_true):
    """The half of the rim facing the camera, which the base cannot hide, and a quarter either side of it."""
    foot_true = np.linalg.solve(np.eye(2), D2T @ camera_foot)
    a = math.atan2(foot_true[1] - c_true[1], foot_true[0] - c_true[0])
    return a + np.radians(np.arange(-110, 110.1, 2.0))


def score(c, r, phis):
    return float(np.mean(sample(ridge, outline(c, r, phis))))


def rays(c0):
    """The other way: on rays out from where the calibration puts the cup, the outermost light line, 16-29 mm out;
    a circle through those points."""
    picked = []
    for phi in np.radians(np.arange(0, 360, 2.0)):
        radii = np.linspace(16, 29, 140)
        ray = project(P, np.column_stack((c0 + radii[:, None] * (math.cos(phi), math.sin(phi)), np.zeros(140))))
        profile = sample(ridge, ray)
        floor = max(3.0, 0.35 * profile.max())
        peaks = [k for k in range(1, 139) if profile[k] >= profile[k - 1] and profile[k] > profile[k + 1]
                 and profile[k] > floor]
        if peaks:
            picked.append(ray[peaks[-1]])
    q = np.linalg.solve(P[:, [0, 1, 3]], np.column_stack((picked, np.ones(len(picked)))).T).T
    on_desk = q[:, :2] / q[:, 2:]
    keep = np.ones(len(on_desk), bool)
    for _ in range(4):
        a = np.column_stack((2 * on_desk[keep], np.ones(keep.sum())))
        cx, cy, c = np.linalg.lstsq(a, (on_desk[keep] ** 2).sum(1), rcond=None)[0]
        r = math.sqrt(c + cx * cx + cy * cy)
        off = np.abs(np.linalg.norm(on_desk - (cx, cy), axis=1) - r)
        keep = off < max(0.4, 3 * np.median(off[keep]))
    return np.array([cx, cy]), r


cups = {"front-right": (60, -60), "front-left": (-60, -60), "back-right": (60, 60)}
found = {}
vis = image.copy()
for name, desk_c in cups.items():
    c0 = D2T @ desk_c
    phis = visible_phis(c0)
    best = None
    for dx in np.arange(-8, 8.01, 0.5):
        for dy in np.arange(-8, 8.01, 0.5):
            for r in np.arange(20, 27.01, 0.5):
                s = score(c0 + (dx, dy), r, phis)
                if best is None or s > best[0]:
                    best = (s, np.array([c0[0] + dx, c0[1] + dy]), r)
    s, c, r = best
    for step in (0.25, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for d in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                s2 = score(c + d[:2], r + d[2], phis)
                if s2 > s:
                    s, c, r, improved = s2, c + d[:2], r + d[2], True
    # how sharp the optimum is: the score 1 mm off each way, as a share of the best
    sharp = [score(c + d, r, phis) / s for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    found[name] = (c, r)
    dc = np.linalg.solve(D2T, c)
    print(f"{name:12s} drawn at desk ({desk_c[0]}, {desk_c[1]}); found desk ({dc[0]:6.2f}, {dc[1]:6.2f}), radius "
          f"{r:5.2f} mm; score {s:.1f}, 1 mm off it {min(sharp):.2f}-{max(sharp):.2f} of that")
    cv2.polylines(vis, [np.round(outline(c, r, np.linspace(0, 2 * np.pi, 145))).astype(np.int32)], True,
                  (255, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(vis, [np.round(outline(c, r, phis)).astype(np.int32)], False, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(vis, [np.round(outline(c0, r, np.linspace(0, 2 * np.pi, 145))).astype(np.int32)], True,
                  (255, 255, 0), 1, cv2.LINE_AA)
    c_ray, r_ray = rays(c0)
    print(f"{'':12s} the outermost light line on rays: desk "
          f"({np.linalg.solve(D2T, c_ray)[0]:6.2f}, {np.linalg.solve(D2T, c_ray)[1]:6.2f}), radius {r_ray:5.2f} mm: "
          f"{np.linalg.norm(c_ray - c):.1f} mm from the circle above")
    cv2.polylines(vis, [np.round(outline(c_ray, r_ray, np.linspace(0, 2 * np.pi, 145))).astype(np.int32)], True,
                  (60, 60, 255), 1, cv2.LINE_AA)
fr, fl, br = (found[k][0] for k in ("front-right", "front-left", "back-right"))
a, b = np.linalg.norm(fl - fr), np.linalg.norm(br - fr)
angle = math.degrees(math.acos((fl - fr) @ (br - fr) / a / b))
centre = np.linalg.solve(D2T, (fl + br) / 2)
print(f"square: sides {a:.1f} (front pair) and {b:.1f} (right pair) mm, corner {angle:.1f} deg; centre desk "
      f"({centre[0]:.2f}, {centre[1]:.2f}), {np.hypot(*centre):.2f} mm from desk (0, 0)")
for radius in (95, 100):
    for name in ("front-right", "front-left"):
        c = np.linalg.solve(D2T, found[name][0])
        print(f"  the {radius} mm arc passes {radius - np.hypot(*c):5.2f} mm outside the {name} cup's centre")
o = project(fit.desk_to_pixel, np.array([[0.0, 0, 0]]))[0]
s = project(fit.desk_to_pixel, np.array([[*centre, 0.0]]))[0]
cv2.drawMarker(vis, tuple(np.round(o).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
cv2.drawMarker(vis, tuple(np.round(s).astype(int)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 2)
for radius, colour in ((95, (0, 140, 255)), (100, (255, 140, 76))):
    b_ = np.radians(np.linspace(-120, 120, 481))
    pts = project(fit.desk_to_pixel, np.column_stack((radius * np.sin(b_), -radius * np.cos(b_), np.zeros(481))))
    cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, colour, 1, cv2.LINE_AA)
crop = vis[380:1100, 420:1620].copy()
cv2.rectangle(crop, (0, 0), (1200, 90), (0, 0, 0), -1)
cv2.putText(crop, "magenta: rim as the best-followed circle (green: the part scored); red: rim as the outermost light "
            "line on rays", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(crop, "cyan: where camera.json puts the cups",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(crop, "orange: 95 mm arc   blue: 100 mm arc   yellow +: desk (0,0)   magenta x: the found cups' centre",
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(prefix + "-cups.png", crop)
