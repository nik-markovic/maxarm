"""Where the arm's turning axis is, from gecko's pencil circles (STATUS-gecko §9.5), against where a calibration
puts desk (0, 0), the centre of the pages' reach arcs. No arm, no camera.

The circles were drawn round the cup on the desk; seven of them with the firmware's reach 197-201 mm, from -90 to
+90 degrees. Whatever the base's turning error, those seven lie on a circle round the axis: their distance from it
is the firmware's reach plus one amount that does not depend on the direction. The circles are found in
gecko's still-3.png through the calibration of the same evening (work/config-old, its frame matched to the still).
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
from maxarm import raw  # noqa: E402

OLD = ROOT.parent / "work" / "config-old"
STILL = ROOT.parent / "gecko" / "agenttools" / "ruler-test" / "still-3.png"
OUT = Path(__file__).resolve().parent
MARKS = {"A": (-201, -1), "B": (-160, -161), "C": (0, -201), "D": (160, -160), "1": (-181, -80), "2": (-80, -181),
         "3": (79, -180), "4": (181, -81), "5": (201, 1), "6": (-159, -1), "7": (-260, -1)}   # firmware x, y (owner)
ON_ARC = ("A", "1", "2", "C", "3", "4", "5")
RADIUS_MM = (9.0, 13.0)
ON_RING_MM = 0.45
SEARCH_MM = 24.0


def load_fit(path):
    return CameraFit.from_json(json.loads(Path(path).read_text()))


fit = load_fit(OLD / "camera.json")             # refined at 19:55 on the 19:36 frame: desk (0, 0) is true (0, 0)
still = cv2.imread(str(STILL))
frame = cv2.imread(str(OLD / "camera-frame.png"))
sift = cv2.SIFT_create(12000)
ka, da = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
kb, db = sift.detectAndCompute(cv2.cvtColor(still, cv2.COLOR_BGR2GRAY), None)
pairs = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
moved, _ = cv2.findHomography(np.float32([ka[p.queryIdx].pt for p in pairs]),
                              np.float32([kb[p.trainIdx].pt for p in pairs]), cv2.RANSAC, 2.0)
back = np.linalg.inv(moved)
plane = fit.true_to_pixel[:, [0, 1, 3]]


def to_mm(pixels):
    on_frame = cv2.perspectiveTransform(np.float32(pixels)[None], back)[0]
    q = np.linalg.solve(plane, np.column_stack((on_frame, np.ones(len(on_frame)))).T).T
    return q[:, :2] / q[:, 2:]


def to_px(mm):
    q = (plane @ np.column_stack((mm, np.ones(len(mm)))).T).T
    return cv2.perspectiveTransform(np.float32(q[:, :2] / q[:, 2:])[None], moved)[0]


hsv = cv2.cvtColor(still, cv2.COLOR_BGR2HSV)
tape = ((hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 70) & (hsv[..., 2] > 70)).astype(np.uint8)
inside = cv2.erode(cv2.morphologyEx(tape, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((7, 7), np.uint8))
dark = cv2.morphologyEx(still[..., 0], cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
rng = np.random.default_rng(0)
found = {}
vis = still.copy()
for name, (x, y) in MARKS.items():
    guess_desk = np.array(raw._true_xy(x, y, 0.0))
    guess_true = fit.desk_to_true @ guess_desk
    c_px = to_px(guess_true[None])[0]
    half = 110
    left, top = int(c_px[0]) - half, int(c_px[1]) - half
    win = dark[top:top + 2 * half, left:left + 2 * half].astype(float)
    tp = inside[top:top + 2 * half, left:left + 2 * half] > 0
    rows, cols = np.nonzero((win > np.percentile(win[tp], 88)) & tp)
    mm = to_mm(np.column_stack((cols + left, rows + top)).astype(float))
    mm = mm[np.linalg.norm(mm - guess_true, axis=1) < SEARCH_MM]
    best = None
    for _ in range(4000):
        a, b, c = mm[rng.choice(len(mm), 3, replace=False)]
        sides = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(sides)) < 1e-9:
            continue
        centre = np.linalg.solve(sides, [b @ b - a @ a, c @ c - a @ a])
        radius = np.linalg.norm(a - centre)
        on = np.abs(np.linalg.norm(mm - centre, axis=1) - radius) < ON_RING_MM
        if RADIUS_MM[0] <= radius <= RADIUS_MM[1] and (best is None or on.sum() > best[2].sum()):
            best = (centre, radius, on)
    centre, radius, on = best
    for _ in range(5):
        ring = mm[on]
        sol = np.linalg.lstsq(np.column_stack((2 * ring, np.ones(len(ring)))), (ring ** 2).sum(1), rcond=None)[0]
        centre, radius = sol[:2], math.sqrt(sol[2] + sol[:2] @ sol[:2])
        on = np.abs(np.linalg.norm(mm - centre, axis=1) - radius) < ON_RING_MM
    cover = len(np.unique(np.floor((np.arctan2(*(mm[on] - centre).T[::-1]) + math.pi) / (2 * math.pi) * 36)))
    found[name] = centre
    print(f"circle {name}: true ({centre[0]:7.2f}, {centre[1]:7.2f}), radius {radius:5.2f} mm, ring {cover}/36")
    t = np.linspace(0, 2 * np.pi, 90)
    cv2.polylines(vis, [to_px(centre + radius * np.column_stack((np.cos(t), np.sin(t)))).astype(np.int32)], True,
                  (0, 0, 255), 2, cv2.LINE_AA)

# the axis: a circle through the seven, each at its own firmware reach plus one common amount
pts = np.array([found[k] for k in ON_ARC])
reach_fw = np.array([math.hypot(*MARKS[k]) for k in ON_ARC])
p = np.array([0.0, 0.0, 7.0])
for _ in range(50):
    def res(q):
        return np.linalg.norm(pts - q[:2], axis=1) - (reach_fw + q[2])
    r0 = res(p)
    J = np.column_stack([(res(p + d) - r0) / 1e-6 for d in np.eye(3) * 1e-6])
    step = np.linalg.lstsq(J, -r0, rcond=None)[0]
    p = p + step
    if np.abs(step).max() < 1e-9:
        break
axis_true, extra = p[:2], p[2]
spread = res(p)
# the same with each point left out, for how sure the centre is
loo = []
for k in range(len(pts)):
    keep = [j for j in range(len(pts)) if j != k]
    q = p.copy()
    for _ in range(50):
        r0 = np.linalg.norm(pts[keep] - q[:2], axis=1) - (reach_fw[keep] + q[2])
        J = np.column_stack([((np.linalg.norm(pts[keep] - (q + d)[:2], axis=1) - (reach_fw[keep] + (q + d)[2])) - r0)
                             / 1e-6 for d in np.eye(3) * 1e-6])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        q = q + step
        if np.abs(step).max() < 1e-9:
            break
    loo.append(q[:2])
loo = np.array(loo)
axis_desk = np.linalg.solve(fit.desk_to_true, axis_true)
print(f"\nthe seven at firmware reach 197-201 lie {extra:+.2f} mm further out than the firmware says, each within "
      f"{np.abs(spread).max():.2f} mm of one circle")
print(f"the axis, from them: desk ({axis_desk[0]:+.2f}, {axis_desk[1]:+.2f}) in the evening's calibration -- "
      f"{np.hypot(*axis_desk):.2f} mm from desk (0, 0), where it draws the arcs")
print(f"  leaving each circle out moves it by up to {np.linalg.norm(loo - axis_true, axis=1).max():.2f} mm")
o = to_px(axis_true[None])[0]
z = to_px(np.zeros((1, 2)))[0]
cv2.drawMarker(vis, tuple(o.astype(int)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 3)
cv2.drawMarker(vis, tuple(z.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
t = np.radians(np.linspace(-120, 120, 241))
for r_, colour in ((200 + extra, (255, 0, 255)),):
    ring = axis_true + r_ * np.column_stack((np.sin(t), -np.cos(t))) @ fit.desk_to_true.T / np.abs(np.linalg.det(fit.desk_to_true)) ** 0.5
    cv2.polylines(vis, [to_px(ring).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
np.save(OUT / "circles-true.npy", {"found": found, "axis_true": axis_true, "extra": extra}, allow_pickle=True)
cv2.putText(vis, "red: pencil circles found; magenta x: the axis they fit, with the 200 mm arc round it; "
            "yellow +: desk (0,0), the evening calibration's arc centre", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            (255, 255, 255), 3)
cv2.imwrite(str(OUT / "circles-axis.jpg"), cv2.resize(vis, (1296, 972)), [cv2.IMWRITE_JPEG_QUALITY, 88])
