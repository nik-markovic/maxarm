"""Grid paper's crossings in a frame, to the sub-pixel, and the lattice index of each. No arm.

python grid-crossings.py frame.png start_camera.json out.npz
start_camera: a 3x4 true-mm-to-pixel camera good to a few squares near the sheet's middle.
The lattice is grown from the sheet's middle outward, a homography refitted at each ring.
"""
import json
import math
import sys

import cv2
import numpy as np

SQUARE_MM = 5.0
PX_PER_MM = 8.0

frame_path, camera_path, out_path = sys.argv[1:4]
frame = cv2.imread(frame_path)
stored = json.load(open(camera_path))
P = np.array(stored["desk_camera"] if "desk_camera" in stored else stored["true_to_pixel"])
H = P[:, [0, 1, 3]]
red = frame[..., 2].astype(np.float32)
dark = cv2.GaussianBlur(red, (0, 0), 6.0) - cv2.GaussianBlur(red, (0, 0), 0.8)


def sample(img, pts):
    return cv2.remap(img, pts[:, 0].astype(np.float32).reshape(-1, 1), pts[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()


def apply(M, pts):
    h = np.column_stack((pts, np.ones(len(pts)))) @ M.T
    return h[:, :2] / h[:, 2:]


# top-down view of the darkness over the desk
inv = np.linalg.inv(H)
c = apply(inv, np.array([[0, 0], [2591, 0], [2591, 1943], [0, 1943]], float))
x0, y0 = np.maximum(c.min(0), (-400, -500))
x1, y1 = np.minimum(c.max(0), (400, 300))
W, Ht = int((x1 - x0) * PX_PER_MM), int((y1 - y0) * PX_PER_MM)
view_to_true = np.array([[1 / PX_PER_MM, 0, x0], [0, 1 / PX_PER_MM, y0], [0, 0, 1]])
M = H @ view_to_true                                       # view px -> frame px
view = cv2.warpPerspective(dark, M, (W, Ht), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
paper = cv2.warpPerspective(red, M, (W, Ht), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
_, bright = cv2.threshold(cv2.GaussianBlur(paper, (0, 0), 6).astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
sheet = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
sheet = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(6 * PX_PER_MM) | 1,) * 2))
sheet_frame = cv2.warpPerspective(sheet, M, (2592, 1944), flags=cv2.INTER_NEAREST)
ys, xs = np.nonzero(sheet)
cy, cx = int(np.median(ys)), int(np.median(xs))
S = 512
patch = view[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2]
spec = np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * np.outer(np.hanning(S), np.hanning(S)))))
f0 = S / (SQUARE_MM * PX_PER_MM)
yy, xx = np.mgrid[-S // 2:S // 2, -S // 2:S // 2]
rr = np.hypot(xx, yy)
spec[(rr < 0.7 * f0) | (rr > 1.3 * f0)] = 0
k = np.unravel_index(np.argmax(spec), spec.shape)
fy, fx = k[0] - S // 2, k[1] - S // 2
theta = math.atan2(fy, fx)
pitch_mm = S / math.hypot(fx, fy) / PX_PER_MM
print(f"lattice turn {math.degrees(theta) % 90:.2f} deg, pitch {pitch_mm:.2f} mm in the start camera's mm")
u = np.array([math.cos(theta), math.sin(theta)])
v = np.array([-math.sin(theta), math.cos(theta)])
centre_true = (view_to_true @ (cx, cy, 1))[:2]


def start_map(phase):
    """lattice (i, j) -> frame px, as a 3x3."""
    T = np.array([[pitch_mm * u[0], pitch_mm * v[0], centre_true[0] + phase[0] * u[0] + phase[1] * v[0]],
                  [pitch_mm * u[1], pitch_mm * v[1], centre_true[1] + phase[0] * u[1] + phase[1] * v[1]],
                  [0, 0, 1]])
    return H @ T


ij = np.array([(i, j) for i in range(-4, 5) for j in range(-4, 5)], float)
best = None
for a in np.arange(0, pitch_mm, pitch_mm / 24):
    for b in np.arange(0, pitch_mm, pitch_mm / 24):
        score = sample(dark, apply(start_map((a, b)), ij)).mean()
        if best is None or score > best[0]:
            best = (score, (a, b))
G = start_map(best[1])


def refine(ijs, G):
    """Each crossing refined: both lines traced either side of it, then intersected."""
    out_ij, out_px = [], []
    for i, j in ijs:
        p = apply(G, np.array([[i, j]]))[0]
        xi, yi = int(round(p[0])), int(round(p[1]))
        if not (25 < xi < 2567 and 25 < yi < 1919) or not sheet_frame[yi, xi]:
            continue
        a = apply(G, np.array([[i - 0.5, j], [i + 0.5, j], [i, j - 0.5], [i, j + 0.5]]))
        lines = []
        for dvec in (a[1] - a[0], a[3] - a[2]):
            size = np.linalg.norm(dvec)
            d = dvec / size
            nrm = np.array([-d[1], d[0]])
            offs = np.linspace(-0.25 * size, 0.25 * size, 41)
            step = offs[1] - offs[0]
            found = []
            for s in np.concatenate((np.linspace(-0.42, -0.14, 9), np.linspace(0.14, 0.42, 9))):
                prof = sample(dark, p + s * size * d + offs[:, None] * nrm)
                m = int(np.argmax(prof))
                if 0 < m < len(offs) - 1 and prof[m] > 1.0:
                    y0_, y1_, y2_ = prof[m - 1:m + 2]
                    den = y0_ - 2 * y1_ + y2_
                    sub = 0.5 * (y0_ - y2_) / den if den < 0 else 0.0
                    found.append(p + s * size * d + (offs[m] + sub * step) * nrm)
            if len(found) < 12:
                break
            found = np.array(found)
            mean = found.mean(0)
            _, _, vt = np.linalg.svd(found - mean)
            off = (found - mean) @ vt[1]
            keep = np.abs(off) < max(0.6, 3 * np.median(np.abs(off)))
            if keep.sum() < 10:
                break
            mean = found[keep].mean(0)
            _, _, vt = np.linalg.svd(found[keep] - mean)
            lines.append((mean, vt[0]))
        if len(lines) < 2:
            continue
        (m1, d1), (m2, d2) = lines
        A = np.column_stack((d1, -d2))
        if abs(np.linalg.det(A)) < 0.3:
            continue
        t = np.linalg.solve(A, m2 - m1)
        q = m1 + t[0] * d1
        size = min(np.linalg.norm(a[1] - a[0]), np.linalg.norm(a[3] - a[2]))
        if np.linalg.norm(q - p) < 0.25 * size:
            out_ij.append((i, j))
            out_px.append(q)
    return np.array(out_ij, float), np.array(out_px)


for radius in (4, 8, 14, 22, 32, 45, 60, 80):
    ijs = [(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
    found_ij, found_px = refine(ijs, G)
    if len(found_ij) >= 8:
        G, _ = cv2.findHomography(found_ij, found_px, cv2.RANSAC, 2.0)
    print(f"radius {radius}: {len(found_ij)} crossings")
np.savez(out_path, ij=found_ij, px=found_px, square_mm=SQUARE_MM)
vis = frame.copy()
for q in found_px:
    cv2.circle(vis, tuple(np.round(q).astype(int)), 4, (0, 0, 255), -1)
cv2.imwrite(out_path.replace(".npz", ".jpg"), cv2.resize(vis, (1296, 972)), [cv2.IMWRITE_JPEG_QUALITY, 85])
