import sys, math, numpy as np
sys.path[:0] = ['/var/b/shared/work/maxarm/gecko', '/var/b/shared/work/maxarm/gecko/calibration']
from pathlib import Path
from camera import store
fit = store.load_camera(Path('/var/b/shared/work/maxarm/gecko/config'))
P, A = fit.true_to_pixel, fit.desk_to_true
found = np.load('/tmp/gecko-cal/rim-points.npy', allow_pickle=True).item()
def to_plane(pixels, h=0.0):
    H = np.column_stack((P[:, 0], P[:, 1], P[:, 3] + h * P[:, 2]))
    q = np.linalg.solve(H, np.column_stack((pixels, np.ones(len(pixels)))).T).T
    return q[:, :2] / q[:, 2:]
def circle(pts, radius=None):
    """Least-squares circle (algebraic, then refined), outliers dropped; radius optionally fixed."""
    keep = np.ones(len(pts), bool)
    for _ in range(3):
        p = pts[keep]
        if radius is None:
            A_ = np.column_stack((2 * p, np.ones(len(p)))); b = (p ** 2).sum(1)
            cx, cy, c = np.linalg.lstsq(A_, b, rcond=None)[0]; r = math.sqrt(c + cx * cx + cy * cy); centre = np.array([cx, cy])
        else:
            centre = p.mean(0); r = radius
            for _ in range(50):
                d = p - centre; n = np.linalg.norm(d, axis=1)
                centre = centre + ((n - r)[:, None] * d / n[:, None]).mean(0)
        res = np.linalg.norm(pts - centre, axis=1) - r
        keep = np.abs(res) < max(1.0, 3 * np.median(np.abs(res[keep])))
    return centre, r, np.abs(res[keep]).mean(), keep.sum()
for h in (0.0,):
    print(f'rims taken at desk height {h} (true mm, the camera alone):')
    circles = {}
    for name, px in found.items():
        c, r, res, n = circle(to_plane(px, h))
        circles[name] = (c, r)
        print(f'  {name:5s}: centre true {np.round(c, 1)}, desk {np.round(np.linalg.solve(A, c), 1)}; radius {r:.1f} mm; {n} points, fit within {res:.2f} mm')
    r_front = circles['front'][1]
    print(f'  with every cup the front one\'s radius, {r_front:.1f} mm:')
    for name, px in found.items():
        c, r, res, n = circle(to_plane(px, h), r_front)
        circles[name] = (c, r)
        print(f'  {name:5s}: centre desk {np.round(np.linalg.solve(A, c), 1)}; fit within {res:.2f} mm')
    L, R, F = (circles[k][0] for k in ('left', 'right', 'front'))
    centre = (L + R) / 2
    print(f'  square from the cups: centre desk {np.round(np.linalg.solve(A, centre), 1)} -> {np.linalg.norm(np.linalg.solve(A, centre)):.1f} mm from the arm axis; '
          f'half-diagonal {np.linalg.norm(R - centre):.1f}, front to centre {np.linalg.norm(F - centre):.1f}; angle at front '
          f'{np.degrees(np.arccos((L - F) @ (R - F) / np.linalg.norm(L - F) / np.linalg.norm(R - F))):.1f} deg')
