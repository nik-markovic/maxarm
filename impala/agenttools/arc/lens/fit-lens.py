"""Grid crossings fitted as a flat lattice seen through a pinhole, with and without radial lens distortion.

python fit-lens.py crossings.npz [focal_px]
The distortion: a pixel lies at c + (q - c)(1 + k1 r^2 + k2 r^4), q where a pinhole puts it, r = |q - c| / focal.
"""
import sys

import cv2
import numpy as np

data = np.load(sys.argv[1])
ij, px = data["ij"].astype(float), data["px"]
focal = float(sys.argv[2]) if len(sys.argv) > 2 else 1950.0
W, Hh = 2592, 1944


def apply(M, pts):
    h = np.column_stack((pts, np.ones(len(pts)))) @ M.T
    return h[:, :2] / h[:, 2:]


def distort(q, centre, k1, k2):
    d = (q - centre) / focal
    r2 = (d ** 2).sum(1, keepdims=True)
    return centre + (q - centre) * (1 + k1 * r2 + k2 * r2 ** 2)


def residuals(params, with_centre):
    Hm = np.append(params[:8], 1.0).reshape(3, 3)
    k1, k2 = params[8], params[9]
    centre = params[10:12] if with_centre else np.array([(W - 1) / 2, (Hh - 1) / 2])
    return (distort(apply(Hm, ij), centre, k1, k2) - px).ravel()


def lm(params, free, with_centre, iters=60):
    params = params.copy()
    lam = 1e-3
    r = residuals(params, with_centre)
    for _ in range(iters):
        J = np.empty((len(r), len(free)))
        for c, k in enumerate(free):
            step = 1e-7 * max(1.0, abs(params[k]))
            p2 = params.copy()
            p2[k] += step
            J[:, c] = (residuals(p2, with_centre) - r) / step
        w = np.where(np.abs(r) < 1.5, 1.0, 1.5 / np.abs(r))          # Huber at 1.5 px
        A = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
            p2 = params.copy()
            p2[free] += step
            r2 = residuals(p2, with_centre)
            if (np.minimum(r2 ** 2, 1.5 * np.abs(r2))).sum() < (np.minimum(r ** 2, 1.5 * np.abs(r))).sum():
                params, r, lam = p2, r2, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return params, r
    return params, r


H0, _ = cv2.findHomography(ij, px, cv2.RANSAC, 3.0)
H0 = H0 / H0[2, 2]
p0 = np.concatenate((H0.ravel()[:8], [0.0, 0.0], [(W - 1) / 2, (Hh - 1) / 2]))
for label, free, with_centre in (("no distortion", list(range(8)), False),
                                 ("k1, centre at the frame's middle", list(range(9)), False),
                                 ("k1 k2, centre at the frame's middle", list(range(10)), False),
                                 ):
    params, r = lm(p0, free, with_centre)
    e = np.hypot(r[0::2], r[1::2])
    inl = e < 3 * np.median(e) + 0.5
    print(f"{label:38s} k1 {params[8]:+.4f} k2 {params[9]:+.4f} centre ({params[10]:.0f}, {params[11]:.0f}): "
          f"median {np.median(e):.3f} px, 90% {np.percentile(e, 90):.3f}, worst {e.max():.2f}  ({inl.sum()}/{len(e)})")
    if label.startswith("no distortion"):
        base = (params.copy(), r.copy())
        np.save(sys.argv[1].replace(".npz", "-plain-res.npy"), np.column_stack((px, r.reshape(-1, 2))))
    last = (params, r)
np.save(sys.argv[1].replace(".npz", "-k-res.npy"), np.column_stack((px, last[1].reshape(-1, 2))))
