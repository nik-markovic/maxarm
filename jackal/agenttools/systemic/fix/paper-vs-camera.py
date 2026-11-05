#!/usr/bin/env python3
"""Does a camera fitted to the cubes' edges match reality? The owner's grid paper (3 Nov, 20:24) is a ruler that needs
neither the arm nor the cubes: its line samples tie pixels to paper mm (a flat sheet, 0.15 mm). The same pixels
through a camera's desk plane give that camera's mm. A similarity laid between the two says how the camera's mm
measure against the paper's: scale, and an affine's two stretches.

    paper-vs-camera.py camera.json ...
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paperlens", HERE / "paper-lens.py")
sys.argv, args = sys.argv[:1], sys.argv[1:]
import contextlib, io                                                                  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    paperlens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paperlens)
pix, mm, is_x = paperlens.pix, paperlens.mm, paperlens.is_x
H = paperlens.homography(pix)                                        # paper mm -> pixels
grid = np.array([(x, y) for x in np.arange(-100, 101, 10.0) for y in np.arange(-80, 81, 10.0)])
# where the paper's own millimetres are: the samples' range, in paper mm
back = np.linalg.inv(H) @ np.c_[pix, np.ones(len(pix))].T
paper_xy = (back[:2] / back[2]).T
lo, hi = np.nanpercentile(paper_xy, 2, axis=0), np.nanpercentile(paper_xy, 98, axis=0)
grid = np.array([(x, y) for x in np.linspace(lo[0], hi[0], 15) for y in np.linspace(lo[1], hi[1], 15)])
pixels = (H @ np.c_[grid, np.ones(len(grid))].T)
pixels = (pixels[:2] / pixels[2]).T

for name in args:
    camera = json.loads(Path(name).read_text())
    T = np.array(camera["true_to_pixel"])
    plane = T[:, [0, 1, 3]]
    w = np.linalg.solve(plane, np.c_[pixels, np.ones(len(pixels))].T)
    cam = (w[:2] / w[2]).T                                           # camera mm on its desk plane
    # similarity cam = s R paper + t (either handedness)
    best = None
    for flip in (1.0, -1.0):
        p = grid * (1.0, flip)
        pc, cc = p - p.mean(0), cam - cam.mean(0)
        u, sv, vt = np.linalg.svd(pc.T @ cc)
        d = np.sign(np.linalg.det(u @ vt))
        R = (u @ np.diag([1, d]) @ vt).T
        s = np.sum(sv * np.array([1, d])) / np.sum(pc ** 2)
        r = cc - s * pc @ R.T
        if best is None or np.sum(r ** 2) < best[0]:
            best = (np.sum(r ** 2), s, np.sqrt(np.mean(np.sum(r ** 2, axis=1))), flip)
    A = np.linalg.lstsq(np.c_[grid * (1.0, best[3]), np.ones(len(grid))], cam, rcond=None)[0][:2].T
    stretches = np.linalg.svd(A, compute_uv=False)
    print(f"{name}: camera mm / paper mm = {best[1]:.4f} (similarity, rms {best[2]:.2f} mm left); "
          f"affine stretches {stretches.max():.4f}, {stretches.min():.4f}")


# --- which way the stretch runs: in paper (desk) directions and in the frame ---------------------------------------
print("\nthe stretch's directions")
for name in args:
    camera = json.loads(Path(name).read_text())
    plane = np.array(camera["true_to_pixel"])[:, [0, 1, 3]]
    centre_px = (H @ np.array([*grid.mean(0), 1.0]))
    centre_px = centre_px[:2] / centre_px[2]

    def cam_of(paper_pt):
        q = H @ np.array([paper_pt[0], paper_pt[1], 1.0]); q = q[:2] / q[2]
        w = np.linalg.solve(plane, np.array([q[0], q[1], 1.0]))
        return w[:2] / w[2], q

    rows = []
    for angle in range(0, 180, 15):
        d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * 40.0
        c = grid.mean(0)
        (a, pa), (b, pb) = cam_of(c - d), cam_of(c + d)
        ratio = np.linalg.norm(b - a) / 80.0
        img = pb - pa
        rows.append((ratio, angle, np.degrees(np.arctan2(img[1], img[0])) % 180))
    for ratio, angle, img_angle in sorted(rows, reverse=True):
        print(f"   {name.split('/')[-1]}: paper direction {angle:3d} deg (0 = along paper x, desk x) -> camera reads "
              f"x{ratio:.4f}; in the frame that line runs at {img_angle:5.1f} deg (0 = frame x, 90 = frame y)")
