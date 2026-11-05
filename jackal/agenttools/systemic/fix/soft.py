#!/usr/bin/env python3
"""The arm's readings as facts with their own scatter: the camera and every cube fitted together to the cubes' edges
(40 mm) and to the arm's readings (true mm through the library, the board's mirror out), each reading held to its cube
within sigma mm. The edges place each cube; whatever they cannot tell apart -- how the desk spreads between the cubes
-- the readings decide together. Checked by today's grid sheets, which no fit sees, and by each point left out.
Also: the 4 Nov points' edges through the camera that matches the sheets, and the camera's motion since 4 Nov.

    soft.py
"""
import importlib.util
import math
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pinned", HERE / "pinned.py")
pn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pn)
onedim, edges, H, cube_corners = pn.onedim, pn.edges, pn.HANDEDNESS, pn.cube_corners


def soft(rig, keep, sigma, first=None):
    true = rig.arm[keep] @ H.T
    points = [rig.points[i] for i in keep]
    camera = first if first is not None else np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
    x = np.concatenate([camera, np.concatenate([[*true[k], rig.starts[i][2]] for k, i in enumerate(keep)])])
    free = [0, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(12, 12 + 3 * len(keep)))

    def residuals(x):
        cubes = x[12:].reshape(-1, 3)
        parts = [edges.signed_distances(onedim.project(x[:12], cube_corners(c[:2], c[2])), *p.edges)
                 for c, p in zip(cubes, points)]
        parts.append(((cubes[:, :2] - true) / sigma).ravel())
        parts.append(np.array([x[2] - rig.width / 2, x[3] - rig.height / 2]) / (0.1 * rig.width))
        return np.concatenate(parts)

    x = onedim.solve(residuals, x, free, 60)
    n = sum(len(p.edges[2]) for p in points)
    r = residuals(x)[:n]
    return x[:12], float(np.sqrt(np.mean(np.minimum(np.abs(r), 3) ** 2)))


def main():
    print("the camera's motion 4 Nov -> today, affine today = A [then, 1]:\n", np.round(pn.A, 5))
    rig = onedim.Rig(Path("/tmp/c1104"))
    both = pickle.load(open("/tmp/today-joint.pkl", "rb"))["both"]
    sheets_camera = np.array([both[0], 1.0, both[1], both[2], *both[3:9], 0.0, 0.0])
    # the 4 Nov edges through the camera that matches the sheets (4 Nov pixels moved into today's)
    fits = []
    for p, s in zip(rig.points, rig.starts):
        starts, ends, found = p.edges
        moved = np.c_[found, np.ones(len(found))] @ pn.A.T
        x = onedim.solve(lambda c: edges.signed_distances(onedim.project(sheets_camera, cube_corners(c[:2], c[2])),
                                                          starts, ends, moved), np.array(s, float), [0, 1, 2])
        r = edges.signed_distances(onedim.project(sheets_camera, cube_corners(x[:2], x[2])), starts, ends, moved)
        fits.append(r)
    r = np.concatenate(fits)
    print(f"4 Nov 12 points' edges, each cube placed freely, through the sheets' camera: "
          f"{np.sqrt(np.mean(np.minimum(np.abs(r), 3) ** 2)):.3f} px (the app's own 12-point camera: see pinned.txt)")
    from applib.camera import store
    saved = store.load_camera(pn.JACKAL / "config")
    nums = pn.refine._numbers(saved.true_to_pixel) if hasattr(pn, "refine") else None
    if nums is None:
        from camera import refine
        nums = refine._numbers(saved.true_to_pixel)
    config_camera = np.array([nums[0], 1.0, *nums[1:9], 0.0, 0.0])
    print(f"config/camera.json (11:45 SAVE) on today's pixels, the motion not taken out: "
          f"{pn.sheets_read(config_camera, 'today')}")
    print(f"config/camera.json (11:45 SAVE), the motion taken out:                    "
          f"{pn.sheets_read(config_camera)}\n")
    keep = list(range(len(rig.points)))
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
        camera, edge = soft(rig, keep, sigma)
        m = np.hypot(*pn.misses(rig, camera).T)
        print(f"12 points, readings held within {sigma:3.1f} mm: edges {edge:.3f} px; {pn.describe(camera)}\n"
              f"      {pn.sheets_read(camera)}; cubes against the arm {np.median(m):.2f} typ, {m.max():.2f} worst",
              flush=True)
        if sigma in (1.0, 2.0):
            rows = []
            for i in keep:
                cam_i, _ = soft(rig, [j for j in keep if j != i], sigma, camera)
                rows.append(pn.misses(rig, cam_i)[i])
            lo = np.hypot(*np.array(rows).T)
            print(f"      each point left out of the fit, against its arm reading: {np.median(lo):.2f} typ, "
                  f"{lo.max():.2f} worst", flush=True)
    for sigma in (1.0, 2.0):
        camera, edge = soft(rig, [0, 1, 2], sigma)
        m = np.hypot(*pn.misses(rig, camera).T)[3:]
        print(f"CALIBRATE's 3 cubes, readings held within {sigma:3.1f} mm: edges {edge:.3f} px; {pn.describe(camera)}\n"
              f"      {pn.sheets_read(camera)}; the other 9 points against the arm {np.median(m):.2f} typ, "
              f"{m.max():.2f} worst", flush=True)


if __name__ == "__main__":
    main()
