#!/usr/bin/env python3
"""The arm's readings as facts: a camera fitted to the points' cube edges with every cube standing where the arm read
the cup over it -- true mm through the library, the board's mirror taken out -- 40 mm high, only its turn free.
Focal length, the lens's middle and the pose come from that. Against the app's way (each cube placed freely, the arm
laid on them after by a turn and a shift). Checked by today's grid sheets, which no fit here sees (the camera's 4 px
turn since 4 Nov 10:23 taken out as in arm-through-real.py), and by each point left out: the camera fitted to the
others, its cube placed by its own edges alone, against its arm reading.

    pinned.py
"""
import importlib.util
import math
import pickle
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


onedim = load("onedim", "one-dimension.py")
tj = load("tj", "today-joint.py")
from applib.camera import edges  # noqa: E402
from applib.camera.fit import HANDEDNESS  # noqa: E402
from applib.camera.model import cube_corners  # noqa: E402

JACKAL = tj.JACKAL

# the camera's motion, 4 Nov frame -> today's, from bare desk (as arm-through-real.py)
then = cv2.imread(str(JACKAL / "config" / "camera-frame.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
now = cv2.imread(str(HERE / "out" / "scene.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
src, dst = [], []
for x0, y0, x1, y1 in [(60, 1330, 700, 1470), (800, 1400, 1400, 1470), (1050, 0, 1240, 120), (2000, 0, 2300, 200),
                       (1450, 1420, 2100, 1472), (2250, 250, 2540, 520)]:
    a, b = then[y0:y1, x0:x1], now[y0:y1, x0:x1]
    (dx, dy), peak = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    if peak >= 0.3:
        c = ((x0 + x1) / 2, (y0 + y1) / 2)
        src.append(c); dst.append((c[0] + dx, c[1] + dy))
A = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T      # today = A @ [then, 1]
TODAY_SHEETS = list(tj.SHEETS)
THEN_SHEETS = [(np.linalg.solve(A[:, :2], (pix - A[:, 2]).T).T, mm, is_x) for pix, mm, is_x in tj.SHEETS]


def as9(camera):
    return np.array([camera[0], camera[2], camera[3], *camera[4:10]])


def describe(camera):
    R = cv2.Rodrigues(np.array(camera[4:7], float))[0]
    lens = -R.T @ np.array(camera[7:10], float)
    axis = R.T @ (0, 0, 1.0)
    return (f"f {camera[0]:6.1f}, middle ({camera[2]:6.1f}, {camera[3]:5.1f}), lens {lens[2]:5.1f} up, "
            f"looks {math.degrees(math.asin(-axis[2])):4.1f} deg down")


def sheets_read(camera, frame="then"):
    """How the camera's desk mm read today's two sheets' 5 mm squares, along each sheet's two axes."""
    tj.SHEETS[:] = THEN_SHEETS if frame == "then" else TODAY_SHEETS
    x = np.zeros(tj.SHEET0 + 3 * len(tj.SHEETS))
    x[:tj.CAM] = as9(camera)
    x = tj.sheet_poses(x)
    r = [tj.reads(x, k) for k in range(len(tj.SHEETS))]
    flat = max(np.median(np.abs(tj.sheet_res(x, k, tj.FLIP[k]))) for k in range(len(tj.SHEETS)))
    return (f"sheets read: left x {r[0][0]:.4f} y {r[0][1]:.4f}, right x {r[1][0]:.4f} y {r[1][1]:.4f} "
            f"(as flat sheets to {flat:.2f} mm)")


def place(camera, p, start):
    x = onedim.solve(lambda c: edges.signed_distances(onedim.project(camera, cube_corners(c[:2], c[2])), *p.edges),
                     np.array(start, float), [0, 1, 2])
    return x[:2]


def pinned(rig, keep, prior=True):
    """The camera with each kept point's cube at the arm's reading (true mm), only its turn free."""
    true = rig.arm[keep] @ HANDEDNESS.T
    points = [rig.points[i] for i in keep]
    camera = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
    x = np.concatenate([camera, [rig.starts[i][2] for i in keep]])
    free = [0, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(12, 12 + len(keep)))

    def residuals(x):
        parts = [edges.signed_distances(onedim.project(x[:12], cube_corners(true[k], x[12 + k])), *p.edges)
                 for k, p in enumerate(points)]
        if prior:
            parts.append(np.array([x[2] - rig.width / 2, x[3] - rig.height / 2]) / (0.1 * rig.width))
        return np.concatenate(parts)

    x = onedim.solve(residuals, x, free, 60)
    r = residuals(x)[:-2] if prior else residuals(x)
    return x[:12], float(np.sqrt(np.mean(np.minimum(np.abs(r), 3) ** 2)))


def misses(rig, camera):
    """Every point's cube placed by its edges through `camera` (in the arm's true frame), against its arm reading,
    no turn or shift: out along the reach, round, mm."""
    rows = []
    for p, a, s in zip(rig.points, rig.arm, rig.starts):
        d = HANDEDNESS @ (place(camera, p, s) - HANDEDNESS @ a)
        b = math.atan2(a[0], -a[1])
        rows.append((d @ (math.sin(b), -math.cos(b)), d @ (math.cos(b), math.sin(b))))
    return np.array(rows)


def tie_left_out(rig, camera):
    cubes = np.array([place(camera, p, s) for p, s in zip(rig.points, rig.starts)])
    return rig.arm_misses(cubes)


def main():
    both = pickle.load(open("/tmp/today-joint.pkl", "rb"))["both"]
    sheets_camera = np.array([both[0], 1.0, both[1], both[2], *both[3:9], 0.0, 0.0])
    print(f"the sheets' camera (today's frame): {describe(sheets_camera)}; {sheets_read(sheets_camera, 'today')}")
    rig = onedim.Rig(Path("/tmp/c1104"))
    saved = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
    print(f"4 Nov camera.json:                  {describe(saved)}; {sheets_read(saved)}")
    for name, keep in (("12 points", list(range(len(rig.points)))), ("CALIBRATE's 3 cubes", [0, 1, 2])):
        sub = onedim.Rig.__new__(onedim.Rig); sub.__dict__.update(rig.__dict__)
        sub.points = [rig.points[i] for i in keep]; sub.starts = [rig.starts[i] for i in keep]; sub.arm = rig.arm[keep]
        free, _, edge_free = sub.fit({})
        print(f"\n4 Nov, {name}, cubes placed freely (the app's): edges {edge_free:.3f} px; {describe(free)}\n"
              f"      {sheets_read(free)}")
        _, lo = tie_left_out(rig, free)
        print(f"      all 12 points, the arm laid by a turn and a shift, left out: {np.median(lo):.2f} typ, "
              f"{lo.max():.2f} worst")
        camera, edge = pinned(rig, keep)
        print(f"4 Nov, {name}, cubes where the arm read them:   edges {edge:.3f} px; {describe(camera)}\n"
              f"      {sheets_read(camera)}")
        m = np.hypot(*misses(rig, camera).T)
        _, lo = tie_left_out(rig, camera)
        print(f"      all 12 points' cubes through it against the arm, no turn or shift: {np.median(m):.2f} typ, "
              f"{m.max():.2f} worst; with a turn and a shift laid, left out: {np.median(lo):.2f} typ, "
              f"{lo.max():.2f} worst", flush=True)
        if name == "12 points":
            camera_np, _ = pinned(rig, keep, prior=False)
            print(f"      the same without holding the middle near the frame's: {describe(camera_np)}; "
                  f"{sheets_read(camera_np)}")
            rows = []
            for i in keep:
                cam_i, _ = pinned(rig, [j for j in keep if j != i])
                rows.append(misses(rig, cam_i)[i])
            rows = np.array(rows)
            lo = np.hypot(*rows.T)
            print(f"      each point left out of the fit, its cube through the others' camera against its arm "
                  f"reading: {np.median(lo):.2f} typ, {lo.max():.2f} worst")
            for pid, a, (o, r) in sorted(zip(rig.ids, rig.arm, rows), key=lambda t: math.atan2(t[1][0], -t[1][1])):
                print(f"         {pid:3d} bear {math.degrees(math.atan2(a[0], -a[1])):+6.1f} reach "
                      f"{np.hypot(*a):5.0f}: out {o:+5.2f} round {r:+5.2f}")
    imp = onedim.Rig(Path("/tmp/cimp"))
    keep = list(range(len(imp.points)))
    free, cubes, edge_free = imp.fit({})
    _, lo = imp.arm_misses(cubes)
    print(f"\nimpala 2 Nov, 22 points, free: edges {edge_free:.3f} px; {describe(free)}; left out by turn and shift "
          f"{np.median(lo):.2f} typ, {lo.max():.2f} worst")
    camera, edge = pinned(imp, keep)
    rows = []
    for i in keep:
        cam_i, _ = pinned(imp, [j for j in keep if j != i])
        rows.append(misses(imp, cam_i)[i])
    lo = np.hypot(*np.array(rows).T)
    print(f"impala 2 Nov, 22 points, pinned: edges {edge:.3f} px; {describe(camera)}; each left out of the fit "
          f"{np.median(lo):.2f} typ, {lo.max():.2f} worst")


if __name__ == "__main__":
    main()
