"""The owner's 17 calibration points (3 cubes, 14 red cube points) as the arm and the camera have them. No arm."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera.fit import CameraFit  # noqa: E402
from camera.refine import Sighting, seen_true  # noqa: E402
from maxarm import raw  # noqa: E402

CONFIG = ROOT / "config"


def load(config=CONFIG):
    config = Path(config)
    fit = CameraFit.from_json(json.loads((config / "camera.json").read_text()))
    cubes = json.loads((config / "camera-points.json").read_text())["cubes"]
    refine = json.loads((config / "camera-refine-points.json").read_text())["points"]
    desk = json.loads((config / "desk.json").read_text())["desk_to_arm"]
    rows = []
    for colour in ("blue", "green", "red"):
        c = cubes[colour]
        rows.append(dict(name=colour[0].upper(), x=c["x"], y=c["y"], z=c["z"],
                         sight=Sighting(c["x"], c["y"], tuple(fit.cubes[colour]["pixel"]), tuple(fit.frame_size))))
    for p in refine:
        rows.append(dict(name=str(p["id"]), x=p["x"], y=p["y"], z=p["z"],
                         sight=Sighting(p["x"], p["y"], tuple(p["pixel"]), tuple(p["frame_size"]))))
    desk = np.array(desk)
    for r in rows:
        r["cam"] = seen_true(fit, r["sight"])            # true mm, the camera alone (refined frame)
        arm_z = desk[2, 0] * r["x"] + desk[2, 1] * r["y"] + r["z"] + desk[2, 3]
        rx, ry, rz = raw.raw_position((r["x"], r["y"], arm_z))
        r["raw"] = np.array([rx, ry])                    # the firmware's own x, y
        r["arm_z"] = arm_z
    return fit, rows


def fit_affine(src, dst, kind):
    """dst ~ T(src). kind: rigid, similarity, affine, homography. Returns mapped src and params."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if kind == "affine":
        A = np.column_stack((src, np.ones(len(src))))
        sol = np.linalg.lstsq(A, dst, rcond=None)[0]
        return A @ sol, sol
    if kind == "homography":
        import cv2
        H, _ = cv2.findHomography(src, dst, 0)
        h = np.column_stack((src, np.ones(len(src)))) @ H.T
        return h[:, :2] / h[:, 2:], H
    # similarity / rigid by Umeyama, with the mirror allowed (the board's mirror)
    best = None
    for mirror in (False, True):
        s = src * ([-1, 1] if mirror else [1, 1])
        ms, md = s.mean(0), dst.mean(0)
        S, D = s - ms, dst - md
        U, sig, Vt = np.linalg.svd(D.T @ S)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, d]) @ Vt
        scale = (sig * [1, d]).sum() / (S ** 2).sum() if kind == "similarity" else 1.0
        mapped = scale * S @ R.T + md
        err = np.linalg.norm(mapped - dst, axis=1)
        if best is None or err.sum() < best[2].sum():
            best = (mapped, (mirror, scale, math.degrees(math.atan2(R[1, 0], R[0, 0]))), err)
    return best[0], best[1]


if __name__ == "__main__":
    fit, rows = load()
    arm = np.array([(r["x"], r["y"]) for r in rows])
    rawxy = np.array([r["raw"] for r in rows])
    cam = np.array([r["cam"] for r in rows])
    print("point   arm x,y (library)     firmware x,y        reach  bearing   z")
    for r in rows:
        b = math.degrees(math.atan2(r["x"], -r["y"]))
        print(f"{r['name']:>4}  ({r['x']:8.2f},{r['y']:8.2f})  ({r['raw'][0]:8.2f},{r['raw'][1]:8.2f})  "
              f"{math.hypot(r['x'], r['y']):6.1f}  {b:7.1f}  {r['z']:5.1f}")
    for label, src in (("library x,y", arm), ("firmware x,y", rawxy)):
        print(f"\n{label} -> camera true mm")
        for kind in ("rigid", "similarity", "affine", "homography"):
            mapped, params = fit_affine(src, cam, kind)
            err = np.linalg.norm(mapped - cam, axis=1)
            extra = f"  scale {params[1]:.4f}" if kind in ("rigid", "similarity") else ""
            print(f"  {kind:11s} median {np.median(err):5.2f}  mean {err.mean():5.2f}  worst {err.max():5.2f} mm{extra}")
