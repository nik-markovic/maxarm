#!/usr/bin/env python3
"""Where the cup's rim is in a live frame: the desk x, y whose rim (21 mm across, at height z) lies best on the
frame's edges, through config/camera.json. Only the reach across the camera's line of sight is clean: a height
error moves the rim along that line.

    rim-fit.py FRAME X Y Z [--out picture.png]     (X Y: where to start, desk mm; Z: the rim's height, desk mm)
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

CAMERA = Path(__import__("os").environ.get("CAMERA_JSON") or Path(__file__).resolve().parents[2] / "config" / "camera.json")
RIM_MM = float(__import__("os").environ.get("RIM_MM", "10.5"))


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("frame"); args.add_argument("x", type=float); args.add_argument("y", type=float)
    args.add_argument("z", type=float); args.add_argument("--out")
    a = args.parse_args()
    cam = json.loads(CAMERA.read_text())
    P = np.array(cam["desk_to_pixel"]); T2D = np.linalg.inv(np.array(cam["desk_to_true"]))
    im = cv2.imread(a.frame)
    best, rim, project = fit(im, P, T2D, a.x, a.y, a.z)
    print(json.dumps({"x": round(best[1], 2), "y": round(best[2], 2), "score": round(best[0], 1)}))
    if a.out:
        picture(im, rim, project, a, best)


def fit(im, P, T2D, x0, y0, z):
    """(score, x, y) of the rim that lies best on `im`'s edges, starting from x0, y0; and the rim and projection."""
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    turns = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ring = (T2D @ np.vstack([RIM_MM * np.cos(turns), RIM_MM * np.sin(turns)])).T

    def project(points):
        h = np.c_[points, np.ones(len(points))] @ P.T
        return h[:, :2] / h[:, 2:]

    def rim(x, y):
        return project(np.c_[ring + (x, y), np.full(len(ring), z)])

    def score(x, y):
        pts = rim(x, y)
        centre = project(np.array([[x, y, z]]))[0]
        normal = pts - centre
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        keep = normal[:, 0] > -0.2                      # the side away from the cup's body (image right)
        u, v = pts[keep, 0].astype(np.float32), pts[keep, 1].astype(np.float32)
        sx = cv2.remap(gx, u.reshape(1, -1), v.reshape(1, -1), cv2.INTER_LINEAR)[0]
        sy = cv2.remap(gy, u.reshape(1, -1), v.reshape(1, -1), cv2.INTER_LINEAR)[0]
        return float(np.mean(sx * normal[keep, 0] + sy * normal[keep, 1]))   # dark cup -> bright paper outward

    best = (-1e9, x0, y0)
    for step, span in ((0.5, 10.0), (0.1, 1.0)):
        cx, cy = best[1], best[2]
        for x in np.arange(cx - span, cx + span + 1e-9, step):
            for y in np.arange(cy - span, cy + span + 1e-9, step):
                s = score(x, y)
                if s > best[0]:
                    best = (s, x, y)
    return best, rim, project


def picture(im, rim, project, a, best) -> None:
    c = project(np.array([[best[1], best[2], a.z]]))[0].astype(int)
    crop_at = np.array([c[0] - 110, c[1] - 80])
    crop = cv2.resize(im[crop_at[1]:crop_at[1] + 160, crop_at[0]:crop_at[0] + 220], None, fx=4, fy=4,
                      interpolation=cv2.INTER_CUBIC)
    for pts, ink in ((rim(a.x, a.y), (255, 155, 61)), (rim(best[1], best[2]), (0, 140, 255))):
        cv2.polylines(crop, [np.round((pts - crop_at) * 4).astype(np.int32)], True, ink, 2, cv2.LINE_AA)
    for i, (text, ink) in enumerate((("blue: where it was sent", (255, 155, 61)),
                                     ("orange: the rim fitted to the frame", (0, 140, 255)))):
        for width, colour in ((4, (0, 0, 0)), (2, ink)):
            cv2.putText(crop, text, (12, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, width, cv2.LINE_AA)
    cv2.imwrite(a.out, crop)


if __name__ == "__main__":
    main()
