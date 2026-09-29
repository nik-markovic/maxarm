#!/usr/bin/env python3
"""Ask the tiles whether the top-down view is stretched, and how thick they are.

Every single tile is fitted as a 20 x 17.5 mm box whose side faces the camera
sees as a band, and the whole scene shares four unknowns besides each tile's
own position and angle: a scale along each view axis, a shear, and the tile
thickness that sets the band's width. The tiles lie at many angles, so a
stretch along one axis of the view and a band that always points at the camera
pull the fit in different directions and can be told apart.

The silhouette is a brightness cut, which is fine for measuring on a dark desk
and is not how the detector finds tiles.

    ./agenttools/fit-aspect.py files/tiles-1.png
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("measure", HERE / "measure-tiles.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)

import cv2                                   # noqa: E402

import topdown                               # noqa: E402
from mapping import DeskMapping              # noqa: E402

TILE_MM = (20.0, 17.5)


def silhouette(view, camera_xyz, correction, thickness, pose):
    centre, angle = pose[:2], pose[2]
    c, s = np.cos(angle), np.sin(angle)
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * (TILE_MM[0] / 2, TILE_MM[1] / 2)
    top = centre + half @ np.array([[c, s], [-s, c]])
    bottom = top + (thickness / camera_xyz[2]) * (camera_xyz[:2] - top)
    # The correction acts on true millimetres to give what the view shows.
    shown = (np.vstack((top, bottom)) - centre) @ correction.T + centre
    points = np.array([view.to_view(tuple(p)) for p in shown], np.float32)
    return cv2.convexHull(points)


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    mapping = DeskMapping.load(measure.ROOT / "config" / "calibration.json")
    view = topdown.render(frame, mapping, measure.TILE_HEIGHT_MM)
    _, _, vt = np.linalg.svd(mapping.camera())
    camera_xyz = vt[-1][:3] / vt[-1][3]
    tiles = tile_masks(view)
    print(f"{len(tiles)} single tiles")

    poses = [None] * len(tiles)

    def total(globals_):
        # globals_: scale along arm x, scale along arm y, shear, thickness
        correction = np.array([[globals_[0], globals_[2]], [globals_[2], globals_[1]]])
        score = 0.0
        for index, (region, origin, start) in enumerate(tiles):
            starts = [poses[index]] if poses[index] is not None else \
                [(*start, a) for a in np.radians(np.arange(0, 180, 15))]
            best = None
            for s in starts:
                pose, value = measure.refine(
                    lambda p: measure.iou(region, silhouette(view, camera_xyz, correction,
                                                             globals_[3], p), origin),
                    s, (0.5, 0.5, np.radians(2)), rounds=30)
                if best is None or value > best[1]:
                    best = (pose, value)
            poses[index] = best[0]
            score += best[1]
        return score / len(tiles)

    for label, start, steps in (
            ("nominal: no correction, 4 mm thick", (1, 1, 0, 4), (0, 0, 0, 0)),
            ("thickness free", (1, 1, 0, 4), (0, 0, 0, 0.5)),
            ("scale and shear free, 4 mm thick", (1, 1, 0, 4), (0.02, 0.02, 0.02, 0)),
            ("all four free", (1, 1, 0, 4), (0.02, 0.02, 0.02, 0.5))):
        poses[:] = [None] * len(tiles)
        if any(steps):
            total(start)
            fitted, value = measure.refine(total, start, steps, rounds=12)
        else:
            fitted, value = np.array(start, float), total(start)
        print(f"{label:38s} mean IoU {value:.4f}  x-scale {fitted[0]:.3f}  y-scale {fitted[1]:.3f}"
              f"  shear {fitted[2]:+.3f}  thickness {fitted[3]:.2f} mm")
    return 0


def tile_masks(view):
    lightness = cv2.cvtColor(view.image, cv2.COLOR_BGR2LAB)[..., 0]
    per_mm = view.px_per_mm
    disk = lambda mm: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(mm * per_mm / 2) + 1,) * 2)
    closed = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, disk(3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, disk(12))
    valid = cv2.erode((view.image.sum(2) > 0).astype(np.uint8), np.ones((9, 9))) > 0
    upper = opened[valid & (opened > np.median(opened[valid]))]
    threshold, _ = cv2.threshold(upper.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seeds, fine = (opened > threshold) & valid, (closed > threshold) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds.astype(np.uint8))
    tiles = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area > 1.5 * 22 * 20 * per_mm ** 2:
            continue
        pad = round(8 * per_mm)
        y0, x0 = max(0, y - pad), max(0, x - pad)
        region = (cv2.dilate((labels[y0:y + h + pad, x0:x + w + pad] == label).astype(np.uint8), disk(4))
                  & fine[y0:y + h + pad, x0:x + w + pad]).astype(np.uint8)
        moments = cv2.moments(region)
        centre = view.to_ground((moments["m10"] / moments["m00"] + x0, moments["m01"] / moments["m00"] + y0))
        tiles.append((region, np.array([x0, y0]), centre))
    return tiles


if __name__ == "__main__":
    sys.exit(main())
