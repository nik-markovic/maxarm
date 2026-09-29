#!/usr/bin/env python3
"""Fit a box to every single-tile blob with its width and depth left free.

The tiles are 20 x 17.5 mm by the owner's ruler. What the fit reports is what
the calibrated top-down view says they are, which is a check of that view's
scale along each axis -- an aspect error would show as the fitted size
following the tile's angle to the view, not the tile.

    ./agenttools/measure-tiles.py files/tiles-1.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import topdown                          # noqa: E402
from mapping import DeskMapping         # noqa: E402

TILE_HEIGHT_MM = 4.0


def silhouette(view, camera_xyz, centre, angle, width, depth):
    """The tile's outline in the z=4 view: top face, plus the bottom face as the camera sees it."""
    c, s = np.cos(angle), np.sin(angle)
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * (width / 2, depth / 2)
    top = centre + half @ np.array([[c, s], [-s, c]])
    bottom = top + (TILE_HEIGHT_MM / camera_xyz[2]) * (camera_xyz[:2] - top)
    points = np.array([view.to_view(tuple(p)) for p in np.vstack((top, bottom))], np.float32)
    return cv2.convexHull(points)


def iou(mask, polygon, origin):
    drawn = np.zeros_like(mask)
    cv2.fillPoly(drawn, [np.round(polygon - origin).astype(np.int32)], 1)
    union = np.count_nonzero(drawn | mask)
    return np.count_nonzero(drawn & mask) / union if union else 0.0


def refine(score, start, steps, rounds=60):
    """Coordinate descent with shrinking steps; no scipy on the target."""
    best, value = np.array(start, float), score(start)
    steps = np.array(steps, float)
    for _ in range(rounds):
        improved = False
        for axis in range(len(best)):
            for sign in (1, -1):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                trial_value = score(trial)
                if trial_value > value:
                    best, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
    return best, value


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    view = topdown.render(frame, mapping, TILE_HEIGHT_MM)
    camera = mapping.camera()
    _, _, vt = np.linalg.svd(camera)
    camera_xyz = vt[-1][:3] / vt[-1][3]

    lightness = cv2.cvtColor(view.image, cv2.COLOR_BGR2LAB)[..., 0]
    per_mm = view.px_per_mm
    disk = lambda mm: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(mm * per_mm / 2) + 1,) * 2)
    closed = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, disk(3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, disk(12))
    valid = cv2.erode((view.image.sum(2) > 0).astype(np.uint8), np.ones((9, 9))) > 0
    upper = opened[valid & (opened > np.median(opened[valid]))]
    threshold, _ = cv2.threshold(upper.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seeds = (opened > threshold) & valid
    fine = (closed > threshold) & valid

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds.astype(np.uint8))
    print(f"threshold L*={threshold:.0f}; camera at {np.round(camera_xyz, 1)}")
    print("  view px      ground mm        width  depth  angle   IoU   band")
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area > 1.5 * 22 * 20 * per_mm ** 2:
            print(f"  ({x + w // 2},{y + h // 2}) {area / per_mm ** 2:.0f} mm^2: more than one tile, skipped")
            continue
        pad = round(8 * per_mm)
        y0, x0 = max(0, y - pad), max(0, x - pad)
        region = (cv2.dilate((labels[y0:y + h + pad, x0:x + w + pad] == label).astype(np.uint8),
                             disk(4)) & fine[y0:y + h + pad, x0:x + w + pad]).astype(np.uint8)
        contour = max(cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
        (cx, cy), _, _ = cv2.minAreaRect(contour)
        centre = np.array(view.to_ground((cx + x0, cy + y0)))
        origin = np.array([x0, y0])
        best = None
        for start_angle in np.radians(np.arange(0, 180, 15)):
            params, value = refine(
                lambda p: iou(region, silhouette(view, camera_xyz, p[:2], p[2], p[3], p[4]), origin),
                (*centre, start_angle, 20.0, 17.5), (1.0, 1.0, np.radians(4), 1.0, 1.0))
            if best is None or value > best[1]:
                best = (params, value)
        (gx, gy, angle, width, depth), value = best
        if width < depth:
            width, depth, angle = depth, width, angle + np.pi / 2
        band = TILE_HEIGHT_MM * np.hypot(*(camera_xyz[:2] - (gx, gy))) / camera_xyz[2]
        print(f"  ({cx + x0:4.0f},{cy + y0:4.0f})  ({gx:7.1f},{gy:7.1f})  {width:5.1f}  {depth:5.1f}"
              f"  {np.degrees(angle) % 180:5.1f}  {value:.3f}  {band:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
