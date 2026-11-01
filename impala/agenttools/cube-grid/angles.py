#!/usr/bin/env python3
"""The corner angles perspective gives a cube's top and bottom, and the grid's cells, on the snapshot. No arm.

    .venv/bin/python impala/agenttools/cube-grid/angles.py frame.png

Writes grid40.jpg (Fine Tune's grid drawn 40 mm up instead of on the desk) and angles.jpg (round each cube: the grid
40 mm up, cyan; on the desk, magenta, faint; the cube as fitted, its top square orange, its bottom yellow, each
corner's angle in the picture). Prints a 10 mm cell's corner angles across the frame, on the desk and 40 mm up.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera.model import cube_corners                                      # noqa: E402

CUBES = {"red": (-106.1, -69.4, 89.8), "blue": (21.1, -232.7, 0.4)}       # cubes.py on frame.png: true mm, turn
CROP = 300


def main():
    frame = cv2.imread(str(HERE / sys.argv[1]))
    camera = json.loads((ROOT / "config" / "camera.json").read_text())
    true_to_pixel = np.array(camera["true_to_pixel"])
    desk_to_pixel = np.array(camera["desk_to_pixel"])

    def px(point, matrix=true_to_pixel):
        p = matrix @ (*point, 1.0)
        return p[:2] / p[2]

    def on_plane(pixel, z, matrix=true_to_pixel):
        plane = np.column_stack((matrix[:, 0], matrix[:, 1], matrix[:, 3] + z * matrix[:, 2]))
        x, y, w = np.linalg.solve(plane, (*pixel, 1.0))
        return np.array([x / w, y / w])

    def angles(quad):
        result = []
        for i in range(4):
            a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
            u, v = a - b, c - b
            result.append(math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        return result

    def cell(centre, z, matrix=true_to_pixel):
        x, y = centre
        return np.array([px((x + dx, y + dy, z), matrix) for dx, dy in ((-5, -5), (5, -5), (5, 5), (-5, 5))])

    # a 10 mm cell's angles across the frame: on the desk, and 40 mm up, at the same place in the picture
    print("a 10 mm square's corner angles in the picture, at places across the frame (desk / 40 mm up, same place):")
    for v in (300, 768, 1250):
        row = []
        for u in (400, 1152, 1900):
            on_desk, up = on_plane((u, v), 0.0), on_plane((u, v), 40.0)
            a0, a40 = angles(cell(on_desk, 0.0)), angles(cell(up, 40.0))
            row.append(f"({u:4d},{v:4d}) {a0[0]:5.1f}/{a0[1]:5.1f} | {a40[0]:5.1f}/{a40[1]:5.1f}")
        print("   " + "   ".join(row))

    # Fine Tune's grid drawn 40 mm up, the whole frame
    picture = frame.copy()
    steps = np.arange(-300, 301, 10)
    for k in steps:
        width = 2 if k % 50 == 0 else 1
        for line in ([px((k, a, 40.0), desk_to_pixel) for a in steps], [px((a, k, 40.0), desk_to_pixel) for a in steps]):
            cv2.polylines(picture, [np.round(np.array(line) * 4).astype(np.int32)], False, (255, 255, 0), width,
                          cv2.LINE_AA, shift=2)
    cv2.putText(picture, "Fine Tune's grid drawn 40 mm above the desk (10 mm squares)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(HERE / "grid40.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # round each cube
    tiles = []
    for name, (x, y, turn) in CUBES.items():
        picture = frame.copy()
        corners = cube_corners(np.array([x, y]), math.radians(turn))
        bottom = np.array([px(c) for c in corners[:4]])
        top = np.array([px(c) for c in corners[4:]])
        centre = np.array([x, y])
        for z, ink, width in ((0.0, (200, 0, 200), 1), (40.0, (255, 255, 0), 1)):
            for k in np.arange(-70, 71, 10):
                along = np.arange(-70, 71, 5)
                for line in ([px((x + k, y + t, z)) for t in along], [px((x + t, y + k, z)) for t in along]):
                    cv2.polylines(picture, [np.round(np.array(line) * 4).astype(np.int32)], False, ink, width,
                                  cv2.LINE_AA, shift=2)
        for quad, ink in ((bottom, (0, 255, 255)), (top, (0, 140, 255))):
            cv2.polylines(picture, [np.round(quad * 4).astype(np.int32)], True, ink, 3, cv2.LINE_AA, shift=2)
            middle = quad.mean(axis=0)
            for corner, angle in zip(quad, angles(quad)):
                at = corner + 0.22 * (middle - corner) - np.array([28, -10])
                cv2.putText(picture, f"{angle:.1f}", tuple(np.round(at).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(picture, f"{angle:.1f}", tuple(np.round(at).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            ink, 2, cv2.LINE_AA)
        u, v = px((x, y, 20.0))
        x0 = int(np.clip(u - CROP, 0, frame.shape[1] - 2 * CROP))
        y0 = int(np.clip(v - CROP, 0, frame.shape[0] - 2 * CROP))
        tile = picture[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP]
        cv2.putText(tile, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(tile)
        print(f"{name}: bottom corners {', '.join(f'{a:.1f}' for a in angles(bottom))}; top "
              f"{', '.join(f'{a:.1f}' for a in angles(top))}")
    sheet = np.hstack(tiles)
    band = np.zeros((90, sheet.shape[1], 3), np.uint8)
    for row, text in enumerate(("cyan: grid 40 mm up   magenta: grid on the desk   orange: cube top as fitted   "
                                "yellow: cube bottom", "numbers: each corner's angle in the picture, degrees")):
        cv2.putText(band, text, (10, 34 + 38 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(HERE / "angles.jpg"), np.vstack((band, sheet)), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
