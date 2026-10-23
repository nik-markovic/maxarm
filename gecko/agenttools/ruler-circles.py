#!/usr/bin/env python3
"""Pencil circles drawn round the cup on blue tape, found in a camera still and measured in mm; no arm.

    ../.venv/bin/python agenttools/ruler-circles.py A=-201,-1 B=-160,-161 C=0,-201 D=160,-160
    ../.venv/bin/python agenttools/ruler-circles.py --still agenttools/ruler-test/still.png A=-201,-1 ...

Takes a full-size still (the app must not be showing its live view), or reads one. The camera may have moved
since `camera.json`: the still is matched to `config/camera-frame.png` (SIFT, one homography, the desk being
most of what matches), and its pixels go back through it and the three cubes' camera onto the desk, z = 0.
Near where each mark (x, y as the arm read them) shows, the dark pixels on the tape are fitted with a circle
(RANSAC, radius 9-13 mm). Prints each circle, how much of its ring was found, and the distances between them;
draws agenttools/ruler-test/circles.png.
"""

import itertools
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                      # noqa: E402
from camera.capture import capture                            # noqa: E402
from camera.devices import find_camera                        # noqa: E402

OUT = ROOT / "agenttools" / "ruler-test"
RADIUS_MM = (9.0, 13.0)
ON_RING_MM = 0.45
TRIES = 4000
MIN_RING = 18                       # of 36 parts of the ring, for a circle to count as found


def main(argv) -> int:
    args = argv[1:]
    if args[:1] == ["--still"]:
        new, args = cv2.imread(args[1]), args[2:]
    else:
        new, described = capture(find_camera())
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "still.png"), new)
        print(f"still: {described}, kept as {(OUT / 'still.png').relative_to(ROOT)}")
    marks = {text.split("=")[0]: tuple(map(float, text.split("=")[1].split(","))) for text in args}

    saved = store.load_camera(ROOT / "config")
    plane = np.array(saved.cube_fit().true_to_pixel, float)[:, [0, 1, 3]]    # desk, z = 0, onto the frame
    to_plane = np.linalg.inv(plane)
    desk_to_pixel = np.array(saved.desk_to_pixel, float)
    old = cv2.imread(str(ROOT / "config" / "camera-frame.png"))
    sift = cv2.SIFT_create(12000)
    found_old, described_old = sift.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
    found_new, described_new = sift.detectAndCompute(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), None)
    pairs = [first for first, second in cv2.BFMatcher().knnMatch(described_old, described_new, k=2)
             if first.distance < 0.7 * second.distance]
    before = np.float32([found_old[pair.queryIdx].pt for pair in pairs])
    after = np.float32([found_new[pair.trainIdx].pt for pair in pairs])
    moved, kept = cv2.findHomography(before, after, cv2.RANSAC, 3.0)
    kept = kept.ravel().astype(bool)
    shift = np.linalg.norm(after[kept] - before[kept], axis=1)
    print(f"against camera-frame.png: {kept.sum()} matches, moved {np.median(shift):.1f} px (median)")
    back = np.linalg.inv(moved)

    def to_mm(pixels):
        on_old = cv2.perspectiveTransform(np.float32(pixels)[None], back)[0]
        mapped = (to_plane @ np.column_stack((on_old, np.ones(len(on_old)))).T).T
        return mapped[:, :2] / mapped[:, 2:]

    def to_pixels(mm):
        mapped = (plane @ np.column_stack((mm, np.ones(len(mm)))).T).T
        return cv2.perspectiveTransform(np.float32(mapped[:, :2] / mapped[:, 2:])[None], moved)[0]

    hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
    tape = ((hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    inside = cv2.erode(cv2.morphologyEx(tape, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(new[:, :, 0], cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    generator, circles, tiles = np.random.default_rng(0), {}, []
    for name, (x, y) in marks.items():
        seen = desk_to_pixel @ [x, y, 0.0, 1.0]
        centre_px = cv2.perspectiveTransform(np.float32([[seen[:2] / seen[2]]]), moved)[0, 0]
        left, top = int(centre_px[0]) - 160, int(centre_px[1]) - 120
        window, on_tape = dark[top:top + 240, left:left + 320].astype(float), inside[top:top + 240, left:left + 320] > 0
        if on_tape.sum() < 50:
            print(f"{name}: no tape near ({x:g}, {y:g})")
            continue
        rows, columns = np.nonzero((window > np.percentile(window[on_tape], 88)) & on_tape)
        pixels = np.column_stack((columns + left, rows + top)).astype(float)
        mm = to_mm(pixels)
        best = None
        for _ in range(TRIES):
            a, b, c = mm[generator.choice(len(mm), 3, replace=False)]
            sides = 2 * np.array([b - a, c - a])
            if abs(np.linalg.det(sides)) < 1e-9:
                continue
            centre = np.linalg.solve(sides, [b @ b - a @ a, c @ c - a @ a])
            radius = np.linalg.norm(a - centre)
            on = np.abs(np.linalg.norm(mm - centre, axis=1) - radius) < ON_RING_MM
            if RADIUS_MM[0] <= radius <= RADIUS_MM[1] and (best is None or on.sum() > best[2].sum()):
                best = (centre, radius, on)
        if best is None:
            print(f"{name}: no circle found")
            continue
        centre, radius, on = best
        for _ in range(5):
            ring = mm[on]
            solution = np.linalg.lstsq(np.column_stack((2 * ring, np.ones(len(ring)))), (ring ** 2).sum(1), rcond=None)[0]
            centre, radius = solution[:2], math.sqrt(solution[2] + solution[:2] @ solution[:2])
            on = np.abs(np.linalg.norm(mm - centre, axis=1) - radius) < ON_RING_MM
        angles = np.arctan2(*(mm[on] - centre).T[::-1])
        cover = len(np.unique(np.floor((angles + math.pi) / (2 * math.pi) * 36)))
        if cover < MIN_RING:
            print(f"{name}: no circle found: the best had {cover}/36 of a ring")
            continue
        circles[name] = centre
        print(f"{name}: centre ({centre[0]:.2f}, {centre[1]:.2f}) mm in the cubes' camera, radius {radius:.2f}, "
              f"{cover}/36 of the ring found")
        big = cv2.resize(new[top:top + 240, left:left + 320], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        outline = to_pixels(centre + radius * np.column_stack((np.cos(np.linspace(0, 2 * math.pi, 90)),
                                                                np.sin(np.linspace(0, 2 * math.pi, 90)))))
        cv2.polylines(big, [((outline - [left, top]) * 2).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        middle = to_pixels(centre[None])[0]
        cv2.drawMarker(big, tuple(((middle - [left, top]) * 2).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(big, f"{name}  r {radius:.1f} mm, ring {cover}/36", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
        tiles.append(big)
    for first, second in itertools.combinations(circles, 2):
        print(f"{first}-{second}: {np.linalg.norm(circles[first] - circles[second]):6.1f} mm by the camera, "
              f"{math.dist(marks[first], marks[second]):6.1f} by the arm")
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)]
        OUT.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "circles.png"), np.vstack(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
