#!/usr/bin/env python3
"""The camera held nearer, where the owner measured it (about 220 mm up, 210-220 mm across from x = 0), against
SAVE's: how the 8 points' cube edges fit, how the arm lies on them, and how far apart the two put a tile's top
(2 mm) over the work area, out from the base and round it (+ counterclockwise seen from above).

    ../.venv/bin/python agenttools/systemic/owner-camera.py [config_dir]
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import store                                                     # noqa: E402
from applib.camera.model import CUBE_MM, least_squares                              # noqa: E402
from applib.camera.sighting import in_frame, misses, place, start                   # noqa: E402
from camera import refine                                                           # noqa: E402
from maxarm.spaces import Camera                                                    # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
saved = store.load_camera(config)
alone = saved.cube_fit()
points = [p.sighting for p in store.load_refine_points(config, saved)]
seen = [p for p in points if p.edges is not None]
width, height = saved.frame_size
TILE_MM = 2.0


def held_focal(share):
    numbers = refine._numbers(alone.true_to_pixel)
    numbers[0] *= share
    cubes = [start(alone.true_to_pixel, p)[:3] for p in seen]

    def residuals(trial):
        parts = []
        for i, p in enumerate(seen):
            matrix = in_frame(refine._pinhole(trial).matrix, saved.frame_size, p)
            cubes[i] = place(matrix, p, cubes[i])
            parts.append(misses(matrix, p, np.append(cubes[i], CUBE_MM)))
        parts.append((trial[1:3] - (width / 2, height / 2)) / (0.1 * width))
        return np.concatenate(parts)

    numbers = least_squares(residuals, numbers, refine.CAMERA_STEPS, huber=1.0, iterations=40, free=range(1, 9))[0]
    edges_px = float(np.sqrt(np.mean(residuals(numbers)[:-2] ** 2)))
    return refine._laid(alone, refine._pinhole(numbers).matrix, points), edges_px


def report(name, fit, edges_px):
    camera = Camera(fit.desk_to_pixel).position()
    offs = [refine.off_by(fit, p).across for p in points]
    print(f"{name}: lens {camera[2]:.0f} mm up, {camera[0]:.0f} across from x = 0, {math.hypot(*camera[:2]):.0f} from "
          f"the base; edges {edges_px:.2f} px; arm on the points {np.median(offs):.2f} typ, {max(offs):.2f} worst mm")


def compare(fit):
    before, after = Camera(saved.desk_to_pixel), Camera(fit.desk_to_pixel)
    rows = []
    for reach in (110, 150, 190, 230, 260):
        for bearing in range(-80, 81, 20):
            b = math.radians(bearing)
            x, y = reach * math.sin(b), -reach * math.cos(b)
            pixel = before.from_desk((x, y, TILE_MM))
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                continue
            nx, ny, _ = after.to_desk((pixel[0], pixel[1], TILE_MM))
            d = np.array([nx - x, ny - y])
            out = d @ (math.sin(b), -math.cos(b))
            ccw = -(d @ (math.cos(b), math.sin(b)))
            rows.append((reach, bearing, out, ccw))
    rows = np.array(rows)
    print("   a tile seen by SAVE's camera, sent by this one instead: mm further out / further counterclockwise")
    print("   reach  " + " ".join(f"{b:>11d}" for b in range(-80, 81, 20)))
    for reach in (110, 150, 190, 230, 260):
        cells = []
        for bearing in range(-80, 81, 20):
            match = rows[(rows[:, 0] == reach) & (rows[:, 1] == bearing)]
            cells.append(f"{match[0, 2]:+5.1f}/{match[0, 3]:+5.1f}" if len(match) else " " * 11)
        print(f"   {reach:5d}  " + " ".join(cells))
    total = np.hypot(rows[:, 2], rows[:, 3])
    print(f"   apart {np.median(total):.2f} typ, {total.max():.2f} worst mm")


report("SAVE (camera.json)", saved, float("nan"))
for share in (1.0, 0.95, 0.937):
    fit, edges_px = held_focal(share)
    report(f"focal x {share}", fit, edges_px)
    compare(fit)
