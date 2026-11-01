#!/usr/bin/env python3
"""The sheet's grid line by line through camera.json: where each line is, and the spacing between neighbours, along
both of the sheet's sides, averaged along the line. Smooth growth from the far side to the near is the camera's
perspective; a change at the sheet's edges alone is the paper not lying flat.

    .venv/bin/python impala/agenttools/lens/grid-lines.py FRAME.png
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pitch", HERE / "grid-pitch.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def line_positions(profile, mm_per_px):
    """The dark lines' places along `profile`, mm, each to a fraction of a sample."""
    signal = profile - np.convolve(profile, np.ones(81) / 81, mode="same")
    places = []
    for index in range(45, len(signal) - 45):
        window = signal[index - 12:index + 13]
        if signal[index] == window.min() and signal[index] < -0.25 * np.abs(signal[45:-45]).max():
            before, here, after = signal[index - 1:index + 2]
            shift = 0.5 * (before - after) / (before - 2 * here + after)
            places.append((index + shift) * mm_per_px)
    return np.array(places)


def main():
    frame = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matrix = np.array(json.loads((grid.CONFIG / "camera.json").read_text())["desk_to_pixel"])
    plane = matrix[:, [0, 1, 3]]
    straight, size = grid.straight, grid.size
    corners = straight.paper_corners(gray)
    lines = [size.line_through(straight.border_points(gray, corners[i], corners[(i + 1) % 4])) for i in range(4)]
    pixels = [size.crossing(lines[i - 1], lines[i]) for i in range(4)]
    desk = [np.linalg.solve(plane, (*pixel, 1.0)) for pixel in pixels]
    desk = [point[:2] / point[2] for point in desk]
    origin = desk[1]
    u = (desk[0] - origin) / np.linalg.norm(desk[0] - origin)
    v = (desk[2] - origin) / np.linalg.norm(desk[2] - origin)
    us = np.arange(3, np.linalg.norm(desk[0] - origin) - 3, 1 / grid.PX_PER_MM)
    vs = np.arange(3, np.linalg.norm(desk[2] - origin) - 3, 1 / grid.PX_PER_MM)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    points = origin + grid_u[..., None] * u + grid_v[..., None] * v
    homogeneous = np.concatenate((points, np.ones(points.shape[:2] + (1,))), axis=2) @ plane.T
    map_x = (homogeneous[..., 0] / homogeneous[..., 2]).astype(np.float32)
    map_y = (homogeneous[..., 1] / homogeneous[..., 2]).astype(np.float32)
    top_down = cv2.remap(gray.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    for name, profile, start, pixel_at in (
            ("u: from the far corner toward the camera (down the picture)", top_down.mean(axis=1), us[0],
             lambda mm: homogeneous[min(int((mm - us[0]) * grid.PX_PER_MM), len(us) - 1), len(vs) // 2]),
            ("v: from the picture's right toward its left", top_down.mean(axis=0), vs[0],
             lambda mm: homogeneous[len(us) // 2, min(int((mm - vs[0]) * grid.PX_PER_MM), len(vs) - 1)])):
        places = line_positions(profile, 1 / grid.PX_PER_MM) + start
        spacing = np.diff(places)
        keep = (spacing > 4.0) & (spacing < 6.2)
        print(f"{name}: {len(places)} lines")
        middles = (places[:-1] + places[1:]) / 2
        for group in np.array_split(np.arange(len(spacing))[keep], 6):
            if len(group):
                at = pixel_at(middles[group].mean())
                print(f"  at {middles[group[0]]:5.1f}-{middles[group[-1]]:5.1f} mm (px y {at[1] / at[2]:5.0f}, "
                      f"x {at[0] / at[2]:5.0f}): {spacing[group].mean():.3f} mm a square over {len(group)}")
        slope = np.polyfit(middles[keep], spacing[keep], 1)
        print(f"  straight-line trend: {slope[1] + slope[0] * middles[keep][0]:.3f} -> "
              f"{slope[1] + slope[0] * middles[keep][-1]:.3f} mm; the squares' scatter about it "
              f"{np.std(spacing[keep] - np.polyval(slope, middles[keep])):.3f} mm")


if __name__ == "__main__":
    main()
