#!/usr/bin/env python3
"""A camera fitted to the sheet's grid alone (5 mm squares, flat on the desk), against camera.json's, fitted to the
three cubes; then the three cubes of CALIBRATE's frame (their edges, camera-refine-points.json) through each.

    .venv/bin/python impala/agenttools/lens/grid-camera.py FRAME.png [square-mm]

The grid's lines are found in the sheet seen from above through camera.json, in strips 10 mm wide, and taken back to
the frame's pixels. The grid camera: focal length, principal point (held loosely), rotation and shift, so that every
line found lies on its line of a perfect 5 mm grid, the grid's place and turn free. Prints both cameras' focal
length, height above the desk and tilt; the grid's misses; the cubes' heights, edges and the arm's triangle.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from camera import store                                                   # noqa: E402
from camera.fit import CENTRE_SPREAD, HANDEDNESS                           # noqa: E402
from camera.model import least_squares                                     # noqa: E402
from camera.refine import _numbers, _pinhole                               # noqa: E402
from camera.sighting import in_frame, measure, misses, start               # noqa: E402

spec = importlib.util.spec_from_file_location("lines", HERE / "grid-lines.py")
lines_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lines_module)
grid = lines_module.grid
STRIP_MM = 10.0


def grid_points(gray, desk_to_pixel, square):
    """Points on the grid's lines, in frame pixels, each with its family (0: constant u, 1: constant v) and its
    line's number."""
    plane = desk_to_pixel[:, [0, 1, 3]]
    corners = grid.straight.paper_corners(gray)
    borders = [grid.size.line_through(grid.straight.border_points(gray, corners[i], corners[(i + 1) % 4]))
               for i in range(4)]
    pixels = [grid.size.crossing(borders[i - 1], borders[i]) for i in range(4)]
    desk = [np.linalg.solve(plane, (*pixel, 1.0)) for pixel in pixels]
    desk = [point[:2] / point[2] for point in desk]
    origin = desk[1]
    u = (desk[0] - origin) / np.linalg.norm(desk[0] - origin)
    v = (desk[2] - origin) / np.linalg.norm(desk[2] - origin)
    us = np.arange(3, np.linalg.norm(desk[0] - origin) - 3, 1 / grid.PX_PER_MM)
    vs = np.arange(3, np.linalg.norm(desk[2] - origin) - 3, 1 / grid.PX_PER_MM)
    mesh_u, mesh_v = np.meshgrid(us, vs, indexing="ij")
    desk_points = origin + mesh_u[..., None] * u + mesh_v[..., None] * v
    homogeneous = np.concatenate((desk_points, np.ones(desk_points.shape[:2] + (1,))), axis=2) @ plane.T
    top_down = cv2.remap(gray.astype(np.float32), (homogeneous[..., 0] / homogeneous[..., 2]).astype(np.float32),
                         (homogeneous[..., 1] / homogeneous[..., 2]).astype(np.float32), cv2.INTER_LINEAR)
    found = []                                          # (family, line number, u mm, v mm)
    strip = int(STRIP_MM * grid.PX_PER_MM)
    for family, (positions, across, image) in enumerate(((us, vs, top_down), (vs, us, top_down.T))):
        whole = lines_module.line_positions(image.mean(axis=1), 1 / grid.PX_PER_MM) + positions[0]
        reference = whole[len(whole) // 2]
        for first in range(0, len(across) - strip + 1, strip):
            middle = across[first:first + strip].mean()
            for place in lines_module.line_positions(image[:, first:first + strip].mean(axis=1),
                                                     1 / grid.PX_PER_MM) + positions[0]:
                number = round((place - reference) / square)
                found.append((family, number, place, middle) if family == 0 else (family, number, middle, place))
    found = np.array(found)
    desk_found = origin + found[:, 2:3] * u + found[:, 3:4] * v
    pixel = np.column_stack((desk_found, np.ones(len(found)))) @ plane.T
    return found[:, :2].astype(int), pixel[:, :2] / pixel[:, 2:], (origin, u, v)


def on_plane(true_to_pixel, pixels):
    """Pixels to desk x, y (height 0) through a camera in true mm."""
    plane = true_to_pixel[:, [0, 1, 3]]
    true = np.linalg.solve(plane, np.column_stack((pixels, np.ones(len(pixels)))).T).T
    return (true[:, :2] / true[:, 2:]) @ HANDEDNESS.T


def describe(name, numbers):
    camera = _pinhole(numbers)
    axis = camera.matrix[2, :3] / np.linalg.norm(camera.matrix[2, :3])
    tilt = math.degrees(math.acos(abs(axis[2])))
    print(f"  {name}: focal {numbers[0]:.1f} px, principal point {numbers[1]:.0f},{numbers[2]:.0f}, "
          f"{camera.position[2]:.1f} mm above the desk, its axis {tilt:.2f} deg from straight down")


def main():
    frame = cv2.imread(sys.argv[1])
    square = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stored = json.loads((IMPALA / "config" / "camera.json").read_text())
    cube_camera = np.array(stored["true_to_pixel"])
    width, height = stored["frame_size"]
    middle = np.array([width / 2, height / 2])
    labels, pixels, (origin, u, v) = grid_points(gray, np.array(stored["desk_to_pixel"]), square)
    print(f"{len(pixels)} grid points on {len(set(map(tuple, labels)))} lines")
    first = _numbers(cube_camera)

    def grid_misses(params):
        numbers, (shift_u, shift_v, turn) = params[:9], params[9:]
        desk = on_plane(_pinhole(numbers).matrix, pixels) - origin
        cos, sin = math.cos(turn), math.sin(turn)
        along = np.column_stack((desk @ u, desk @ v)) @ np.array([[cos, -sin], [sin, cos]])
        mine = np.where(labels[:, 0] == 0, along[:, 0] - shift_u, along[:, 1] - shift_v)
        return np.concatenate((mine - labels[:, 1] * square, (numbers[1:3] - middle) / (CENTRE_SPREAD * width)))

    reference = grid_misses(np.concatenate((first, [0.0, 0.0, 0.0])))[:-2]
    shift_u = np.median(reference[labels[:, 0] == 0])
    shift_v = np.median(reference[labels[:, 0] == 1])
    start_params = np.concatenate((first, [shift_u, shift_v, 0.0]))
    print(f"through camera.json: the grid's points from a perfect {square} mm grid (its place fitted): "
          f"rms {math.sqrt(np.mean(grid_misses(fit_place(grid_misses, start_params))[:-2] ** 2)):.3f} mm")
    steps = np.concatenate((np.array([0.01, 0.01, 0.01, 1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01]), [0.01, 0.01, 1e-5]))
    solution, residuals, _ = least_squares(grid_misses, start_params, steps, huber=0.3, iterations=200)
    left = residuals[:-2]
    print(f"grid camera: the grid's points from a perfect {square} mm grid: rms {math.sqrt(np.mean(left ** 2)):.3f} mm,"
          f" 95% within {np.percentile(np.abs(left), 95):.3f}")
    describe("cubes' camera (camera.json)", first)
    describe("grid camera                ", solution[:9])
    grid_camera = _pinhole(solution[:9]).matrix
    points = [store.RefinePoint(**point) for point in
              json.loads((IMPALA / "config" / "camera-refine-points.json").read_text())["points"]]
    arm = {point.cube: np.array([point.x, point.y]) for point in points}
    for name, camera in (("cubes' camera", cube_camera), ("grid camera  ", grid_camera)):
        seen, cells = {}, []
        for point in points:
            sighting = point.sighting
            matrix = in_frame(camera, (width, height), sighting)
            cube = measure(matrix, sighting, start(matrix, sighting))
            edges = misses(matrix, sighting, cube)
            seen[point.cube] = HANDEDNESS @ cube[:2]
            cells.append(f"{point.cube} {cube[3] - 40:+.2f} mm tall, edges {math.sqrt(np.mean(edges ** 2)):.2f} px")
        sides = []
        for a, b in (("blue", "green"), ("blue", "red"), ("green", "red")):
            arm_side, camera_side = np.linalg.norm(arm[a] - arm[b]), np.linalg.norm(seen[a] - seen[b])
            sides.append(f"{a}-{b} {camera_side - arm_side:+.2f} mm ({100 * (camera_side / arm_side - 1):+.2f}%)")
        print(f"  {name}: " + "; ".join(cells))
        print("      the cubes' triangle against the arm's: " + ", ".join(sides))


def fit_place(grid_misses, params):
    """The grid's place and turn alone fitted, the camera held."""
    return least_squares(grid_misses, params, np.concatenate(([1.0] * 9, [0.01, 0.01, 1e-5])), huber=0.3,
                         free=[9, 10, 11], iterations=50)[0]


if __name__ == "__main__":
    main()
