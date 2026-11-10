#!/usr/bin/env python3
"""The arm down on the red cube, nothing touched: in one frame the cup's pale skirt against the red top, and the
cube placed by the edges the arm does not hide -- each through CALIBRATE's camera (three cubes) and SAVE's (four
points) -- against the servos' reading. Cup and cube are at one place and nearly one height, so how far apart they
are is the frame's, whichever camera. No arm moved (frame from the app's live view).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/cup-now.py config-dir frame.png x y z out.png
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from applib.camera import cubes, edges, store
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, REACH
from applib.camera.model import CUBE_MM, camera_position, cube_corners, least_squares, project, to_polygon, \
    visible_edges

CONFIG = Path(sys.argv[1])
RIM_MM = 10.5
frame = cv2.imread(sys.argv[2])
reading = np.array([float(sys.argv[3]), float(sys.argv[4])])
z_read = float(sys.argv[5])
saved = store.load_camera(CONFIG)
cameras = {"CALIBRATE": saved.cube_fit().true_to_pixel, "SAVE": saved.true_to_pixel}
true_reading = HANDEDNESS @ reading

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
h, s, v = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
arm = ((h > 30) & (h < 95) & (s > 0.25)) | (v < 0.22)
arm = cv2.dilate(arm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
red = ((h < 14) | (h > 170)) & (s > 0.55) & (v > 0.35)
blob = max((b for b in cubes.find_blobs(frame, is_warm_only=True) if cubes.is_warm(b)), key=lambda b: b.area)
crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
cup_px = project(cameras["CALIBRATE"], np.array([[*true_reading, z_read]]))[0]


def corners(matrix, p):
    return project(matrix, cube_corners(p[:2], p[2]))


def overlap_unhidden(matrix, p):
    drawn = np.zeros(red.shape, np.uint8)
    cv2.fillConvexPoly(drawn, cv2.convexHull(np.round(corners(matrix, p)).astype(np.int32)), 1)
    drawn = (drawn > 0) & ~arm
    mine = red & ~arm
    return np.count_nonzero(drawn & mine) / max(1, np.count_nonzero(drawn | mine))


def place_cube(matrix):
    start = max(([*true_reading, math.radians(t)] for t in np.arange(0, 90, 1.0)),
                key=lambda p: overlap_unhidden(matrix, p))
    p = np.array(start)
    for reach in REACH:
        seen = visible_edges(p[:2], p[2], camera_position(matrix))
        starts, ends, found = edges.search(crop, corners(matrix, p), seen, reach)
        clear = ~arm[np.clip(found[:, 1].round().astype(int), 0, arm.shape[0] - 1),
                     np.clip(found[:, 0].round().astype(int), 0, arm.shape[1] - 1)]
        starts, ends, found = starts[clear], ends[clear], found[clear]
        p, distances, _ = least_squares(lambda t: edges.signed_distances(corners(matrix, t), starts, ends, found),
                                        p, np.array([0.01, 0.01, 1e-5]), huber=EDGE_HUBER_PX, iterations=30)
    return p, (starts, ends, found), float(np.median(np.abs(distances)))


def rim(matrix, centre, z, radius=RIM_MM, count=180):
    a = np.linspace(0, 2 * math.pi, count, endpoint=False)
    return project(matrix, np.column_stack((centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a),
                                            np.full(count, z))))


# the cup: its skirt's outer edge, where the saturation steps from the translucent skirt (~0.4) to the red top
# (~1), along rays from inside the visible skirt; only rays that reach red past it without crossing the arm
sat = cv2.GaussianBlur(s, (0, 0), 1.0)
pale = (s < 0.5) & (v > 0.5) & ~arm
window = 150
x0, y0 = int(cup_px[0] - window), int(cup_px[1] - window)
ys, xs = np.nonzero(pale[y0:y0 + 2 * window, x0:x0 + 2 * window])
middle = np.array([xs.mean() + x0, ys.mean() + y0])
rim_list = []
for angle in np.radians(np.arange(0, 360, 1.0)):
    direction = np.array([math.cos(angle), math.sin(angle)])
    radii = np.arange(5, 160, 0.25)
    path = middle + radii[:, None] * direction
    xi, yi = path[:, 0].round().astype(int), path[:, 1].round().astype(int)
    values = cv2.remap(sat, path[:, 0].astype(np.float32).reshape(1, -1), path[:, 1].astype(np.float32).reshape(1, -1),
                       cv2.INTER_LINEAR).ravel()
    over = np.nonzero(values > 0.7)[0]
    if not len(over) or over[0] == 0:
        continue
    k = over[0]
    if arm[yi[:k + 24], xi[:k + 24]].any() or not red[yi[min(k + 24, len(xi) - 1)], xi[min(k + 24, len(xi) - 1)]]:
        continue
    r = radii[k - 1] + (0.7 - values[k - 1]) / (values[k] - values[k - 1]) * (radii[k] - radii[k - 1])
    rim_list.append(middle + r * direction)
rim_points = np.array(rim_list)
print(f"cup projected at ({cup_px[0]:.0f}, {cup_px[1]:.0f}); rim: {len(rim_points)} rays of 360 from "
      f"({middle[0]:.0f}, {middle[1]:.0f}) reach red past the skirt")
shown = frame.copy()
for name, matrix in cameras.items():
    p, found, edge_px = place_cube(matrix)
    cube_desk = HANDEDNESS @ p[:2]
    print(f"{name}: cube by its unhidden edges at desk ({cube_desk[0]:.2f}, {cube_desk[1]:.2f}), {len(found[2])} edge "
          f"points, {edge_px:.2f} px typ; minus the reading {np.round(cube_desk - reading, 2).tolist()} "
          f"({np.linalg.norm(cube_desk - reading):.2f} mm)")
    for free in (False, True):
        def residuals(t):
            return to_polygon(rim_points, rim(matrix, t[:2], z_read, t[2] if free else RIM_MM))
        start = np.array([*true_reading, RIM_MM])
        t = least_squares(residuals, start if free else start[:2], np.array([0.01, 0.01, 0.01][:3 if free else 2]),
                          huber=1.0, iterations=60)[0]
        cup_desk = HANDEDNESS @ t[:2]
        print(f"   cup rim at z {z_read:.2f}{', radius ' + format(t[2], '.2f') if free else ', radius 10.5'}: desk "
              f"({cup_desk[0]:.2f}, {cup_desk[1]:.2f}); minus the reading {np.round(cup_desk - reading, 2).tolist()} "
              f"({np.linalg.norm(cup_desk - reading):.2f}); cube minus cup {np.round(cube_desk - cup_desk, 2).tolist()} "
              f"({np.linalg.norm(cube_desk - cup_desk):.2f} mm); rim fit {np.median(np.abs(residuals(t))):.2f} px")
        if name == "CALIBRATE" and not free:
            cup_fit = t[:2]
    if name == "CALIBRATE":
        drawn = {"reading": ((0, 255, 0), [*true_reading, p[2]]), "cube": ((0, 215, 255), p)}
        for colour, cube in drawn.values():
            c = corners(matrix, cube)
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                cv2.line(shown, tuple(np.round(c[a] * 4).astype(int)), tuple(np.round(c[b] * 4).astype(int)), colour,
                         1, cv2.LINE_AA, shift=2)
        cv2.polylines(shown, [np.round(rim(matrix, true_reading, z_read) * 4).astype(np.int32)], True, (0, 255, 0), 1,
                      cv2.LINE_AA, shift=2)
        cv2.polylines(shown, [np.round(rim(matrix, cup_fit, z_read) * 4).astype(np.int32)], True, (255, 255, 255), 1,
                      cv2.LINE_AA, shift=2)
        for point, colour in ((cup_fit, (255, 255, 255)), (true_reading, (0, 255, 0)), (p[:2], (0, 215, 255))):
            z = z_read if colour != (0, 215, 255) else CUBE_MM
            cv2.drawMarker(shown, tuple(np.round(project(matrix, np.array([[*point, z]]))[0]).astype(int)), colour,
                           cv2.MARKER_CROSS, 14, 1)
for q in rim_points:
    cv2.circle(shown, tuple(np.round(q).astype(int)), 1, (255, 0, 255), -1)
cx, cy = int(cup_px[0]), int(cup_px[1])
view = shown[max(0, cy - 260):cy + 260, max(0, cx - 330):cx + 330]
out = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
legend = ["Through CALIBRATE's camera (the 3 cubes). The arm down on the red cube (point 4), untouched since RECORD;",
          f"its servos read desk ({reading[0]:.2f}, {reading[1]:.2f}, {z_read:.2f}).",
          "green: a 40 mm cube and the cup's 21 mm rim where the servos say the cup is; + its centre",
          "magenta: the cup's skirt edge found against the red top (the bracket hides the rest)",
          "white: the 21 mm rim fitted to those points; + its centre",
          "yellow: the cube placed by its own unhidden edges; + its top's centre"]
band = np.full((30 * len(legend) + 16, out.shape[1], 3), 32, np.uint8)
for i, text in enumerate(legend):
    cv2.putText(band, text, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
cv2.imwrite(sys.argv[6], np.vstack((out, band)))
