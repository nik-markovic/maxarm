#!/usr/bin/env python3
"""The arm's base found in CALIBRATE's frame by its own edges, and the cubes predicted from it, against the
calibration's own prediction. Through camera.json's camera (the cubes' edges alone), the base's shape the owner's
(work/PILOT-kiwi-idea.md, as calibration/camera/check.py draws it):

  cyan / magenta / white  the base where camera.json puts it: laid by the arm's three readings (prediction only);
  green                   the base where the frame's edges put it: the same shape moved and turned (x, y, turn) onto
                          the edges round its nuts, cups' rims and plate edges;
  pink                    each cube where the arm read it, the arm's axis and turn taken from the green base;
  yellow                  the cubes as seen (their own edges).

Writes out/base-from-edges-<time>.jpg (the whole frame) and -base.jpg (the base, 2x), and prints the moves.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/base-from-edges.py [frame.png]
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from applib.camera.fit import CameraFit
from applib.camera.model import cube_corners, project, visible_edges, camera_position
from calibration.camera import check

CONFIG, OUT = Path("config"), Path("agenttools/fitcheck/out")
GREEN, PREDICTED = (0, 255, 0), (180, 105, 255)      # BGR: the edge-found base; the cubes predicted (pink)
TRUNCATE_PX = 6.0                       # an edge further than this from a drawn line counts no more: hidden, or grain


def base_points():
    """Sample points of the base, desk mm round the arm's axis: rims, nuts, tips, plate edges."""
    half, out = check.BASE_CUP_SPACING_MM / 2, []
    for cx, cy in ((half, half), (half, -half), (-half, -half), (-half, half)):
        for radius, z in ((check.BASE_CUP_RADIUS_MM, 0.0), (check.NUT_HALF_MM, check.PLATE_TOP_MM),
                          (check.NUT_HALF_MM, check.NUT_TOP_MM), (check.TIP_TOP_HALF_MM, check.TIP_TOP_MM)):
            a = np.linspace(0, 2 * np.pi, 48 if radius > 10 else 16, endpoint=False)
            out += [(cx + radius * math.cos(t), cy + radius * math.sin(t), z) for t in a]
    e, h = check.PLATE_EDGE_MM / 2, check.PLATE_EDGE_HALF_MM
    for t in np.linspace(-h, h, 20):
        out += [(e, t, check.PLATE_TOP_MM), (-e, t, check.PLATE_TOP_MM), (t, e, check.PLATE_TOP_MM),
                (t, -e, check.PLATE_TOP_MM)]
    return np.array(out)


def moved(points, pose):
    dx, dy, turn = pose
    c, s = math.cos(turn), math.sin(turn)
    return np.column_stack((c * points[:, 0] - s * points[:, 1] + dx, s * points[:, 0] + c * points[:, 1] + dy,
                            points[:, 2]))


fit = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text()))
import sys
FRAME = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG / "camera-frame.png"   # or another frame, the camera unmoved
frame = cv2.imread(str(FRAME))
matrix = fit.desk_to_pixel
height, width = frame.shape[:2]
gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
edge_map = cv2.Canny(gray, 30, 90)
distance = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 3)
samples = base_points()


def cost(pose):
    pix = project(matrix, moved(samples, pose))
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < width - 1) & (pix[:, 1] >= 0) & (pix[:, 1] < height - 1)
    d = distance[pix[inside, 1].astype(int), pix[inside, 0].astype(int)]
    return float(np.mean(np.minimum(d, TRUNCATE_PX) ** 2))


best = (cost((0, 0, 0)), (0.0, 0.0, 0.0))
for step, span in ((0.5, 8.0), (0.1, 1.0)):
    centre = best[1]
    for dx in np.arange(-span, span + 1e-9, step):
        for dy in np.arange(-span, span + 1e-9, step):
            for dt in np.radians(np.arange(-3.0, 3.01, 0.5 if step > 0.2 else 0.1)) if step > 0.2 else \
                    np.radians(np.arange(-0.5, 0.51, 0.1)):
                pose = (centre[0] + dx, centre[1] + dy, centre[2] + dt)
                c = cost(pose)
                if c < best[0]:
                    best = (c, pose)
pose = best[1]
print(f"base by its edges: moved ({pose[0]:+.1f}, {pose[1]:+.1f}) mm, turned {math.degrees(pose[2]):+.2f} deg "
      f"from where the arm's readings put it (edge cost {best[0]:.2f}, unmoved {cost((0, 0, 0)):.2f})")

image = check.draw(frame, fit, [])
pieces = check._circle  # same circles as the check image, moved


def draw_base(img, pose_, colour):
    half = check.BASE_CUP_SPACING_MM / 2
    for cx, cy in ((half, half), (half, -half), (-half, -half), (-half, half)):
        for radius, z in ((check.BASE_CUP_RADIUS_MM, 0.0), (check.NUT_HALF_MM, check.PLATE_TOP_MM),
                          (check.NUT_HALF_MM, check.NUT_TOP_MM), (check.TIP_TOP_HALF_MM, check.TIP_TOP_MM)):
            ring = project(matrix, moved(pieces((cx, cy), radius, z), pose_))
            cv2.polylines(img, [np.round(ring * 16).astype(np.int32)], False, colour, 2 if radius > 10 else 1,
                          cv2.LINE_AA, 4)
    e, h = check.PLATE_EDGE_MM / 2, check.PLATE_EDGE_HALF_MM
    for a, b in (((e, -h), (e, h)), ((-e, -h), (-e, h)), ((-h, e), (h, e)), ((-h, -e), (h, -e))):
        line = project(matrix, moved(np.array([(*a, check.PLATE_TOP_MM), (*b, check.PLATE_TOP_MM)]), pose_))
        cv2.polylines(img, [np.round(line * 16).astype(np.int32)], False, colour, 1, cv2.LINE_AA, 4)


draw_base(image, pose, GREEN)
cam_xyz = camera_position(fit.true_to_pixel)
print("cubes, the arm's reading moved with the base onto the frame, against the cube as seen (desk mm):")
for colour, cube in fit.cubes.items():
    reading = moved(np.array([[cube["x"], cube["y"], 0.0]]), pose)[0]
    true_xy = fit.desk_to_true @ reading[:2]
    turn = math.radians(cube["turn_deg"])
    corners = cube_corners(true_xy, turn)
    pix = project(fit.true_to_pixel, corners)
    for edge in visible_edges(true_xy, turn, cam_xyz):
        cv2.line(image, tuple(np.round(pix[edge.start] * 16).astype(int)), tuple(np.round(pix[edge.end] * 16).astype(int)),
                 PREDICTED, 1, cv2.LINE_AA, 4)
    seen = np.linalg.solve(fit.desk_to_true, fit.seen_centre(colour))
    print(f"  {colour:5} reading moved to ({reading[0]:7.1f}, {reading[1]:7.1f}); seen at ({seen[0]:7.1f}, {seen[1]:7.1f});"
          f" {math.dist(reading[:2], seen):.1f} mm apart")
stamp = time.strftime("%H%M")
cv2.imwrite(str(OUT / f"base-from-edges-{stamp}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
pts = project(matrix, moved(samples, (0, 0, 0)))
lo = np.maximum(pts.min(0) - 60, 0).astype(int)
hi = np.minimum(pts.max(0) + 60, (width, height)).astype(int)
crop = cv2.resize(image[lo[1]:hi[1], lo[0]:hi[0]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(OUT / f"base-from-edges-{stamp}-base.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"out/base-from-edges-{stamp}.jpg and -base.jpg")
