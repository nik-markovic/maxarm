"""The owner's reading of the red cube's close-up (2026-10-28): the wireframe's perspective is off -- the far
right edge leans out at its top, the top's far corner sits off its rounded corner to one side. A camera with a
longer focal length stands farther off and draws a cube's top smaller against its bottom. Here the cubes' camera is
refitted on camera-frame.png with the focal length held at several values; through each, the red cube of a frame
of today's (the camera not moved) is found, and each edge's distance from the frame's own edge given at both its
ends. Also how well the three cubes fit, and how far they then stand from the arm's points. No arm.

    ../.venv/bin/python agenttools/arc/focal-outline.py /tmp/formats/yuyv-avg5.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import cubes, edges  # noqa: E402
from camera import fit as fitmod  # noqa: E402
from camera.fit import CENTRE, CENTRE_SPREAD, EDGE_HUBER_PX, FOCAL, REACH, STEPS  # noqa: E402
from camera.locate import find_red_cube  # noqa: E402
from camera.model import least_squares  # noqa: E402

calibration = cv2.imread(str(ROOT / "config/camera-frame.png"))
today = cv2.imread(sys.argv[1])
points = {c: (v["x"], v["y"]) for c, v in json.loads((ROOT / "config/camera-points.json").read_text())["cubes"].items()}

blobs = cubes.find_blobs(calibration)
starts = [(*fitmod._start(calibration, naming, points), naming) for naming in cubes.namings(blobs)]
_, params0, naming = max(starts, key=lambda s: s[0])
params0 = fitmod._fit_outlines(naming, points, params0)
params_free, _ = fitmod._fit_edges(calibration, naming, points, params0)


def fit_edges_held(params, free):
    desk = cubes.desk_chroma(calibration)
    crops = [edges.crop(calibration, naming[c], desk) for c in fitmod.COLOURS]
    centre = np.array([calibration.shape[1] / 2, calibration.shape[0] / 2])
    for reach in REACH:
        found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
                 zip(crops, fitmod._corners(params, points), fitmod._edges_seen(params, points))]

        def residuals(trial):
            parts = [edges.signed_distances(corners, *f) for corners, f in zip(fitmod._corners(trial, points), found)]
            parts.append((trial[CENTRE] - centre) / (CENTRE_SPREAD * calibration.shape[1]))
            return np.concatenate(parts)

        params, distances, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, free=free,
                                             iterations=30)
    return params, distances[:-2]


def edge_ends(fit, frame):
    """The red cube found through `fit`; each edge it shows: the frame's edge outward of the wireframe's (px) at the
    edge's start and at its end, a straight line fitted along it."""
    found = find_red_cube(frame, fit)
    mapped = np.column_stack((found.corners, np.ones(8))) @ fit.desk_to_pixel.T
    corners = mapped[:, :2] / mapped[:, 2:]
    blob = max((b for b in cubes.find_blobs(frame, is_warm_only=True) if cubes.is_warm(b)), key=lambda b: b.size)
    crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
    starts, ends, targets = edges.search(crop, corners, found.seen, 0.02)
    middle = corners.mean(axis=0)
    out = {}
    for edge in found.seen:
        mine = (starts == edge.start) & (ends == edge.end)
        if mine.sum() < 4:
            continue
        a, b = corners[edge.start], corners[edge.end]
        along = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-along[1], along[0]])
        if edge.is_outline:
            normal = normal if normal @ ((a + b) / 2 - middle) > 0 else -normal
        else:
            normal = normal if normal[1] > 0 else -normal
        share = (targets[mine] - a) @ along / np.linalg.norm(b - a)
        offset = (targets[mine] - a) @ normal
        slope, start = np.polyfit(share, offset, 1)
        out[f"{edge.start}-{edge.end}{'o' if edge.is_outline else 'i'}"] = (start, start + slope)
    return found, out


print(f"focal free on the calibration frame: {params_free[FOCAL]:.0f} px")
for focal in (1916, 1960, 2000, 2050, 2100, 2200, 2300):
    held = params_free.copy()
    held[FOCAL] = focal
    held[fitmod.SHIFT] = params_free[fitmod.SHIFT] * focal / params_free[FOCAL]     # the cubes' scale kept
    free = [k for k in range(len(held)) if k != FOCAL]
    held, distances = fit_edges_held(held, free)
    fitted = fitmod._result(held, points, calibration, distances)
    offs = [fitted.cube_off_mm(colour) for colour in fitted.cubes]
    height = fitmod._pinhole(held).position[2]
    found, ends = edge_ends(fitted, today)
    text = "  ".join(f"{name} {s:+.1f}>{e:+.1f}" for name, (s, e) in ends.items())
    print(f"focal {focal}: camera {height:.0f} mm up; cubes' edges {np.median(np.abs(distances)):.2f} px "
          f"(90% {np.percentile(np.abs(distances), 90):.2f}); cubes from the arm's points "
          f"{np.sqrt(np.mean(np.square(offs))):.2f} mm rms | red cube ({found.x:.2f}, {found.y:.2f}), its edges "
          f"{found.edge_px:.2f} px")
    print(f"      edges, start>end px (outline: + outward; inner: + towards the camera): {text}")
