#!/usr/bin/env python3
"""Do the found edges depend on the camera they were searched through? CALIBRATE's frame (config/camera-frame.png,
10 Nov 10:31) and its three cubes: the edges searched again round after round as locate.py searches one cube --
through CALIBRATE's camera, through the six points' camera, through cameras with the focal length 5% off (each
refitted to the cubes' edges with that focal held), and with the search band twice as wide -- and each edge's line
compared with the stored one (camera-refine-points.json, points 1-3), at the edge's two ends, pixels.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/research.py
"""
import json, math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import cubes, edges
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, REACH, CameraFit
from applib.camera.model import camera_position, cube_corners, least_squares, project, visible_edges
from applib.camera.store import RefinePoint
from camera import refine
from pretend import fit_camera
from tops import line

CONFIG = Path("config")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
fit = CameraFit.from_json(json.loads((CONFIG / "camera.json").read_text()))
stored = json.loads((CONFIG / "camera-refine-points.json").read_text())["points"]
points = [RefinePoint(**p) for p in stored]
sightings = [p.sighting for p in points]
alone = fit.cube_fit().true_to_pixel
desk = cubes.desk_chroma(frame)
blobs = cubes.find_blobs(frame)


def blob_for(pixel):
    """The blob holding the cube's base-centre pixel's neighbourhood: the one whose centroid is nearest."""
    return min(blobs, key=lambda b: np.linalg.norm(b.centroid - np.array(pixel)))


def research(matrix, sight, reaches):
    blob = blob_for(sight.pixel)
    crop = edges.crop(frame, blob, desk)
    start = refine.start(matrix, sight)
    params = refine.place(matrix, sight, start[:3])           # the cube as its stored edges put it through `matrix`
    camera = camera_position(matrix)
    found = None
    for reach in reaches:
        seen = visible_edges(params[:2], params[2], camera)
        corners = project(matrix, cube_corners(params[:2], params[2]))
        found = edges.search(crop, corners, seen, reach)
        params = least_squares(lambda t: edges.signed_distances(project(matrix, cube_corners(t[:2], t[2])), *found),
                               params, np.array([0.01, 0.01, 1e-5]), huber=EDGE_HUBER_PX, iterations=30)[0]
    return found, params


def lines_of(found):
    starts, ends, pts = found
    out = {}
    for a, b in set(zip(starts.tolist(), ends.tolist())):
        sel = (starts == a) & (ends == b)
        if sel.sum() >= 5:
            out[tuple(sorted((a, b)))] = (line(pts[sel]), pts[sel])
    return out


def compare(reference, other):
    """Per edge in both: how far the other's line is from the reference line at the reference points' two ends."""
    gaps = {}
    for key, ((m, d, _), pts) in reference.items():
        if key not in other:
            gaps[key] = None
            continue
        (m2, d2, _), _ = other[key]
        normal = np.array((-d2[1], d2[0]))
        along = (pts - m) @ d
        ends = [m + along.min() * d, m + along.max() * d]
        gaps[key] = max(abs((e - m2) @ normal) for e in ends)
    return gaps


def held_focal(scale):
    """The three cubes' camera with the focal length `scale` times CALIBRATE's, the rest refitted to their edges."""
    numbers = refine._numbers(alone)
    numbers[0] *= scale
    on_arm = refine._OnArm(fit.frame_size, sightings[:3], alone)
    free = [1, 2, 3, 4, 5, 6, 7, 8]
    numbers = least_squares(on_arm.residuals, numbers, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=100,
                            free=free)[0]
    return refine._pinhole(numbers).matrix


six = refine._pinhole(fit_camera(fit.frame_size, sightings, alone)[0]).matrix
cameras = {"CALIBRATE's camera": (alone, REACH), "the six points' camera": (six, REACH),
           "focal 5% shorter": (held_focal(0.95), REACH), "focal 5% longer": (held_focal(1.05), REACH),
           "band twice as wide": (alone, tuple(2 * r for r in REACH))}
for k in range(3):
    reference = lines_of(sightings[k].edges)
    print(f"#{points[k].id} {points[k].cube}: stored edges' lines, rms px: "
          + " ".join(f"{a}{b}:{v[0][2]:.2f}" for (a, b), v in sorted(reference.items())))
    for name, (matrix, reaches) in cameras.items():
        found, params = research(matrix, sightings[k], reaches)
        gaps = compare(reference, lines_of(found))
        shown = " ".join(f"{a}{b}:{'--' if g is None else f'{g:.2f}'}" for (a, b), g in sorted(gaps.items()))
        worst = max(g for g in gaps.values() if g is not None)
        print(f"   {name:24} {len(found[2]):3} points; each edge's line from the stored one, px: {shown}  "
              f"(worst {worst:.2f})")
