#!/usr/bin/env python3
"""Each cube face's colour in every calibration frame kept, against the numbers the cube finder decides by.

    .venv/bin/python impala/agenttools/green/faces.py

For each frame: the desk's chroma and how far from grey it is, the Otsu cut, the desk's spread (median distance
from the desk's chroma over the frame), and for each visible face of each cube -- drawn through the frame's own
calibration -- its median distance from the desk's chroma (d) and from grey (own), and its lightness.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMPALA = Path(__file__).resolve().parents[2]
WORK = IMPALA.parent
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration"), str(IMPALA / "agenttools")]

from camera import cubes                                  # noqa: E402
from camera.fit import CameraFit                          # noqa: E402
from camera.model import _FACES, camera_position, cube_corners, project   # noqa: E402

FRAMES = [  # (name, frame, camera.json)
    ("C920 FPS 28 Oct", IMPALA / "config/camera-frame-previous.png", IMPALA / "config/camera-previous.json"),
    ("doc impala-old", WORK / "work/config-impala-old/camera-frame.png", WORK / "work/config-impala-old/camera.json"),
    ("doc config-old", WORK / "work/config-old/camera-frame.png", WORK / "work/config-old/camera.json"),
    ("doc gecko", WORK / "gecko/config/camera-frame.png", WORK / "gecko/config/camera.json"),
    ("doc hyena", WORK / "hyena/config/camera-frame.png", WORK / "hyena/config/camera.json"),
]


def numbers(frame):
    desk = cubes.desk_chroma(frame)
    chroma = cubes.chroma_from_desk(frame, desk)
    d = np.hypot(chroma[..., 0], chroma[..., 1])
    own = np.hypot(chroma[..., 0] + desk[0], chroma[..., 1] + desk[1])
    cut = cv2.threshold(np.clip(d * 4.0, 0, 255).astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 4.0
    lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    return desk, d, own, cut, lightness


def faces_of(fit, colour, frame_shape):
    """Each visible face of the cube as a pixel mask, shrunk a little from its edges."""
    cube = fit.cubes[colour]
    centre = fit.seen_centre(colour)
    turn = np.radians(cube["turn_deg"])
    corners = cube_corners(centre, turn)
    pixels = project(fit.true_to_pixel, corners)
    eye = camera_position(fit.true_to_pixel)
    c, s = np.cos(turn), np.sin(turn)
    masks = []
    for indices, (nx, ny, nz) in _FACES:
        outward = np.array((c * nx - s * ny, s * nx + c * ny, nz))
        if (eye - corners[list(indices)].mean(axis=0)) @ outward <= 0:
            continue
        polygon = pixels[list(indices)]
        middle = polygon.mean(axis=0)
        polygon = middle + 0.7 * (polygon - middle)
        mask = np.zeros(frame_shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
        masks.append(("top" if nz else "side", mask > 0))
    return masks


def main():
    for name, frame_path, camera_path in FRAMES:
        frame = cv2.imread(str(frame_path))
        fit = CameraFit.from_json(json.loads(camera_path.read_text())).cube_fit()
        desk, d, own, cut, lightness = numbers(frame)
        print(f"{name}: desk {np.hypot(*desk):.1f} from grey, cut {cut:.1f}, desk spread (median d) "
              f"{np.median(d):.1f}, 90% {np.percentile(d, 90):.1f}")
        for colour in fit.cubes:
            parts = [f"{kind} d {np.median(d[mask]):.1f} own {np.median(own[mask]):.1f} L {np.median(lightness[mask]):.0f}"
                     for kind, mask in faces_of(fit, colour, frame.shape)]
            print(f"   {colour:5s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
