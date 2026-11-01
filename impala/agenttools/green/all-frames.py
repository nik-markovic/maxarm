#!/usr/bin/env python3
"""The cube finder and CALIBRATE on every frame kept, to compare a change of the finder before and after.

    .venv/bin/python impala/agenttools/green/all-frames.py [sheet.jpg]

CALIBRATE on each frame that has the arm's points: the calibration frames of every rig, today's frames of the
owner's desk with today's points, and the drawn scene. The cube finder alone on the old camera shots
(agenttools/camtest/shots, three cameras, exposures, white balances), with the cubes' places marked by hand
(`MARKS`): which blobs it finds, and whether each cube is one of them. With a path, a contact sheet of every
frame with its blobs outlined.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMPALA = Path(__file__).resolve().parents[2]
WORK = IMPALA.parent
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration"), str(IMPALA / "agenttools")]

import cube_scene                                    # noqa: E402
from camera import cubes, store                      # noqa: E402
from camera.fit import fit_camera                    # noqa: E402

SHOTS = WORK / "agenttools/camtest/shots"
GREEN = IMPALA / "agenttools/green"


def arm_points(path: Path) -> dict:
    stored = json.loads(path.read_text())["cubes"]
    return {colour: (point["x"], point["y"]) for colour, point in stored.items() if point}


CALIBRATIONS = [  # (name, frame, the arm's points)
    ("C920 FPS 28 Oct", IMPALA / "config/camera-frame-previous.png", IMPALA / "config/camera-points-previous.json"),
    ("doc impala-old", WORK / "work/config-impala-old/camera-frame.png",
     WORK / "work/config-impala-old/camera-points.json"),
    ("doc gecko", WORK / "gecko/config/camera-frame.png", WORK / "gecko/config/camera-points.json"),
    ("C920 QUALITY today, live view", GREEN / "still-1.png", IMPALA / "config/camera-points.json"),
    ("C920 QUALITY today, camera", GREEN / "now-quality.png", IMPALA / "config/camera-points.json"),
    ("C920 FPS today, camera", GREEN / "now-fps.png", IMPALA / "config/camera-points.json"),
]

# The cubes' middles in the old shots, by eye, as shares of the frame's width and height: red, green, blue.
SCENE_A = {"red": (0.225, 0.35), "green": (0.51, 0.57), "blue": (0.53, 0.23)}     # the C920's view
SCENE_B = {"red": (0.33, 0.26), "green": (0.83, 0.63), "blue": (0.77, 0.12)}      # the Brio's
SCENE_C = {"red": (0.17, 0.31), "green": (0.65, 0.63), "blue": (0.60, 0.15)}      # the document camera's
MARKS = {**{name: SCENE_A for name in ("cam_c920", "C920_FINAL", "c920_e150", "c920_e250", "c920_e350",
                                       "c920_e500", "c920_e700", "c920_e900")},
         **{name: SCENE_B for name in ("cam_brio", "BRIO_FINAL", "brio_e60", "brio_e100", "brio_e140", "brio_e200",
                                       "brio_e400", "brio_e700", "brio_e1000", "brio_e1400")},
         **{name: SCENE_C for name in ("cam_doc", "DOC_FINAL", "DOC_CORRECTED", "FINAL", "FINAL_ccm", "baseline",
                                       "best_s30", "best_s40", "best_s40_g150", "best_s50", "contrast50",
                                       "light_on", "nobacklight", "sat30", "sat30_nobl", "wb2800", "wb3800",
                                       "wb4600", "wb5500", "wb6500", "n_2592x1944", "res_2592x1944")}}


def calibrate(name, frame, points, tiles):
    started = time.monotonic()
    blobs = cubes.find_blobs(frame)
    try:
        fit = fit_camera(frame, points)
        offs = ", ".join(f"{colour} {fit.cube_off_mm(colour):.2f}" for colour in fit.cubes)
        result = f"edges {fit.edge_px[0]:.2f} px; cubes from the arm {offs} mm"
    except Exception as error:
        result = f"REFUSED: {error}"
    print(f"CALIBRATE {name}: {cubes.describe(blobs)}; {result} ({time.monotonic() - started:.1f} s)")
    tiles.append(drawn(frame, blobs, name))


def shot(name, frame, marks, tiles):
    blobs = cubes.find_blobs(frame)
    height, width = frame.shape[:2]
    found = []
    def is_on(blob, u, v):
        return (cv2.pointPolygonTest(blob.contour, (u * width, v * height), False) >= 0
                or np.hypot(blob.centroid[0] - u * width, blob.centroid[1] - v * height) < 0.04 * width)

    for colour, (u, v) in marks.items():
        hit = [blob for blob in blobs if is_on(blob, u, v)]
        kind = ("warm" if cubes.is_warm(hit[0]) else "cool") if hit else "-"
        found.append(f"{colour} {kind}")
    others = len(blobs) - sum(1 for blob in blobs if any(is_on(blob, u, v) for u, v in marks.values()))
    has_naming = any(True for _ in cubes.namings(blobs))
    print(f"shot {name}: {', '.join(found)}; {others} other blob{'s' if others != 1 else ''}"
          f"{'' if has_naming else '; NO NAMING'}")
    tiles.append(drawn(frame, blobs, name))


def drawn(frame, blobs, name):
    image = frame.copy()
    line = max(2, round(frame.shape[1] / 600))
    for blob in blobs:
        cv2.drawContours(image, [blob.contour], -1, (255, 0, 255), line)
    image = cv2.resize(image, (400, int(400 * frame.shape[0] / frame.shape[1])))
    cv2.putText(image, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def main():
    tiles = []
    for name, frame_path, points_path in CALIBRATIONS:
        calibrate(name, cv2.imread(str(frame_path)), arm_points(points_path), tiles)
    scene = cube_scene.render()
    calibrate("drawn scene", scene.frame, scene.points, tiles)
    for name, marks in MARKS.items():
        frame = cv2.imread(str(SHOTS / f"{name}.jpg"))
        if frame is not None:
            shot(name, frame, marks, tiles)
    if len(sys.argv) > 1:
        width = 6
        rows = []
        for start in range(0, len(tiles), width):
            row = tiles[start:start + width]
            height = max(tile.shape[0] for tile in row)
            row = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in row]
            row += [np.zeros_like(row[0])] * (width - len(row))
            rows.append(np.hstack(row))
        cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
