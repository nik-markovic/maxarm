"""The cube finders on every real frame kept: CALIBRATE's fit on each calibration frame with its own points, and
FIND RED CUBE on frames of one red cube, through the camera of that rig. Before and after a change to the finders,
the lines should only get better. No arm, no camera.

    ../.venv/bin/python agenttools/glare/check-frames.py
"""
import json
import sys
import time
from pathlib import Path

import cv2

IMPALA = Path(__file__).resolve().parents[2]
WORK = IMPALA.parent
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]
from camera import cubes, store  # noqa: E402
from camera.fit import fit_camera  # noqa: E402
from camera.locate import find_red_cube  # noqa: E402

CALIBRATIONS = [IMPALA / "config", WORK / "work/config-impala-old", WORK / "work/config-old", WORK / "gecko/config",
                WORK / "hyena/config"]
RED_CUBES = [  # (frame, the rig's config)
    (Path("/tmp/c920-still.png"), IMPALA / "config"),
    (Path("/tmp/blue-still.png"), IMPALA / "config"),
    (Path("/tmp/ft-still.png"), IMPALA / "config"),
    (IMPALA / "agenttools/arc/paper-now.png", WORK / "work/config-impala-old"),
    (IMPALA / "agenttools/arc/paper2-now.png", WORK / "work/config-impala-old"),
    (IMPALA / "agenttools/live-res/yuyv.png", WORK / "work/config-old"),
    (IMPALA / "agenttools/live-res/mjpg.png", WORK / "work/config-old"),
]

for config in CALIBRATIONS:
    frame = cv2.imread(str(config / "camera-frame.png"))
    points = {c: (p.x, p.y) for c, p in store.load_points(config).items() if p}
    started = time.monotonic()
    try:
        fit = fit_camera(frame, points)
        offs = ", ".join(f"{c} {fit.cube_off_mm(c):.2f}" for c in fit.cubes)
        result = f"edges {fit.edge_px[0]:.2f}/{fit.edge_px[1]:.2f} px; cubes from the arm {offs} mm"
    except Exception as error:
        result = f"REFUSED {type(error).__name__}: {error}"
    print(f"CALIBRATE {config.relative_to(WORK)}: {cubes.describe(cubes.find_blobs(frame))}; {result} "
          f"({time.monotonic() - started:.1f} s)")
for path, config in RED_CUBES:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"FIND RED CUBE {path}: no frame")
        continue
    fit = store.load_camera(config)
    try:
        found = find_red_cube(frame, fit)
        result = f"desk ({found.x:.2f}, {found.y:.2f}), covers {found.overlap:.2f}, edges {found.edge_px:.2f} px"
    except Exception as error:
        result = f"REFUSED {type(error).__name__}: {error}"
    print(f"FIND RED CUBE {path.name} ({config.relative_to(WORK)}): {result}")
for path in (Path("/tmp/blue-still.png"),):
    frame = cv2.imread(str(path))
    points = {c: (p.x, p.y) for c, p in store.load_points(IMPALA / "config").items() if p}
    try:
        fit = fit_camera(frame, points)
        offs = ", ".join(f"{c} {fit.cube_off_mm(c):.2f}" for c in fit.cubes)
        result = f"edges {fit.edge_px[0]:.2f}/{fit.edge_px[1]:.2f} px; cubes from the arm {offs} mm"
    except Exception as error:
        result = f"REFUSED {type(error).__name__}: {error}"
    print(f"CALIBRATE {path.name} with config/camera-points.json: {cubes.describe(cubes.find_blobs(frame))}; {result}")
