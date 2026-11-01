#!/usr/bin/env python3
"""FIND RED CUBE on the frames of one red cube kept, through the camera of their rig.

    .venv/bin/python impala/agenttools/green/red-frames.py [calibration-dir]

With a calibration directory (a copy of calibration/ with another cubes.py, say), that code is used instead.
"""

import sys
from pathlib import Path

import cv2

IMPALA = Path(__file__).resolve().parents[2]
WORK = IMPALA.parent
CALIBRATION = Path(sys.argv[1]) if len(sys.argv) > 1 else IMPALA / "calibration"
sys.path[:0] = [str(IMPALA), str(CALIBRATION)]

from camera import store                       # noqa: E402
from camera.locate import find_red_cube        # noqa: E402

RED_CUBES = [  # (frame, the rig's config)
    (IMPALA / "agenttools/arc/paper-now.png", WORK / "work/config-impala-old"),
    (IMPALA / "agenttools/arc/paper2-now.png", WORK / "work/config-impala-old"),
    (IMPALA / "agenttools/live-res/yuyv.png", WORK / "work/config-old"),
    (IMPALA / "agenttools/live-res/mjpg.png", WORK / "work/config-old"),
    (IMPALA / "agenttools/live-res/mjpg-single.png", WORK / "work/config-old"),
    (IMPALA / "config/camera-frame-previous.png", None),
]

for path, config in RED_CUBES:
    frame = cv2.imread(str(path))
    if config is None:
        import json
        from camera.fit import CameraFit
        fit = CameraFit.from_json(json.loads((IMPALA / "config/camera-previous.json").read_text()))
    else:
        fit = store.load_camera(config)
    try:
        found = find_red_cube(frame, fit)
        result = f"desk ({found.x:.2f}, {found.y:.2f}), covers {found.overlap:.2f}, edges {found.edge_px:.2f} px"
    except Exception as error:
        result = f"REFUSED: {error}"
    print(f"FIND RED CUBE {path.name}: {result}")
