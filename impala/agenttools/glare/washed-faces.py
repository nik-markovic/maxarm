"""Glare, drawn: on real calibration frames, a face of every cube washed out the way a bright light does it (light
added, 85% of the way to white, its edge a little soft), then CALIBRATE's fit run again; and the same on the red cube
of the C920 still for FIND RED CUBE. Each against the frame as it was. No arm, no camera.

    ../.venv/bin/python agenttools/glare/washed-faces.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

IMPALA = Path(__file__).resolve().parents[2]
WORK = IMPALA.parent
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]
from camera import check, store  # noqa: E402
from camera.fit import fit_camera  # noqa: E402
from camera.locate import find_red_cube  # noqa: E402

GLARE = 0.85


def wash(frame, polygons):
    mask = np.zeros(frame.shape[:2], np.float32)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None] * GLARE
    return (frame + mask * (255.0 - frame)).clip(0, 255).astype(np.uint8)


def faces(corners, seen, which):
    """The top, or each side the camera sees (its bottom edge is shown), as pixel polygons."""
    if which == "top":
        return [corners[4:8]]
    shown = {frozenset((edge.start, edge.end)) for edge in seen}
    return [corners[[i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i]] for i in range(4)
            if frozenset((i, (i + 1) % 4)) in shown]


import os
for config in [Path(p) for p in os.environ.get("CONFIGS", "").split(":") if p] or (IMPALA / "config", WORK / "work/config-impala-old"):
    frame = cv2.imread(str(config / "camera-frame.png"))
    points = {c: (p.x, p.y) for c, p in store.load_points(config).items() if p}
    plain = fit_camera(frame, points)
    for which in ("top", "a side"):
        polygons = []
        for colour in plain.cubes:
            wire = plain.wireframe(colour)
            found = faces(wire.corners, wire.seen, "top" if which == "top" else "sides")
            polygons += found[:1] if which == "top" else found[:1]
        washed = wash(frame, polygons)
        cv2.imwrite(f"/tmp/washed-{config.parent.name}-{which.replace(' ', '-')}.png", washed)
        try:
            fit = fit_camera(washed, points)
            typical, worst = check.difference_mm(plain, fit)
            offs = ", ".join(f"{c} {fit.cube_off_mm(c):.2f}" for c in fit.cubes)
            result = (f"edges {fit.edge_px[0]:.2f} px; cubes from the arm {offs} mm; the desk {typical:.2f} mm from "
                      f"the plain frame's calibration, {worst:.2f} at worst")
        except Exception as error:
            result = f"REFUSED {type(error).__name__}: {error}"
        print(f"CALIBRATE {config}, {which} of every cube washed out: {result}")

fit = store.load_camera(IMPALA / "config")
frame = cv2.imread("/tmp/c920-still.png")
plain = find_red_cube(frame, fit)
mapped = np.column_stack((plain.corners, np.ones(8))) @ fit.desk_to_pixel.T
corners = mapped[:, :2] / mapped[:, 2:]
for which, polygons in (("its sides, too", faces(corners, plain.seen, "sides")),
                        ("one side, the top not", faces(corners, plain.seen, "sides")[:1])):
    washed = wash(frame, polygons)
    try:
        found = find_red_cube(washed, fit)
        result = (f"desk ({found.x:.2f}, {found.y:.2f}), {np.hypot(found.x - plain.x, found.y - plain.y):.2f} mm "
                  f"from the plain frame's, edges {found.edge_px:.2f} px")
    except Exception as error:
        result = f"REFUSED {type(error).__name__}: {error}"
    print(f"FIND RED CUBE, C920, top washed by the light and {which}: {result}")
