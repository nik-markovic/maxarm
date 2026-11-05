#!/usr/bin/env python3
"""Each session's misses on a map of the desk: for every point, an arrow from the arm's reading to where the
session's camera sees its cube, the arm laid on all the points by a turn and a shift (as SAVE does), drawn 10x.
The arm's base axis at (0, 0); the camera's lens straight above the cross. Writes out/arrows.png.

    arrows.py config_dir ...
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from camera import refine  # noqa: E402

SCALE, GAIN = 1.6, 10.0                   # pixels per desk mm; arrow length per mm of miss
panels = []
for config in map(Path, sys.argv[1:]):
    saved = store.load_camera(config)
    points = [p for p in store.load_refine_points(config, saved)]
    sightings = [p.sighting for p in points]
    camera = refine.refine(saved, sightings) if any(s.edges is not None for s in sightings) else saved
    seen = np.array([refine._seen(refine.in_frame(camera.true_to_pixel, camera.frame_size, s), s) for s in sightings])
    arm = np.array([(s.x, s.y) for s in sightings])
    turn, shift, mirror = link_arm(arm, seen)
    seen_desk = np.linalg.solve(turn @ mirror, (seen - shift).T).T
    lens = camera.camera_position
    w, h = int(640 * SCALE / 1.6 * 1.0), int(480 * SCALE / 1.6)
    img = np.full((520, 640, 3), 255, np.uint8)

    def px(x, y):
        return int(320 - x * SCALE * 0.55), int(60 - y * SCALE * 0.55)   # +x to the left: the camera's mirror

    for r in (100, 150, 200, 250):
        cv2.ellipse(img, px(0, 0), (int(r * SCALE * 0.55),) * 2, 0, 0, 180, (225, 225, 225), 1, cv2.LINE_AA)
    for b in range(-90, 91, 30):
        e = px(270 * math.sin(math.radians(b)), -270 * math.cos(math.radians(b)))
        cv2.line(img, px(0, 0), e, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.drawMarker(img, px(0, 0), (0, 0, 0), cv2.MARKER_SQUARE, 10, 2)
    cx, cy = px(lens[0], lens[1])
    cv2.drawMarker(img, (min(max(cx, 8), 632), min(max(cy, 8), 512)), (200, 0, 200), cv2.MARKER_CROSS, 18, 2)
    for p, a, s in zip(points, arm, seen_desk):
        start = px(*a)
        end = px(*(a + GAIN * (s - a)))
        colour = (0, 0, 200) if np.linalg.norm(s - a) > 2.5 else (0, 140, 0) if np.linalg.norm(s - a) <= 1.5 else (0, 150, 220)
        cv2.circle(img, start, 3, (60, 60, 60), -1)
        cv2.arrowedLine(img, start, end, colour, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(img, str(p.id), (start[0] + 4, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    title = f"{config.name}: camera at desk ({lens[0]:.0f}, {lens[1]:.0f}), {lens[2]:.0f} up"
    cv2.putText(img, title, (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "arrow: arm reading -> cube as the camera sees it, x10 (turn+shift tie)", (10, 510),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(img, "square: arm base; magenta cross: camera (edge if off the map); +x to the left", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    panels.append(img)
rows = [np.hstack(panels[i:i + 2]) if i + 1 < len(panels) else np.hstack([panels[i], np.full_like(panels[i], 255)])
        for i in range(0, len(panels), 2)]
cv2.imwrite(str(Path(__file__).parent / "out" / "arrows.png"), np.vstack(rows))
