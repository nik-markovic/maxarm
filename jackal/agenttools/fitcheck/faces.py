#!/usr/bin/env python3
"""How each point's cube shows its sides to CALIBRATE's camera: the width in the picture of each side face it sees,
and its edges found by edge; and the point placed again without the edges of its narrowest side. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/faces.py camera.json points.json
"""
import json, math, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import HANDEDNESS, CameraFit
from applib.camera.model import cube_corners, project, visible_edges, camera_position
from applib.camera.sighting import Sighting, place, start
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
m = fit.cube_fit().true_to_pixel
for p in json.loads(Path(sys.argv[2]).read_text())["points"]:
    s = RefinePoint(**p).sighting
    cube = place(m, s, start(m, s)[:3])
    corners = project(m, cube_corners(cube[:2], cube[2]))
    seen = visible_edges(cube[:2], cube[2], camera_position(m))
    sides = []
    for k in range(4):
        a, b = k, (k + 1) % 4
        if any({e.start, e.end} == {a, b} for e in seen):
            face = np.array([corners[a], corners[b], corners[b + 4], corners[a + 4]])
            # width across the face, perpendicular to its uprights, at mid height
            up = (face[3] - face[0] + face[2] - face[1]) / 2
            across = (face[1] - face[0] + face[2] - face[3]) / 2
            normal = np.array([-up[1], up[0]]) / np.linalg.norm(up)
            sides.append((k, abs(across @ normal)))
    st, en, f = s.edges
    counts = {}
    for a, b in zip(st, en):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    narrow = min(sides, key=lambda t: t[1])
    k = narrow[0]
    face_corners = {k, (k + 1) % 4, k + 4, (k + 1) % 4 + 4}
    keep = ~(np.isin(st, list(face_corners)) & np.isin(en, list(face_corners)))
    # edges wholly on that face, but keep the shared upright of the wider face? drop all edges of that face
    s2 = Sighting(s.x, s.y, s.pixel, s.frame_size, s.turn_deg, (st[keep], en[keep], f[keep]))
    c2 = place(m, s2, start(m, s2)[:3])
    d1 = np.linalg.norm(HANDEDNESS @ cube[:2] - (s.x, s.y))
    d2 = np.linalg.norm(HANDEDNESS @ c2[:2] - (s.x, s.y))
    print(f"#{p['id']} turn {p['turn_deg']:5.1f}: sides seen {', '.join(f'face {k} {w:5.0f} px' for k, w in sides)}; "
          f"miss {d1:4.2f} mm, without face {narrow[0]}'s edges {d2:4.2f} mm (moved "
          f"{np.linalg.norm(cube[:2] - c2[:2]):.2f})")
