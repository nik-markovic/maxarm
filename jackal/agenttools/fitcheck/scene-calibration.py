#!/usr/bin/env python3
"""A spare calibration for the 18:55 scene, no arm readings: each cube found through the calibration in config/ the
way FIND RED CUBE finds the red one (locate.py's fit, that camera held) -- blue and green too -- and those three
taken as if the arm had recorded them, z 40 (the tops). The camera: the frame's own, from the cubes' edges alone
(scene.py). The three points laid on the cubes' tops by a turn, shift and stretch: exact there. Written in the app's
form -- camera.json, camera-points.json, camera-refine-points.json, camera-frame.png, camera-check.jpg -- to
work/config-jackal-1110-1855-spare/, not config/.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/scene-calibration.py
"""
import math, sys
from pathlib import Path
import numpy as np

sys.argv = [sys.argv[0], "agenttools/fitcheck/out/scene-1855.png"]
src = Path("agenttools/fitcheck/scene.py").read_text().split("\nfor label, cam, placed in")[0]
exec(compile(src, "scene", "exec"))                           # frame, naming, found, own_cam, own, COLOURS, edges ...
from applib.camera import locate
from applib.camera.fit import CameraFit, arm_measure
from applib.camera.model import project
from applib.camera.store import CubePoint
from camera import check

OUT_DIR = Path("../work/config-jackal-1110-1855-spare")
old = store.load_camera(Path("config"))                       # the calibration in config/: 17:30's CALIBRATE

# each cube found through it, as FIND RED CUBE does (the camera held, the cube's edges fitted)
matrix = locate.view_camera(old, frame)
points, desk3 = {}, []
for c in COLOURS:
    overlap, params_c, distances, found_c = locate._fit(frame, matrix, naming[c])
    cube = locate._found(old, matrix, params_c, overlap, distances, found_c)
    points[c] = CubePoint(round(cube.x, 2), round(cube.y, 2), 40.0, [round(cube.x, 2), round(cube.y, 2), 40.0])
    desk3.append((points[c].x, points[c].y))
    print(f"found through config/camera.json: {c:5} ({cube.x:8.2f}, {cube.y:8.2f}), edges {cube.edge_px:.2f} px, "
          f"covers {overlap:.0%}")
desk3 = np.array(desk3)

# the tie: the found points onto the cubes' centres (the frame's own camera), exact
world3 = np.array([own[k][0] for k in range(3)])
tie = np.linalg.solve(np.column_stack((desk3, np.ones(3))), world3)
A, shift = tie[:2].T, tie[2]
scales = np.linalg.svd(A @ F.HANDEDNESS, compute_uv=False)
print(f"the tie: stretch x{scales.min():.4f} to x{scales.max():.4f} (config/'s millimetres as the cubes measure them)")

true_to_pixel = own_cam.matrix @ np.array([[1, 0, 0, shift[0]], [0, 1, 0, shift[1]], [0, 0, 1, 0], [0, 0, 0, 1.0]])
seen_true = [world3[k] - shift for k in range(3)]
pixels = project(true_to_pixel, np.array([(*s, 0.0) for s in seen_true]))
cubes_json = {c: {"x": float(points[c].x), "y": float(points[c].y), "turn_deg": round(math.degrees(own[k][1]) % 90, 2),
                  "pixel": [round(float(pixels[k][0]), 3), round(float(pixels[k][1]), 3)]} for k, c in enumerate(COLOURS)}
e = np.abs(np.concatenate([edges.signed_distances(own_cam.project(cube_corners(*own[k])), *found[k]) for k in range(3)]))
fit = CameraFit(true_to_pixel, A, cubes_json, (frame.shape[1], frame.shape[0]),
                (float(np.median(e)), float(np.percentile(e, 90))), len(e),
                cube_edges={c: (round(math.degrees(own[k][1]), 4), found[k]) for k, c in enumerate(COLOURS)},
                arm_stretch=arm_measure([tuple(r) for r in desk3], seen_true))
OUT_DIR.mkdir(exist_ok=True)
store.save_points(OUT_DIR, points)
store.save_refine_points(OUT_DIR, store.cube_refine_points(fit, points, "10 Nov 18:55"))
store.save_frame(OUT_DIR, frame)
store.save_camera(OUT_DIR, fit, lens=store.load_lens(Path("config")))
loaded = store.load_camera(OUT_DIR)
store.save_check(OUT_DIR, check.draw(frame, loaded, ["spare: the 18:55 scene, the cubes' points found through 17:30's "
                                                     "calibration, the camera from the edges", store.describe_fit(loaded)]))

for k, c in enumerate(COLOURS):
    top = project(loaded.desk_to_pixel, np.array([[*desk3[k], 40.0]]))[0]
    want = own_cam.project(np.array([[*world3[k], 40.0]]))[0]
    print(f"spare camera.json: the {c:5} point at z 40 drawn {np.linalg.norm(top - want):.4f} px from its cube's top centre")
red = locate.find_red_cube(frame, loaded)
print(f"FIND RED CUBE through the spare: ({red.x:.2f}, {red.y:.2f}), the red point "
      f"({points['red'].x:.2f}, {points['red'].y:.2f}): {math.dist((red.x, red.y), (points['red'].x, points['red'].y)):.2f} mm")
print(f"written: {sorted(p.name for p in OUT_DIR.iterdir())}")
