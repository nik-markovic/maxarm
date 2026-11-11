#!/usr/bin/env python3
"""A camera.json from the cubes' edges alone, the arm laid onto it at the cubes' tops only.

1. The camera: one camera for both of today's CALIBRATE frames (10:31 and 17:30, the camera unmoved), every cube by
   its edges, no arm numbers (walk-camera.py's best) -- the wireframes the owner checked ("GREAT").
2. The arm onto it: the 17:30 CALIBRATE's three readings (true as raw.py has it now: no turn fix) laid on those
   cubes' centres, at their tops (z 40), by a turn, shift and stretch -- six numbers, which three points fix
   exactly: the three tops with no error.
3. Checked with what is not in it: the morning's six points (CALIBRATE's three, Fine Tune's 4-6), each cube placed by
   its own edges through the same camera, against its reading made true the same way (taken back from the turn fix
   it was read with).
4. Written in the app's form (store.save_camera) to work/config-jackal-1110-edges/ -- not config/ -- and tried with
   the app's own code: loaded, the red cube found in the 17:30 frame (FIND RED CUBE's target), the check image.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/edges-calibration.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np

src = Path("agenttools/fitcheck/walk-camera.py").read_text().split("\n\ndef report(")[0]
ns = {}
exec(compile(src, "walk-camera", "exec"), ns)                 # the one camera for both frames: ns["cam_best"] etc.
from applib.camera import locate, store
from applib.camera.fit import HANDEDNESS, CameraFit, arm_measure
from applib.camera.model import project
from applib.camera.store import RefinePoint
from camera import check, refine

F = ns["F"]
cam, a_best, b_best, data = ns["cam_best"], ns["a_best"], ns["b_best"], ns["data"]
OUT_DIR = Path("../work/config-jackal-1110-edges")
TURN_THEN = (-1.36, 0.33)                                     # the turn fix the morning's readings were made with


def no_turn(x, y, turn):
    """A reading made true with `turn`, made true again without any: the reach kept, the servos' bearing back."""
    r, bt = math.hypot(x, y), math.atan2(x, -y)
    b = bt
    for _ in range(12):
        b = bt - math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return np.array((r * math.sin(b), -r * math.cos(b)))


evening = json.loads(Path("config/camera-points.json").read_text())["cubes"]
morning = json.loads(Path("../work/config-jackal-1110-1720/camera-points.json").read_text())["cubes"]
fine = [RefinePoint(**p) for p in
        json.loads(Path("../work/config-jackal-1110-1720/camera-refine-points.json").read_text())["points"]][3:]

# 2. the tie: desk (x, y) -> the camera's (x, y), exact at the evening's three cube centres
desk3 = np.array([(evening[c]["x"], evening[c]["y"]) for c in F.COLOURS])
world3 = np.array([b_best[k][0] for k in range(3)])
rows = np.column_stack((desk3, np.ones(3)))
tie = np.linalg.solve(rows, world3)                           # world = [x, y, 1] @ tie
A, shift = tie[:2].T, tie[2]                                  # world = A @ desk + shift
print(f"the tie, desk mm -> the camera's mm at the cubes' tops: {np.round(A, 5).tolist()} + {np.round(shift, 2).tolist()}")
scales = np.linalg.svd(A @ HANDEDNESS, compute_uv=False)
print(f"  its stretch: the arm's distances x{scales.min():.4f} to x{scales.max():.4f} as the camera measures them")
for k, c in enumerate(F.COLOURS):
    print(f"  17:30 {c:5}: the reading laid on its cube's top: {np.linalg.norm(A @ desk3[k] + shift - world3[k]):.3f} mm")

# 3. the morning's six, not in it
print("\nthe morning's points, not in the tie: each cube by its own edges through the camera, against its reading")
checks = []
for k, c in enumerate(F.COLOURS):
    d = no_turn(morning[c]["x"], morning[c]["y"], TURN_THEN)
    checks.append((f"10:31 {c}", d, a_best[k][0]))
for p in fine:
    d = no_turn(p.x, p.y, TURN_THEN)
    seen = refine._seen(cam.matrix, p.sighting)
    checks.append((f"point {p.id} ({p.taken[7:12]})", d, seen))
for name, d, w in checks:
    miss = A @ d + shift - w
    print(f"  {name:20} reading ({d[0]:7.1f}, {d[1]:7.1f}): the arm sent there would be {np.linalg.norm(miss):4.2f} mm "
          f"from the cube's top centre")

# 4. camera.json in the app's form
true_to_pixel = cam.matrix @ np.array([[1, 0, 0, shift[0]], [0, 1, 0, shift[1]], [0, 0, 1, 0], [0, 0, 0, 1.0]])
seen_true = [b_best[k][0] - shift for k in range(3)]
pixels = project(true_to_pixel, np.array([(*s, 0.0) for s in seen_true]))
cubes = {c: {"x": float(evening[c]["x"]), "y": float(evening[c]["y"]),
             "turn_deg": round(math.degrees(b_best[k][1]) % 90.0, 2),
             "pixel": [round(float(pixels[k][0]), 3), round(float(pixels[k][1]), 3)]} for k, c in enumerate(F.COLOURS)}
e = np.abs(ns["frame_misses"](cam, b_best, data["1730"]["found"]))
fit = CameraFit(true_to_pixel, A, cubes, (2560, 1472), (float(np.median(e)), float(np.percentile(e, 90))), len(e),
                arm_stretch=arm_measure([tuple(r) for r in desk3], seen_true))
OUT_DIR.mkdir(exist_ok=True)
store.save_camera(OUT_DIR, fit, lens=store.load_lens(Path("config")))
loaded = store.load_camera(OUT_DIR)
for k, c in enumerate(F.COLOURS):
    top = project(loaded.desk_to_pixel, np.array([[*desk3[k], 40.0]]))[0]
    want = project(cam.matrix, np.array([[*world3[k], 40.0]]))[0]
    print(f"camera.json, 17:30 {c}: the reading at z 40 drawn {np.linalg.norm(top - want):.4f} px from its cube's top centre")
frame = data["1730"]["frame"]
found = locate.find_red_cube(frame, loaded)
red = np.array((evening["red"]["x"], evening["red"]["y"]))
print(f"FIND RED CUBE on the 17:30 frame through it: target ({found.x:.2f}, {found.y:.2f}), the reading "
      f"({red[0]:.2f}, {red[1]:.2f}): {math.dist((found.x, found.y), red):.2f} mm; edges {found.edge_px:.2f} px")
image = check.draw(frame, loaded, ["camera.json from the cubes' edges alone, the arm laid on at the cubes' tops",
                                   store.describe_fit(loaded)])
cv2.imwrite("agenttools/fitcheck/out/edges-calibration-check.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"written: {OUT_DIR / 'camera.json'}; check image: agenttools/fitcheck/out/edges-calibration-check.jpg")
