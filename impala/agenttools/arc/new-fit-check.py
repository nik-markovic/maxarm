"""The new calibration (arm placed by a turn and shift) on the owner's saved frame, points and red cube points,
written to a scratch directory -- config/ is not touched. Then: each point's off, the axis, and how far the new
calibration puts the desk from the saved camera.json. No arm, no camera.

python new-fit-check.py config-dir out-dir
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import check, store  # noqa: E402
from camera.fit import fit_camera  # noqa: E402
from camera.model import project  # noqa: E402
from camera.refine import cube_sightings, off_mm, refine  # noqa: E402

config, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
frame = cv2.imread(str(config / store.FRAME_FILE))
points = store.load_points(config)
refine_points = store.load_refine_points(config)
saved = store.load_camera(config)
fit = fit_camera(frame, {colour: (p.x, p.y) for colour, p in points.items()})
print("CALIBRATE (three cubes):", store.describe_fit(fit))
refined = refine(fit, [p.sighting for p in refine_points])
print("SAVE (with the red cube points):", store.describe_fit(refined))
print(" point   off with the three cubes alone   after SAVE   (saved camera.json: now)")
for p in refine_points:
    s = p.sighting
    print(f"  {p.id:3d}       {off_mm(fit, s):5.2f}                       {off_mm(refined, s):5.2f}"
          f"        {off_mm(saved, s):5.2f}")
for colour, s in zip(fit.cubes, cube_sightings(fit)):
    print(f"  {colour:5s}     {off_mm(fit, s):5.2f}                       {off_mm(refined, s):5.2f}"
          f"        {off_mm(saved, s):5.2f}")
for label, a, b in (("new three cubes vs new SAVE", fit, refined), ("new SAVE vs saved camera.json", refined, saved),
                    ("new three cubes vs saved three cubes", fit, saved.cube_fit())):
    typical, worst = check.difference_mm(a, b)
    o = project(a.desk_to_pixel, np.array([[0.0, 0, 0]]))[0]
    H = b.desk_to_pixel[:, [0, 1, 3]]
    q = np.linalg.solve(H, (*o, 1.0))
    print(f"{label}: the desk {typical:.2f} mm apart typically, {worst:.2f} at worst; the axis {np.hypot(*q[:2] / q[2]):.2f} mm")
store.save_camera(out, refined)
cv2.imwrite(str(out / "check-new.jpg"), check.draw(frame, refined, ["the new calibration, refined, on the saved frame",
                                                                    store.describe_fit(refined)]))
