"""If the cubes' outlines are smaller than a sharp 40 mm cube's, the camera fitted to them reads every length long.
The camera refitted on the calibration's own frame and points (config/camera-frame.png, camera-points.json) with the
model cube a little smaller each time; each camera written to OUT/<size>/camera.json for paper-lines.py. No arm;
config/ is only read.

    ../.venv/bin/python agenttools/live-res/cube-size-camera.py OUT 40 39.75 39.5 39.25 39
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import fit as fit_module  # noqa: E402
from camera import locate, model, store  # noqa: E402

out = Path(sys.argv[1])
frame = cv2.imread(str(ROOT / "config" / "camera-frame.png"))
points = {colour: (point.x, point.y) for colour, point in store.load_points(ROOT / "config").items()}
for size in (float(value) for value in sys.argv[2:]):
    for module in (model, fit_module, locate):
        module.CUBE_MM = size
    fitted = fit_module.fit_camera(frame, points)
    folder = out / f"{size:g}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "camera.json").write_text(json.dumps(fitted.to_json()))
    offs = ", ".join(f"{colour} {fitted.cube_off_mm(colour):.2f}" for colour in fitted.cubes)
    print(f"cube {size:5.2f} mm: edges {fitted.edge_px[0]:.2f} px typical, {fitted.edge_px[1]:.2f} worst; "
          f"cubes from the arm's points {offs} mm")
