"""Each tile's sides: where its strongest edge is, from the fitted 25 mm square's side, mm (+ out), per side."""
import math, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maxarm import Camera
from applib.tiles import find as F, topdown, pose
from applib.tiles.edges import EdgeField

frame = cv2.imread(sys.argv[1]); camera = Camera.load(sys.argv[2])
seeds = [tuple(map(float, s.split(","))) for s in sys.argv[3:]]
view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM))
field = EdgeField(view, camera.position())
cam = camera.position()
fits = [pose.close_fit(field, pose.rough_fit(field, (t.x, t.y), 25.0), 25.0) for t in F.find_tiles(frame, camera)]
for px in seeds:
    x, y, _ = camera.to_desk((px[0], px[1], 2.0))
    fits.append(pose.close_fit(field, pose.rough_fit(field, (x, y), 25.0), 25.0))
offsets = np.arange(-3, 3.01, 0.1)
for fit in fits:
    points, normals = pose.outline(fit.centre, math.radians(fit.turn_deg), 25.0, 0.5)
    to_cam = cam[:2] - np.array(fit.centre); to_cam /= np.linalg.norm(to_cam)
    found = []
    for side in range(4):
        strength = [np.percentile(field.across(points[side] + normals[side] * d, normals[side]), 20) for d in offsets]
        found.append((offsets[int(np.argmax(strength))], normals[side][0] @ to_cam))
    dist = math.hypot(*(cam[:2] - np.array(fit.centre)))
    size_a = 25 + found[0][0] + found[2][0]; size_b = 25 + found[1][0] + found[3][0]
    px, py, _ = camera.from_desk((fit.centre[0], fit.centre[1], 2.0))
    sides = "  ".join(f"{d:+.1f}{'c' if f > 0.3 else ' '}" for d, f in found)
    print(f"pixel ({px:5.0f},{py:5.0f}) desk ({fit.centre[0]:6.1f},{fit.centre[1]:6.1f}) from camera {dist:4.0f} mm  "
          f"contrast {fit.contrast:5.2f}  sides {sides}  size {size_a:4.1f} x {size_b:4.1f}")
