"""Per tile: how wide its edge is (the rise from desk to face, 20% to 80%), in mm and in frame pixels."""
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
box = (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM)
fine = topdown.render(frame, camera, 2.0, box, px_per_mm=10.0)
field = EdgeField(topdown.render(frame, camera, 2.0, box), camera.position())
cam = camera.position()
fits = [pose.close_fit(field, pose.rough_fit(field, (t.x, t.y), 25.0), 25.0) for t in F.find_tiles(frame, camera)]
for px in seeds:
    x, y, _ = camera.to_desk((px[0], px[1], 2.0))
    fits.append(pose.close_fit(field, pose.rough_fit(field, (x, y), 25.0), 25.0))
green = fine.image[..., 1].astype(np.float32)
steps = np.arange(-4, 5.01, 0.05)
rows = []
for fit in fits:
    points, normals = pose.outline(fit.centre, math.radians(fit.turn_deg), 25.0, 0.5)
    to_cam = cam[:2] - np.array(fit.centre); to_cam /= np.linalg.norm(to_cam)
    widths = []
    for side in range(4):
        if normals[side][0] @ to_cam > 0.3:
            continue        # the side face shows beyond the sides facing the camera
        pts = points[side][None] + steps[:, None, None] * normals[side][None]
        u, v = fine.to_view(pts[..., 0], pts[..., 1])
        prof = np.median(cv2.remap(green, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR), axis=1)
        inside, outside = np.median(prof[steps < -2.5]), np.median(prof[steps > 3.5])
        level = lambda f: steps[np.where(prof >= outside + f * (inside - outside))[0].max()]
        widths.append(level(0.2) - level(0.8))
    p0 = camera.from_desk((fit.centre[0], fit.centre[1], 2.0)); p1 = camera.from_desk((fit.centre[0] + 1, fit.centre[1], 2.0)); p2 = camera.from_desk((fit.centre[0], fit.centre[1] + 1, 2.0))
    ppm = math.sqrt(abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])))
    w = float(np.mean(widths))
    print(f"pixel ({p0[0]:5.0f},{p0[1]:5.0f}) from camera {math.hypot(*(cam[:2]-np.array(fit.centre))):4.0f} mm  {ppm:4.1f} px/mm  edge 20-80%: {w:4.2f} mm = {w*ppm:4.1f} px")
