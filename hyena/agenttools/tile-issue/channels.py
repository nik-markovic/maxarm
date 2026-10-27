"""Per tile: the face's size measured per colour channel at the half-way level, and how clipped the face is."""
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
view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM), px_per_mm=8.0)
field = EdgeField(topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM)), camera.position())
cam = camera.position()
fits = [pose.close_fit(field, pose.rough_fit(field, (t.x, t.y), 25.0), 25.0) for t in F.find_tiles(frame, camera)]
for px in seeds:
    x, y, _ = camera.to_desk((px[0], px[1], 2.0))
    fits.append(pose.close_fit(field, pose.rough_fit(field, (x, y), 25.0), 25.0))
img = view.image.astype(np.float32)
steps = np.arange(-4, 5.01, 0.125)
def sample(points):
    u, v = view.to_view(points[..., 0], points[..., 1])
    return np.stack([cv2.remap(img[..., c], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR) for c in range(3)], -1)
for fit in fits:
    turn = math.radians(fit.turn_deg)
    points, normals = pose.outline(fit.centre, turn, 25.0, 0.5)
    # Middle of each side only (avoid the letter): along the straight part, profiles across.
    sizes = []
    for c in range(3):
        edge = []
        for side in range(4):
            prof = np.median(sample(points[side][None] + steps[:, None, None] * normals[side][None]), axis=1)[:, c]
            inside, outside = np.median(prof[steps < -2.5]), np.median(prof[steps > 3.0])
            half = (inside + outside) / 2
            idx = np.where((prof[:-1] >= half) & (prof[1:] < half))[0]
            idx = idx[steps[idx] > -2.5]
            edge.append(steps[idx[0]] + (prof[idx[0]] - half) / (prof[idx[0]] - prof[idx[0] + 1]) * 0.125 if len(idx) else np.nan)
        sizes.append((25 + edge[0] + edge[2] + 25 + edge[1] + edge[3]) / 2)
    corners = pose.square_corners(fit.centre, turn, 21.0)
    pix = camera.to_pixels(np.column_stack((corners, np.full(4, 2.0)))).astype(np.int32)
    mask = np.zeros(frame.shape[:2], np.uint8); cv2.fillConvexPoly(mask, pix, 255)
    face = frame[mask > 0]
    clipped = (face.max(axis=1) >= 254).mean()
    p0 = camera.from_desk((fit.centre[0], fit.centre[1], 2.0)); p1 = camera.from_desk((fit.centre[0] + 1, fit.centre[1], 2.0)); p2 = camera.from_desk((fit.centre[0], fit.centre[1] + 1, 2.0))
    ppm = math.sqrt(abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])))
    dist = math.hypot(*(cam[:2] - np.array(fit.centre)))
    print(f"pixel ({p0[0]:5.0f},{p0[1]:5.0f}) from camera {dist:4.0f} mm  {ppm:4.1f} px/mm  face BGR {np.median(face, axis=0).astype(int)} clipped {clipped:4.0%}  "
          f"size by B G R: {sizes[0]:4.1f} {sizes[1]:4.1f} {sizes[2]:4.1f}")
