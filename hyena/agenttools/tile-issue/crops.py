"""Top-down crops round tiles: the fit (green), the lines 'beyond' each side (red = the side that fails), edges."""
import math, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maxarm import Camera
from applib.tiles import find as F, topdown, votes, pose
from applib.tiles.edges import EdgeField

frame = cv2.imread(sys.argv[1]); camera = Camera.load(ROOT / "config/camera.json")
named = {"Z": (1264, 1257), "O": (1510, 1229), "T": (1925, 1134), "Tnear": (1100, 1570), "W": (1700, 960), "Q": (1900, 410)}
view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM))
field = EdgeField(view, camera.position())
s, _ = field.strongest()
norm = np.percentile(s[field.seen], 99.5)
tiles = []
for name, px in named.items():
    x, y, _ = camera.to_desk((px[0], px[1], 2.0))
    fit = pose.close_fit(field, pose.rough_fit(field, (x, y), 25.0), 25.0)
    ratios = [a / b for a, b in zip(fit.sides, fit.beyond)]
    u, v = view.to_view(*fit.centre)
    half = 26 * view.px_per_mm
    box = (int(u - half), int(v - half), int(u + half), int(v + half))
    colour = view.image[box[1]:box[3], box[0]:box[2]].copy()
    edges = cv2.cvtColor(np.clip(s[box[1]:box[3], box[0]:box[2]] * 255 / norm, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    points, normals = pose.outline(fit.centre, math.radians(fit.turn_deg), 25.0, 0.5)
    for image in (colour, edges):
        for side in range(4):
            worst = side == int(np.argmin(ratios))
            for distance, ink in ((0, (0, 255, 0)),) + tuple((d, (0, 0, 255) if worst else (255, 160, 0)) for d in pose.BEYOND_MM):
                pu, pv = view.to_view(points[side][:, 0] + normals[side][:, 0] * distance, points[side][:, 1] + normals[side][:, 1] * distance)
                for a, b in zip(pu, pv):
                    image[int(round(b)) - box[1], int(round(a)) - box[0]] = ink
    tile = np.hstack((colour, edges))
    tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, f"{name} contrast {fit.contrast:.2f} sides/beyond {' '.join(f'{r:.1f}' for r in ratios)}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(tile)
legend = np.zeros((40, tiles[0].shape[1], 3), np.uint8)
cv2.putText(legend, "top-down view at the tiles' top, 4 px/mm, x2. green: fitted side; red: lines 6/7.5/9 mm beyond the weakest-ratio side; orange: the other sides' beyond lines. right: edge strength", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(str(Path(sys.argv[1]).with_name("crops.jpg")), np.vstack([legend] + tiles))
