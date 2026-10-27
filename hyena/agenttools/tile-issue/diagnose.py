"""Why tiles are missed on the issue frame: votes, fits and contrast round given pixels."""
import math, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maxarm import Camera
from applib.tiles import find as F, topdown, votes, pose
from applib.tiles.edges import EdgeField

frame = cv2.imread(sys.argv[1]); camera = Camera.load(ROOT / "config/camera.json")
named = {"Z": (1264, 1257), "O": (1510, 1229), "T": (1925, 1134), "A": (975, 1260), "Tnear": (1100, 1570), "W": (1700, 960)}
view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM))
field = EdgeField(view, camera.position())
print("camera at", camera.position())
cands = votes.candidates(field, 12.5, 150)
best = cands[0].votes
for name, px in named.items():
    x, y, _ = camera.to_desk((px[0], px[1], 2.0))
    near = sorted(cands, key=lambda c: math.hypot(c.x - x, c.y - y))[0]
    d = math.hypot(near.x - x, near.y - y)
    rank = cands.index(near)
    rough = pose.rough_fit(field, (near.x, near.y), 25.0)
    fit = pose.close_fit(field, rough, 25.0)
    print(f"{name}: desk ({x:.1f},{y:.1f}) {math.hypot(x,y):.0f} mm; nearest peak {d:.1f} mm off, rank {rank}, "
          f"votes {near.votes/best:.2f}; rough ({rough.centre[0]:.1f},{rough.centre[1]:.1f}) {rough.turn_deg:.0f} "
          f"seen {rough.is_seen}; close ({fit.centre[0]:.1f},{fit.centre[1]:.1f}) {fit.turn_deg:.1f} "
          f"contrast {fit.contrast:.2f} sides {[round(s,1) for s in fit.sides]} beyond {[round(b,1) for b in fit.beyond]}")
cv2.imwrite(str(Path(sys.argv[1]).with_name("topdown.jpg")), view.image)
s, _ = field.strongest()
cv2.imwrite(str(Path(sys.argv[1]).with_name("edges.jpg")), np.clip(s * 255 / np.percentile(s, 99.5), 0, 255).astype(np.uint8))
print("view origin", view.origin, view.image.shape)
