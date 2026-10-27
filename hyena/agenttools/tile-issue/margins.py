"""Every candidate's sized fit on a frame: contrast as the finder measures it (median of the three lines beyond)
and as the quietest of the three lines; the true tiles' lowest against the best of the rest."""
import math, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "calibration"), str(HERE.parent), str(HERE)]
from maxarm import Camera
from applib.tiles import find as F, topdown, votes, pose
from applib.tiles.edges import EdgeField
from sized import SIZES, mean_side


def contrasts(field, fit, size):
    points, normals = pose.outline(fit.centre, math.radians(fit.turn_deg), size, pose.SAMPLE_MM)
    lines = np.array([np.median(field.across(points + normals * d, normals), axis=1) for d in pose.BEYOND_MM])
    sides = np.array(fit.sides)
    return float(np.min(sides / np.maximum(fit.beyond, 1e-3))), float(np.min(sides / np.maximum(lines.min(axis=0), 1e-3)))


def margins(frame, camera, truth):
    view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM))
    field = EdgeField(view, camera.position())
    tiles, others = [], []
    for candidate in votes.candidates(field, 12.5, 150):
        place = (candidate.x, candidate.y)
        if not F._is_searched(place):
            continue
        rough = pose.rough_fit(field, place, 25.5)
        if not rough.is_seen:
            continue
        _, size, fit = max(((mean_side(f), s, f) for s in SIZES for f in [pose.close_fit(field, rough, s)]), key=lambda t: t[0])
        if not fit.is_seen:
            continue
        near = min(math.hypot(fit.centre[0] - x, fit.centre[1] - y) for x, y in truth)
        (tiles if near < 2.0 else others if near > 8.0 else []).append(contrasts(field, fit, size))
    old_t, new_t = min(t[0] for t in tiles), min(t[1] for t in tiles)
    old_o, new_o = max((o[0] for o in others), default=0), max((o[1] for o in others), default=0)
    found = {round(x) for x in []}
    return f"true tiles' lowest: median {old_t:5.2f}, quietest {new_t:5.2f};  best non-tile: median {old_o:5.2f}, quietest {new_o:5.2f}"


if __name__ == "__main__":
    import tile_scene, test_tiles
    from sized import find_sized
    shots = HERE.parent / "tile-shots"
    issue = cv2.imread(str(HERE / "frame.jpg")); issue_cam = Camera.load(HERE.parents[1] / "config/camera.json")
    # The issue frame's truth: the 15 tiles as the sized finder put them (checked by eye on the drawn outlines).
    frames = [("issue", issue, issue_cam, [f.centre for f, _ in find_sized(issue, issue_cam)])]
    for name in ("still-1", "still-2", "still-3"):
        frame, cam = cv2.imread(str(shots / f"{name}.png")), Camera.load(shots / "still-camera.json")
        frames.append((name, frame, cam, test_tiles.STILL_TILES[f"{name}.png"]))
    for desk in ("wood", "grey", "paper"):
        scene = tile_scene.render(test_tiles.drawn_layout(), desk=desk)
        frames.append((f"drawn {desk}", scene.frame, Camera(scene.desk_to_pixel, tile_scene.SIZE), [t[:2] for t in scene.tiles]))
    for name, frame, cam, truth in frames:
        print(f"{name:12s} {len(truth):2d} tiles  {margins(frame, cam, truth)}")
