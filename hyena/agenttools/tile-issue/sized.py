"""Experiment: the finder with the face's size fitted (25 to 27.5 mm) instead of fixed at 25.

    sized.py FRAME CAMERA.json        # a still
    sized.py drawn                    # tile_scene's wood, grey and paper scenes, against where they were drawn
"""
import math, sys, time
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE.parent)]
from maxarm import Camera
from applib.tiles import find as F, topdown, votes, pose
from applib.tiles.edges import EdgeField

SIZES = np.arange(25.0, 27.51, 0.5)


def mean_side(fit):
    return math.exp(np.mean(np.log(np.maximum(fit.sides, 1e-3))))


def find_sized(frame, camera, verbose=False):
    view = topdown.render(frame, camera, 2.0, (-F.SEARCH_MM, F.SEARCH_MM, -F.SEARCH_MM, F.SEARCH_MM))
    field = EdgeField(view, camera.position())
    fits, sizes = [], {}
    for candidate in votes.candidates(field, 12.5, 150):
        place = (candidate.x, candidate.y)
        if not F._is_searched(place) or F._is_on(fits, place):
            continue
        rough = pose.rough_fit(field, place, 25.5)
        if not rough.is_seen:
            continue
        _, size, best = max(((mean_side(fit), s, fit) for s in SIZES for fit in [pose.close_fit(field, rough, s)]),
                            key=lambda scored: scored[0])
        if best.is_seen and best.contrast >= F.MIN_CONTRAST:
            fits.append(best)
            sizes[id(best)] = size
        elif verbose and best.contrast > 1.5:
            print(f"  rejected ({best.centre[0]:6.1f},{best.centre[1]:6.1f}) size {size:4.1f} contrast {best.contrast:4.2f}")
    fits = F._without_overlaps(fits)
    return [(fit, sizes[id(fit)]) for fit in sorted(fits, key=lambda f: math.atan2(f.centre[0], -f.centre[1]))]


if __name__ == "__main__":
    if sys.argv[1] == "drawn":
        import tile_scene, test_tiles
        for desk in ("wood", "grey", "paper"):
            scene = tile_scene.render(test_tiles.drawn_layout(), desk=desk)
            camera = Camera(scene.desk_to_pixel, tile_scene.SIZE)
            for name, found in (("fixed 25", [(t.x, t.y, t.turn) for t in F.find_tiles(scene.frame, camera)]),
                                ("sized", [(f.centre[0], f.centre[1], f.turn_deg) for f, _ in find_sized(scene.frame, camera)])):
                offs = [min(math.hypot(x - tx, y - ty) for x, y, _ in found) if found else math.inf for tx, ty, _ in scene.tiles]
                strays = sum(min(math.hypot(x - tx, y - ty) for tx, ty, _ in scene.tiles) > 3 for x, y, _ in found)
                print(f"{desk:5s} {name:8s}: {len(found)} found, strays {strays}, mm off {[round(o, 2) for o in offs]}")
    else:
        started = time.monotonic()
        found = find_sized(cv2.imread(sys.argv[1]), Camera.load(sys.argv[2]), verbose=True)
        print(f"{len(found)} found in {time.monotonic() - started:.1f} s")
        for fit, size in found:
            print(f"  ({fit.centre[0]:6.1f},{fit.centre[1]:6.1f}) turn {fit.turn_deg:5.1f} size {size:4.1f} contrast {fit.contrast:5.2f}")


def draw(frame, camera, found, path):
    """Each sized fit's outline over the frame, pink, its size beside it."""
    drawn = frame.copy()
    for fit, size in found:
        corners = pose.square_corners(fit.centre, math.radians(fit.turn_deg), size)
        pixels = camera.to_pixels(np.column_stack((corners, np.full(4, 2.0))))
        cv2.polylines(drawn, [np.round(pixels).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
        top = pixels[np.argmin(pixels[:, 1])]
        for ink, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(drawn, f"{size:.1f} / {fit.contrast:.1f}", (round(top[0]) - 40, round(top[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, width, cv2.LINE_AA)
    legend = (f"{len(found)} tiles, face size fitted: pink, each top face as fitted (corners sharp); "
              f"above it, the size fitted, mm / its weakest side's contrast (the finder keeps 4 and up)")
    cv2.rectangle(drawn, (0, 0), (drawn.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(drawn, legend, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
