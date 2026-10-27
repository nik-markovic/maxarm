"""Every tile's dot, as ink.py finds it and with two guards: round-ish (no longer than twice its width), and still
apart from the letter at a quarter of the strongest ink, not only at half."""
import sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1])]
from maxarm import Camera
from applib.tiles import find_tiles
from applib.letters import face, ink


def guarded_dot(grey):
    found = ink.measure(grey)
    if found.dot is None:
        return None
    image = grey.astype(np.float32)
    patch = ink._odd(ink.PATCH_MM * ink.PX_PER_MM)
    light = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch, patch)))
    contrast = (light - image) / np.maximum(light, 1.0)
    strongest = float(np.percentile(contrast, ink.STRONGEST_PERCENTILE))
    size = image.shape[0]
    at = tuple(int(round(v * ink.PX_PER_MM + (size - 1) / 2)) for v in found.dot)
    high = contrast >= max(strongest / 2, ink.MIN_INK_CONTRAST / 2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8), connectivity=8)
    dot = labels[at[1], at[0]]
    _, _, width, height, _ = stats[dot]
    if max(width, height) > 2 * min(width, height):
        return None
    _, low_labels = cv2.connectedComponents((contrast >= strongest / 4).astype(np.uint8), connectivity=8)
    joined = low_labels[at[1], at[0]]
    if ((low_labels == joined) & high & (labels != dot)).any():
        return None
    return found.dot


shots = HERE.parent / "tile-shots"
frames = [("issue", HERE / "frame.jpg", HERE.parents[1] / "config/camera.json")] + \
         [(n, shots / f"{n}.png", shots / "still-camera.json") for n in ("still-1", "still-2", "still-3")]
for name, path, cam_path in frames:
    frame, camera = cv2.imread(str(path)), Camera.load(cam_path)
    print(name)
    for tile in find_tiles(frame, camera):
        grey = face.window(frame, camera, tile, 0, ink.WINDOW_MM, ink.size_px())
        before, after = ink.measure(grey).dot, guarded_dot(grey)
        if before or after:
            print(f"  ({tile.x:6.1f},{tile.y:6.1f})  dot now {before is not None}, guarded {after is not None}")
