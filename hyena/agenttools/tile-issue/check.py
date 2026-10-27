"""The library's finder and reader as they now are, on the issue frame, the stills and the drawn scenes."""
import math, sys, time
from pathlib import Path
import cv2
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "calibration"), str(HERE.parent)]
from maxarm import Camera
from applib.tiles import find_tiles
from applib.letters import read_letters
import tile_scene, test_tiles

shots = HERE.parent / "tile-shots"
frames = [("issue", cv2.imread(str(HERE / "frame.jpg")), Camera.load(HERE.parents[1] / "config/camera.json"), None)]
frames += [(n, cv2.imread(str(shots / f"{n}.png")), Camera.load(shots / "still-camera.json"), test_tiles.STILL_TILES[f"{n}.png"])
           for n in ("still-1", "still-2", "still-3")]
for desk in ("wood", "grey", "paper"):
    scene = tile_scene.render(test_tiles.drawn_layout(), desk=desk)
    frames.append((f"drawn {desk}", scene.frame, Camera(scene.desk_to_pixel, tile_scene.SIZE), [t[:2] for t in scene.tiles]))
for name, frame, camera, truth in frames:
    started = time.monotonic()
    tiles = find_tiles(frame, camera)
    took = time.monotonic() - started
    letters = read_letters(frame, camera, tiles)
    word = "".join((l.letter if l.is_read else "?") if l.letter else "-" for l in letters)
    line = f"{name:12s} {len(tiles):2d} found in {took:4.1f} s, weakest {min(t.contrast for t in tiles):5.2f}, letters {word}"
    if truth:
        offs = [min(math.hypot(t.x - x, t.y - y) for t in tiles) for x, y in truth]
        strays = sum(min(math.hypot(t.x - x, t.y - y) for x, y in truth) > 3 for t in tiles)
        line += f"; of {len(truth)}, worst {max(offs):5.2f} mm off, strays {strays}"
    print(line)
