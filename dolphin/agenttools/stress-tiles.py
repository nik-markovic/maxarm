#!/usr/bin/env python3
"""The same scene made harder, compared against what the finder says on the original.

- **Desks**: everything outside the tiles' silhouettes replaced -- lightness
  inverted (a light, grainy desk lighter than the tiles), and near-white paper
  (a smooth desk the tiles barely differ from). Crude: the silhouettes are the
  finder's own, and the arm and mouse get replaced along with the desk.
- **Resolutions**: the frame shrunk, with the calibration scaled to match,
  down to 640 wide -- what a cheaper camera, or the IMX95 pipeline, might give.

Not a substitute for real scenes; a check that nothing leans on the tiles being
lighter than the desk, or on this camera's 5 MP.

    ./agenttools/stress-tiles.py files/tiles-1.png
"""

import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import read                             # noqa: E402
import scene                            # noqa: E402
from mapping import DeskMapping         # noqa: E402

SAME_TILE_MM = 3.0


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    reader = read.Reader()
    truth = scene.find_tiles(frame, mapping, reader)
    print(f"original: {len(truth)} tiles, {''.join(sorted(t.letter for t in truth))}")
    for name, image in desks(frame, mapping, truth):
        report(name, scene.find_tiles(image, mapping, reader), truth, 0.0)
    for scale in (0.5, 0.35, 0.25):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        shrink = np.diag([1 / scale, 1 / scale, 1.0])
        scaled = dataclasses.replace(mapping, matrix=mapping.matrix @ shrink,
                                     lifted=mapping.lifted @ shrink)
        started = time.monotonic()
        found = scene.find_tiles(small, scaled, reader)
        report(f"{small.shape[1]}x{small.shape[0]}", found, truth, time.monotonic() - started)
    return 0


def desks(frame, mapping, tiles):
    silhouettes = np.zeros(frame.shape[:2], np.uint8)
    for tile in tiles:
        box = np.vstack((np.column_stack((tile.corners_mm, np.full(4, scene.TILE_HEIGHT_MM))),
                         np.column_stack((tile.corners_mm, np.zeros(4)))))
        hull = cv2.convexHull(mapping.world_to_pixel(box).astype(np.float32)).astype(np.int32)
        cv2.fillPoly(silhouettes, [hull], 255)
    silhouettes = cv2.dilate(silhouettes, np.ones((9, 9), np.uint8))
    keep = cv2.GaussianBlur(silhouettes.astype(np.float32) / 255, (0, 0), 2)[..., None]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = 255 - 0.5 * lab[..., 0]
    inverted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    noise = np.random.default_rng(0).normal(0, 2, frame.shape)
    paper = cv2.GaussianBlur(np.array((236, 240, 242), np.float32) + noise.astype(np.float32), (0, 0), 1.5)
    for name, desk in (("inverted desk", inverted), ("paper desk", paper)):
        yield name, np.clip(keep * frame + (1 - keep) * desk, 0, 255).astype(np.uint8)


def report(name, found, truth, elapsed):
    matched, wrong, false, moved, turned = 0, [], [], [], []
    for tile in found:
        nearest = min(truth, key=lambda t: np.hypot(*np.subtract(t.centre_mm, tile.centre_mm)))
        distance = float(np.hypot(*np.subtract(nearest.centre_mm, tile.centre_mm)))
        if distance > SAME_TILE_MM:
            false.append(f"{tile.letter}@({tile.centre_mm[0]:.0f},{tile.centre_mm[1]:.0f})")
            continue
        matched += 1
        moved.append(distance)
        turned.append(abs((nearest.baseline_deg - tile.baseline_deg + 180) % 360 - 180))
        if nearest.letter != tile.letter:
            wrong.append(f"{nearest.letter}->{tile.letter}")
    timing = f", {elapsed:.2f} s" if elapsed else ""
    print(f"{name:14s} {matched}/{len(truth)} tiles, {len(wrong)} misread {wrong or ''}, "
          f"{len(false)} false {false or ''}; centre max {max(moved, default=0):.1f} mm, "
          f"turn max {max(turned, default=0):.0f} deg{timing}")


if __name__ == "__main__":
    sys.exit(main())
