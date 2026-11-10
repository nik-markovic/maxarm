#!/usr/bin/env python3
"""The cubes' top faces as what they are -- 40 mm squares, 40 mm over the desk -- through one plane-to-image map (a
homography: no lens, focal length, principal point or 3-D cube), against the arm's readings. Each top's corners are
where the found lines through them cross (tops.py). Each square's centre is held at its reading, or left free (then
the squares' shapes and the other points alone place it); each turns as it likes; their size is one number, free
(the outlines' bleed), or held at 40.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/squares.py [refine-points.json]
"""
import itertools, json, math, sys
from pathlib import Path

import numpy as np

from applib.camera.fit import HANDEDNESS
from applib.camera.model import least_squares
from tops import POINTS, apply, centre_of, homography, top_face

SQUARE = np.array(((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)))   # corners 4, 5, 6, 7 as model.py


def load():
    stored = json.loads(POINTS.read_text())["points"]
    readings, corners, turns = [], [], []
    for p in stored:
        c, _ = top_face(p)
        readings.append(HANDEDNESS @ np.array((p["x"], p["y"])))
        corners.append(np.array([c[k] for k in (4, 5, 6, 7)]))
        turns.append(math.radians(p["turn_deg"]))
    return stored, np.array(readings), corners, np.array(turns)


class Squares:
    """Parameters: 8 of the homography, a turn per cube, the size, then x, y for each free cube."""

    def __init__(self, readings, corners, used, free, size_free=True):
        self.readings, self.corners, self.used, self.free = readings, corners, list(used), list(free)
        self.size_free = size_free

    def unpack(self, p):
        h = np.append(p[:8], 1.0).reshape(3, 3)
        turns = p[8:8 + len(self.used)]
        size = p[8 + len(self.used)] if self.size_free else 40.0
        at = dict(zip(self.free, p[8 + len(self.used) + int(self.size_free):].reshape(-1, 2)))
        return h, turns, size, at

    def residuals(self, p):
        h, turns, size, at = self.unpack(p)
        out = []
        for k, turn in zip(self.used, turns):
            centre = at.get(k, self.readings[k])
            c, s = math.cos(turn), math.sin(turn)
            square = centre + size * SQUARE @ np.array([[c, s], [-s, c]])
            out.append((apply(h, square) - self.corners[k]).ravel())
        return np.concatenate(out)

    def fit(self, turns, start_h):
        start = np.concatenate((start_h.ravel()[:8], turns[self.used], [40.0] if self.size_free else [],
                                np.concatenate([self.readings[k] for k in self.free]) if self.free else []))
        steps = np.concatenate((np.maximum(np.abs(start_h.ravel()[:8]), 1e-9) * 1e-6, [1e-5] * len(self.used),
                                [1e-3] if self.size_free else [], [1e-3] * (2 * len(self.free))))
        p, r, _ = least_squares(self.residuals, start, steps, huber=1e9, iterations=300)
        return p, r


def main():
    stored, readings, corners, turns = load()
    n = len(stored)
    ids = [p["id"] for p in stored]
    centres = np.array([centre_of({k: c for k, c in zip((4, 5, 6, 7), cs)}) for cs in corners])
    start_h = homography(readings, centres)

    for size_free in (True, False):
        label = "size free" if size_free else "size 40"
        model = Squares(readings, corners, range(n), [], size_free)
        p, r = model.fit(turns, start_h)
        h, _, size, _ = model.unpack(p)
        per = np.sqrt((r.reshape(n, 4, 2) ** 2).sum(axis=2).mean(axis=1))
        print(f"All {n} squares, centres at the readings, {label} ({size:.2f} mm): corner rms px per cube "
              + " ".join(f"#{i} {v:4.2f}" for i, v in zip(ids, per)))

        print(f"  each cube's centre left free, the others at their readings ({label}): where its square puts it")
        for k in range(n):
            model = Squares(readings, corners, range(n), [k], size_free)
            p, r = model.fit(turns, start_h)
            _, _, size, at = model.unpack(p)
            d = HANDEDNESS @ (at[k] - readings[k])
            print(f"    #{ids[k]}: {np.linalg.norm(d):4.2f} mm (dx {d[0]:+5.2f}, dy {d[1]:+5.2f}); corners rms "
                  f"{np.sqrt(np.mean(r ** 2)):4.2f} px, size {size:.2f}")

    print("\nThree squares at their readings predicting the others (their centres free), size free:")
    every = []
    for three in itertools.combinations(range(n), 3):
        rest = [k for k in range(n) if k not in three]
        model = Squares(readings, corners, range(n), rest, True)
        try:
            p, r = model.fit(turns, homography(readings, centres))
        except np.linalg.LinAlgError:
            continue
        _, _, size, at = model.unpack(p)
        offs = {k: float(np.linalg.norm(at[k] - readings[k])) for k in rest}
        every += offs.values()
        print(f"  {'+'.join(str(ids[k]) for k in three):7} -> " + "  ".join(f"#{ids[k]} {v:4.2f}" for k, v in offs.items())
              + f"   size {size:.2f}")
    print(f"  median {np.median(every):.2f}, 90% {np.percentile(every, 90):.2f}, worst {max(every):.2f}")

    print("\nOnly three squares used at all (the others' edges unused), predicting the rest by their top centres:")
    every = []
    for three in itertools.combinations(range(n), 3):
        rest = [k for k in range(n) if k not in three]
        model = Squares(readings, corners, three, [], True)
        try:
            p, r = model.fit(turns, homography(readings, centres))
        except np.linalg.LinAlgError:
            continue
        h, _, size, _ = model.unpack(p)
        inv = np.linalg.inv(h)
        offs = {k: float(np.linalg.norm(apply(inv, centres[k][None])[0] - readings[k])) for k in rest}
        every += offs.values()
        print(f"  {'+'.join(str(ids[k]) for k in three):7} -> " + "  ".join(f"#{ids[k]} {v:4.2f}" for k, v in offs.items())
              + f"   size {size:.2f}, corners rms {np.sqrt(np.mean(r ** 2)):.2f} px")
    print(f"  median {np.median(every):.2f}, 90% {np.percentile(every, 90):.2f}, worst {max(every):.2f}")


if __name__ == "__main__":
    main()
