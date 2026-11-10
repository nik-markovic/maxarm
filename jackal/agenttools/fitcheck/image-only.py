#!/usr/bin/env python3
"""The picture alone: the six top faces as 40 mm squares on one plane through one homography, every cube free but
#3 and #6 (held at their readings only to fix where the plane lies and which way it turns). How well the squares
fit (corners, px) says whether the picture is one plane of 40 mm squares; where the free cubes land, against the
readings, is the picture's layout against the arm's.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/image-only.py
"""
import json, math
import numpy as np
from applib.camera.fit import HANDEDNESS
from squares import Squares, load
from tops import centre_of, homography

stored, readings, corners, turns = load()
n = len(stored)
ids = [p["id"] for p in stored]
centres = np.array([centre_of({k: c for k, c in zip((4, 5, 6, 7), cs)}) for cs in corners])
for held in ([2, 5], [0, 1], [0, 2], [1, 2], [2]):
    free = [k for k in range(n) if k not in held]
    for size_free in (False, True):
        model = Squares(readings, corners, range(n), free, size_free)
        p, r = model.fit(turns, homography(readings, centres))
        h, tt, size, at = model.unpack(p)
        per = np.sqrt((r.reshape(n, 4, 2) ** 2).sum(axis=2).mean(axis=1))
        print(f"held {'+'.join(str(ids[k]) for k in held)}, size {'free' if size_free else 'held'} {size:.2f}: corners rms px "
              + " ".join(f"#{i} {v:4.2f}" for i, v in zip(ids, per)))
        print("   free cubes against their readings, mm: " + "  ".join(
            f"#{ids[k]} {np.linalg.norm(at[k] - readings[k]):4.2f} (dx {(HANDEDNESS @ (at[k] - readings[k]))[0]:+5.2f}, "
            f"dy {(HANDEDNESS @ (at[k] - readings[k]))[1]:+5.2f})" for k in free))
# the per-corner pattern for one run
model = Squares(readings, corners, range(n), [0, 1, 3, 4], False)
p, r = model.fit(turns, homography(readings, centres))
r = r.reshape(n, 4, 2)
print("\nheld 3+6, size 40: each corner's miss (px, dx dy) -- corners 4 5 6 7")
for k in range(n):
    print(f"  #{ids[k]}: " + "  ".join(f"({r[k, j, 0]:+5.1f},{r[k, j, 1]:+5.1f})" for j in range(4)))
