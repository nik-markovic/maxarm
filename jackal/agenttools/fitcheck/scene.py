#!/usr/bin/env python3
"""A new scene, no arm readings: how far apart the three cubes are, from their edges alone (40 mm cubes).

The start CALIBRATE's step 1 needs (where the cubes roughly are on the desk) is taken from the picture: each blob's
centre carried down onto the desk at the cubes' half height through the edges-only camera.json -- only a start, the
arm's numbers play no part. Then edges-only-draw.py's fit: one camera and the three cubes from their edges (the
frame's own camera). Also the camera of both of today's CALIBRATE frames (walk-camera.py), its lens and focal length
held, only its pointing fitted again (the camera shifted ~0.9 px on the bare wood), the cubes from their edges.
Draws the wireframes: out/scene-<name>.jpg, out/scene-<name>-<colour>.jpg (3x).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/scene.py out/scene-1849.png
"""
import itertools, math, sys
from pathlib import Path
import cv2
import numpy as np
from applib.camera import cubes, edges, store, fit as F
from applib.camera.model import Pinhole, cube_corners, least_squares

FRAME = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agenttools/fitcheck/out/scene-1849.png")
NAME = FRAME.stem.replace("scene-", "")
OUT = Path("agenttools/fitcheck/out")
src = Path("agenttools/fitcheck/walk-camera.py").read_text().split("\n\ndef report(")[0]
ns = {}
exec(compile(src, "walk-camera", "exec"), ns)                 # the camera of both CALIBRATE frames: ns["cam_best"]
eod = ns["eod"]
COLOURS = list(F.COLOURS)
frame = cv2.imread(str(FRAME))
guide = store.load_camera(Path("../work/config-jackal-1110-edges"))


def on_desk(pixel, z=F.CUBE_MM / 2):
    m = guide.desk_to_pixel
    h = np.column_stack((m[:, 0], m[:, 1], z * m[:, 2] + m[:, 3]))
    q = np.linalg.solve(h, np.array([pixel[0], pixel[1], 1.0]))
    return q[:2] / q[2]


# 1-3: CALIBRATE's start, outlines and edges, the start's desk points from the picture; every naming of the blobs tried
starts = []
blobs = cubes.find_blobs(frame)
named = list(cubes.namings(blobs))
if not named:          # red not among the blobs (warm light on the wood): blue and green by hue, red the warm pass's largest
    by_hue = lambda lo, hi: max((b for b in blobs if lo <= b.hue <= hi), key=lambda b: b.area)
    named = [{"blue": by_hue(200, 300), "green": by_hue(100, 200),
              "red": max(cubes.find_blobs(frame, is_warm_only=True), key=lambda b: b.area)}]
    print("red taken from the warm-only pass: " + ", ".join(f"{c} ({b.centroid[0]:.0f}, {b.centroid[1]:.0f})"
                                                            for c, b in named[0].items()))
for naming in named:
    points = {c: on_desk(naming[c].centroid) for c in COLOURS}
    starts.append((*F._start(frame, naming, points), naming, points))
score, params, naming, points = max(starts, key=lambda s: s[0])
print(f"{FRAME}: {len(starts)} namings of the blobs tried; best overlap {score:.2f}; start points "
      + ", ".join(f"{c} ({p[0]:.0f}, {p[1]:.0f})" for c, p in points.items()))
params = F._fit_outlines(naming, points, params)
step3, _, found = F._fit_edges(frame, naming, points, params)
print(f"edge points found: {sum(len(f[2]) for f in found)} ({', '.join(f'{c} {len(f[2])}' for c, f in zip(COLOURS, found))})")

# the frame's own camera: edges-only-draw's fit from here (green's place and blue's direction held, all else free)
centres = F._centres(step3, points)
g, bl, rd = (COLOURS.index(c) for c in ("green", "blue", "red"))
axis = (centres[bl] - centres[g]) / np.linalg.norm(centres[bl] - centres[g])
normal = np.array((-axis[1], axis[0]))


def layout(p):
    cs = [None] * 3
    cs[g], cs[bl], cs[rd] = centres[g], centres[g] + p[10] * axis, centres[g] + p[12] * axis + p[13] * normal
    return cs, {g: p[9], bl: p[11], rd: p[14]}


def own_residuals(p):
    cam = Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9])
    cs, turns = layout(p)
    parts = [edges.signed_distances(cam.project(cube_corners(cs[k], turns[k])), *found[k]) for k in range(3)]
    parts.append((p[1:3] - np.array([frame.shape[1] / 2, frame.shape[0] / 2])) / (F.CENTRE_SPREAD * frame.shape[1]))
    return np.concatenate(parts)


rel = centres[rd] - centres[g]
start = np.concatenate((step3[:9], [step3[F.TURNS][g], np.linalg.norm(centres[bl] - centres[g]), step3[F.TURNS][bl],
                                    rel @ axis, rel @ normal, step3[F.TURNS][rd]]))
p, r, _ = least_squares(own_residuals, start, np.concatenate((F.STEPS[:9], [1e-5, 0.01, 1e-5, 0.01, 0.01, 1e-5])),
                        huber=F.EDGE_HUBER_PX, iterations=200)
own_cam = Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9])
cs, turns = layout(p)
own = [(cs[k], turns[k]) for k in range(3)]

# the camera of both CALIBRATE frames, re-pointed: focal, aim and lens held; its turn and the cubes fitted
held = ns["cam_best"]
r_held = cv2.Rodrigues(np.asarray(held.rotation, float))[0]


def held_camera(rot):
    r = cv2.Rodrigues(np.asarray(rot, float))[0]
    return Pinhole(held.focal, tuple(held.centre), rot, -r @ held.position)


def ground(cam, pixel):
    m = cam.matrix
    h = np.column_stack((m[:, 0], m[:, 1], m[:, 3]))
    q = np.linalg.solve(h, np.array([pixel[0], pixel[1], 1.0]))
    return q[:2] / q[2]


under = [own_cam.project(np.array([[*own[k][0], 0.0]]))[0] for k in range(3)]
starts_h = [ground(held, u) for u in under]
spin = math.atan2(*(starts_h[rd] - starts_h[bl])[::-1]) - math.atan2(*(own[rd][0] - own[bl][0])[::-1])
q0 = np.concatenate((held.rotation, *[(starts_h[k][0], starts_h[k][1], own[k][1] + spin) for k in range(3)]))


def held_residuals(q):
    cam = held_camera(q[:3])
    return np.concatenate([edges.signed_distances(cam.project(cube_corners(q[3 + 3 * k:5 + 3 * k], q[5 + 3 * k])),
                                                  *found[k]) for k in range(3)])


q, _, _ = least_squares(held_residuals, q0, np.array([1e-5] * 3 + [0.01, 0.01, 1e-5] * 3), huber=F.EDGE_HUBER_PX,
                        iterations=200)
held_cam = held_camera(q[:3])
held_cubes = [(q[3 + 3 * k:5 + 3 * k], q[5 + 3 * k]) for k in range(3)]
turned = math.degrees(math.acos(np.clip((np.trace(cv2.Rodrigues(q[:3])[0] @ r_held.T) - 1) / 2, -1, 1)))

for label, cam, placed in (("the frame's own camera", own_cam, own),
                           ("the camera of both CALIBRATE frames, re-pointed", held_cam, held_cubes)):
    e = np.abs(np.concatenate([edges.signed_distances(cam.project(cube_corners(*placed[k])), *found[k])
                               for k in range(3)]))
    print(f"\n{label}: edges {np.median(e):.2f} px typically, {np.percentile(e, 90):.2f} at 90%; lens "
          f"{cam.position[2]:.1f} mm up, focal {cam.focal:.0f} px"
          + (f"; turned {turned:.3f} deg from where it was" if cam is held_cam else ""))
    for a, b in itertools.combinations(range(3), 2):
        print(f"  {COLOURS[a]:5} - {COLOURS[b]:5} centres {np.linalg.norm(placed[a][0] - placed[b][0]):6.1f} mm apart")
    t = math.atan2(np.mean([math.sin(4 * c[1]) for c in placed]), np.mean([math.cos(4 * c[1]) for c in placed])) / 4
    face = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    for a, b in itertools.combinations(range(3), 2):
        u, v = np.abs(face @ (placed[b][0] - placed[a][0]))
        print(f"  {COLOURS[a]:5} -> {COLOURS[b]:5} along the cubes' faces: {max(u, v):6.1f} and {min(u, v):6.1f} mm")
    print(f"  cube turns relative to each other (deg, mod 90): "
          + ", ".join(f"{COLOURS[a]}-{COLOURS[b]} {math.degrees(placed[a][1] - placed[b][1]) % 90:.1f}"
                      for a, b in itertools.combinations(range(3), 2)))

whole = frame.copy()
for cube in own:
    eod["draw"](whole, own_cam, cube, width=2)
eod["label"](whole, [f"scene {NAME}: the cubes by their edges alone (no arm numbers): 40 mm wireframes",
                     "camera: this frame's own, from the edges",
                     "  ".join(f"{COLOURS[a]}-{COLOURS[b]} {np.linalg.norm(own[a][0] - own[b][0]):.1f} mm"
                               for a, b in itertools.combinations(range(3), 2))])
cv2.imwrite(str(OUT / f"scene-{NAME}-wire.jpg"), whole, [cv2.IMWRITE_JPEG_QUALITY, 92])
for k, cube in enumerate(own):
    pix = own_cam.project(cube_corners(*cube))
    lo, hi = pix.min(axis=0) - 40, pix.max(axis=0) + 40
    x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
    x1, y1 = min(int(hi[0]), frame.shape[1]), min(int(hi[1]), frame.shape[0])
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=eod["ZOOM"], fy=eod["ZOOM"], interpolation=cv2.INTER_CUBIC)
    eod["draw"](crop, own_cam, cube, origin=(x0, y0), zoom=eod["ZOOM"], width=1)
    e = np.abs(edges.signed_distances(pix, *found[k]))
    eod["label"](crop, [f"{COLOURS[k]}, scene {NAME}, 3x: edges {np.median(e):.2f} px typically, "
                        f"{np.percentile(e, 90):.2f} at 90%"], scale=0.9)
    cv2.imwrite(str(OUT / f"scene-{NAME}-{COLOURS[k]}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nwireframes: {OUT}/scene-{NAME}-wire.jpg and -<colour>.jpg")
