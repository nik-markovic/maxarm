#!/usr/bin/env python3
"""Walk the camera and watch the edges: one camera for both of today's CALIBRATE frames (10:31 and 17:30; it did not
move, 0.25 px on the bare desk), every cube of both by its edges alone (no arm numbers). First the camera that fits
both best; then the lens held at steps along its line of sight -- the way three cubes fix it least -- everything else
fitted again at each step (focal length, aim, which way it looks, the cubes). How the edges fit, each frame and both,
step by step; and the 17:30 wireframes drawn with the lens held at the far ends of the walk (out/walk-*.jpg).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/walk-camera.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np
from applib.camera import edges, fit as F
from applib.camera.model import Pinhole, cube_corners, least_squares

src = Path("agenttools/fitcheck/edges-only-draw.py").read_text().split("\nfor stamp, config in FRAMES.items():")[0]
eod = {}
exec(compile(src, "edges-only-draw", "exec"), eod)
OUT = Path("agenttools/fitcheck/out")
FRAMES = {"1031": Path("../work/config-jackal-1110-1720"), "1730": Path("config")}

data = {}
for stamp, config in FRAMES.items():
    frame = cv2.imread(str(config / "camera-frame.png"))
    rec = json.loads((config / "camera-points.json").read_text())["cubes"]
    cam, placed, found, _ = eod["edges_only"](frame, {c: (p["x"], p["y"]) for c, p in rec.items()})
    data[stamp] = dict(frame=frame, cam=cam, placed=placed, found=found)


def desk_of(cam, pixels):
    m = cam.matrix
    h = np.column_stack((m[:, 0], m[:, 1], m[:, 3]))
    q = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(h).T
    return q[:, :2] / q[:, 2:]


# 17:30's cubes into 10:31's desk: the two fits' desks laid on each other by where they put the same pixels
A, B = data["1031"], data["1730"]
us, vs = np.meshgrid(np.linspace(150, 2410, 40), np.linspace(150, 1320, 24))
pix = np.column_stack((us.ravel(), vs.ravel()))
da, db = desk_of(A["cam"], pix), desk_of(B["cam"], pix)
keep = np.min(np.linalg.norm(da[:, None] - np.array([c for c, _ in A["placed"]])[None], axis=2), axis=1) < 200
ma, mb = da[keep].mean(axis=0), db[keep].mean(axis=0)
u, _, vt = np.linalg.svd((da[keep] - ma).T @ (db[keep] - mb))
rot = u @ vt
turn_b = math.atan2(rot[1, 0], rot[0, 0])
cubes_b = [(ma + (c - mb) @ rot.T, t + turn_b) for c, t in B["placed"]]

g = F.COLOURS.index("green")
bl = F.COLOURS.index("blue")
anchor = A["placed"][g][0]
axis_a = (A["placed"][bl][0] - anchor) / np.linalg.norm(A["placed"][bl][0] - anchor)
normal_a = np.array((-axis_a[1], axis_a[0]))
middle = np.array([A["frame"].shape[1] / 2, A["frame"].shape[0] / 2])
spread = F.CENTRE_SPREAD * A["frame"].shape[1]


def unpack_cubes(c):
    """c: 10:31's green turn, blue (along, turn), red (x, y, turn); 17:30's three (x, y, turn)."""
    a = [None] * 3
    a[g] = (anchor, c[0])
    a[bl] = (anchor + c[1] * axis_a, c[2])
    red = [k for k in range(3) if k not in (g, bl)][0]
    a[red] = (np.array(c[3:5]), c[5])
    b = [(np.array(c[6 + 3 * k:8 + 3 * k]), c[8 + 3 * k]) for k in range(3)]
    return a, b


def frame_misses(cam, placed, found):
    return np.concatenate([edges.signed_distances(cam.project(cube_corners(*placed[k])), *found[k]) for k in range(3)])


def camera_of(n, held=None):
    if held is None:
        return Pinhole(n[0], (n[1], n[2]), n[3:6], n[6:9])
    r = cv2.Rodrigues(np.asarray(n[3:6], float))[0]
    return Pinhole(n[0], (n[1], n[2]), n[3:6], -r @ held)


def residuals(p, held=None, which=("1031", "1730")):
    k = 9 if held is None else 6
    cam = camera_of(np.concatenate((p[:k], np.zeros(9 - k))) if held is not None else p[:9], held)
    a, b = unpack_cubes(p[k:])
    parts = []
    if "1031" in which:
        parts.append(frame_misses(cam, a, A["found"]))
    if "1730" in which:
        parts.append(frame_misses(cam, b, B["found"]))
    parts.append((p[1:3] - middle) / spread)
    return np.concatenate(parts)


red = [k for k in range(3) if k not in (g, bl)][0]
cam0 = A["cam"]
cubes0 = np.concatenate(([A["placed"][g][1], (A["placed"][bl][0] - anchor) @ axis_a, A["placed"][bl][1]],
                         A["placed"][red][0], [A["placed"][red][1]],
                         *[(c[0], c[1], t) for c, t in cubes_b]))
cube_steps = np.array([1e-5, 0.01, 1e-5, 0.01, 0.01, 1e-5] + [0.01, 0.01, 1e-5] * 3)
start = np.concatenate(([cam0.focal], cam0.centre, cam0.rotation, cam0.shift, cubes0))
best, _, _ = least_squares(residuals, start, np.concatenate((F.STEPS[:9], cube_steps)), huber=F.EDGE_HUBER_PX,
                           iterations=300)
cam_best = camera_of(best[:9])
a_best, b_best = unpack_cubes(best[9:])


def report(cam, a, b, label):
    ea, eb = np.abs(frame_misses(cam, a, A["found"])), np.abs(frame_misses(cam, b, B["found"]))
    cost = lambda e: float(np.sum(np.where(e <= 1, 0.5 * e ** 2, e - 0.5)))
    br = np.linalg.norm(b[bl][0] - b[red][0])
    print(f"  {label:26} 10:31 {np.median(ea):.2f}/{np.percentile(ea, 90):.2f} px (cost {cost(ea):4.0f})   "
          f"17:30 {np.median(eb):.2f}/{np.percentile(eb, 90):.2f} px (cost {cost(eb):4.0f})   focal {cam.focal:5.0f}, "
          f"lens {cam.position[2]:5.1f} up; 17:30 blue-red {br:5.1f} mm")
    return cost(ea), cost(eb)


print("One camera for both frames, every cube by its edges (typical/90% px):")
report(cam_best, a_best, b_best, "best for both")
axis = cv2.Rodrigues(np.asarray(cam_best.rotation, float))[0][2]          # the line of sight, into the scene
print("\nThe lens held at steps along its line of sight (+ toward the desk), all else fitted again:")
walk = {}
for d in (-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40):
    held = cam_best.position + d * axis
    p0 = np.concatenate((best[:6], best[9:]))
    p, _, _ = least_squares(lambda q: residuals(q, held), p0, np.concatenate((F.STEPS[:6], cube_steps)),
                            huber=F.EDGE_HUBER_PX, iterations=300)
    cam = camera_of(np.concatenate((p[:6], np.zeros(3))), held)
    a, b = unpack_cubes(p[6:])
    walk[d] = (cam, a, b)
    report(cam, a, b, f"{d:+4d} mm along the sight")

for d in (-40, 40, 0):
    cam, _, b = walk[d]
    image = B["frame"].copy()
    for cube in b:
        eod["draw"](image, cam, cube, width=2)
    eod["label"](image, [f"17:30 frame; one camera for both of today's frames, the lens held {d:+d} mm along its line "
                         f"of sight from the best ({cam.position[2]:.0f} mm up, focal {cam.focal:.0f} px)",
                         "every cube placed by its own edges, no arm numbers; 40 mm wireframes"])
    cv2.imwrite(str(OUT / f"walk-1730-{'minus' if d < 0 else 'plus' if d > 0 else 'best'}{abs(d)}.jpg"), image,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
