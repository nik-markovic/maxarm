#!/usr/bin/env python3
"""Where the arm's map and the cubes' own map (edges alone) disagree, by direction. Both of today's CALIBRATE layouts
are two cubes out to the sides (~250 mm out, ~31 deg each side) and one on the centre line (~120 mm out).
  across: the two side cubes apart (left-right in the picture, across the centre line)
  along:  the centre-line cube to the line through the side ones (along the centre line, up-down in the picture)
  the two diagonals, and the angle at the centre-line cube (no scale in it: the triangle's shape alone)
The arm's readings made true without and with the old turn fix; the cubes by their edges alone (40 mm cubes, no arm
numbers): each frame's own fit, the one camera for both frames, and that camera's lens walked along its line of sight
with everything else fitted again (the edges' fit shown at each step).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/spans.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np

src = Path("agenttools/fitcheck/walk-camera.py").read_text().split("\n\ndef report(")[0]
ns = {}
exec(compile(src, "walk-camera", "exec"), ns)
F, least_squares, data = ns["F"], ns["least_squares"], ns["data"]
COLOURS = list(F.COLOURS)
TURN = (-1.36, 0.33)                          # the old turn fix: true bearing = b + TURN[0] sin 2b + TURN[1] (cos 2b - 1)
READ_WITH = {"1031": TURN, "1730": (0.0, 0.0)}                     # what each frame's readings were made true with
CONFIG = {"1031": Path("../work/config-jackal-1110-1720"), "1730": Path("config")}


def polar_back(x, y, turn):
    """Reach and the firmware's bearing of a reading made true with `turn`."""
    r, bt = math.hypot(x, y), math.atan2(x, -y)
    b = bt
    for _ in range(30):
        b = bt - math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return r, b


def made_true(r, b, turn):
    bt = b + math.radians(turn[0] * math.sin(2 * b) + turn[1] * (math.cos(2 * b) - 1))
    return np.array((r * math.sin(bt), -r * math.cos(bt)))


def shape(pts, centre):
    s1, s2 = [c for c in COLOURS if c != centre]
    p, a, b = (np.asarray(pts[c], float) for c in (centre, s1, s2))
    base, rel = b - a, p - a
    across = float(np.linalg.norm(base))
    along = abs(base[0] * rel[1] - base[1] * rel[0]) / across
    d1, d2 = float(np.linalg.norm(a - p)), float(np.linalg.norm(b - p))
    angle = math.degrees(math.acos(float(np.dot(a - p, b - p)) / (d1 * d2)))
    return across, along, d1, d2, angle


def edges_px(cam, placed, stamp):
    e = np.abs(ns["frame_misses"](cam, placed, data[stamp]["found"]))
    return f"{np.median(e):.2f}/{np.percentile(e, 90):5.2f}"


variants = {"1031": [], "1730": []}
for stamp in variants:
    own = data[stamp]
    variants[stamp].append(("cubes: this frame's own fit", own["placed"], edges_px(own["cam"], own["placed"], stamp),
                            own["cam"]))
cam_best, a_best, b_best = ns["cam_best"], ns["a_best"], ns["b_best"]
variants["1031"].append(("cubes: one camera, both frames", a_best, edges_px(cam_best, a_best, "1031"), cam_best))
variants["1730"].append(("cubes: one camera, both frames", b_best, edges_px(cam_best, b_best, "1730"), cam_best))
axis = cv2.Rodrigues(np.asarray(cam_best.rotation, float))[0][2]
for d in (-40, -20, -10, 10, 20, 40):
    held = cam_best.position + d * axis
    p0 = np.concatenate((ns["best"][:6], ns["best"][9:]))
    p, _, _ = least_squares(lambda q: ns["residuals"](q, held), p0, np.concatenate((F.STEPS[:6], ns["cube_steps"])),
                            huber=F.EDGE_HUBER_PX, iterations=300)
    cam = ns["camera_of"](np.concatenate((p[:6], np.zeros(3))), held)
    a, b = ns["unpack_cubes"](p[6:])
    variants["1031"].append((f"  lens {d:+d} mm along its sight", a, edges_px(cam, a, "1031"), cam))
    variants["1730"].append((f"  lens {d:+d} mm along its sight", b, edges_px(cam, b, "1730"), cam))

for stamp in ("1730", "1031"):
    rec = json.loads((CONFIG[stamp] / "camera-points.json").read_text())["cubes"]
    centre = min(rec, key=lambda c: abs(rec[c]["x"]))
    s1, s2 = [c for c in COLOURS if c != centre]
    polar = {c: polar_back(p["x"], p["y"], READ_WITH[stamp]) for c, p in rec.items()}
    print(f"\n{stamp[:2]}:{stamp[2:]}: {s1} and {s2} out to the sides, {centre} on the centre line "
          f"(firmware bearings {', '.join(f'{c} {math.degrees(polar[c][1]):+.2f}' for c in COLOURS)} deg)")
    print(f"  {'':34} {'across':>7} {'along':>7} {centre[0]}-{s1[0]:<5} {centre[0]}-{s2[0]:<5} {'angle':>6}"
          f"   edges px (typ/90%)  lens up, focal")
    for name, turn in (("arm, no turn fix", (0.0, 0.0)), ("arm, old turn fix", TURN)):
        pts = {c: made_true(*polar[c], turn) for c in rec}
        print(f"  {name:34} {shape(pts, centre)[0]:7.1f} {shape(pts, centre)[1]:7.1f} {shape(pts, centre)[2]:7.1f} "
              f"{shape(pts, centre)[3]:7.1f} {shape(pts, centre)[4]:6.1f}")
    for name, placed, e, cam in variants[stamp]:
        s = shape({c: placed[k][0] for k, c in enumerate(COLOURS)}, centre)
        print(f"  {name:34} {s[0]:7.1f} {s[1]:7.1f} {s[2]:7.1f} {s[3]:7.1f} {s[4]:6.1f}   {e:>18}  "
              f"{cam.position[2]:5.1f}, {cam.focal:4.0f}")
