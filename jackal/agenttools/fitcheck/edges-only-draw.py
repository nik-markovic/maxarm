#!/usr/bin/env python3
"""A calibration from the cubes' edges alone -- one square-pixel camera and each 40 mm cube's place and turn fitted
to the edges found in CALIBRATE's frame, no arm numbers -- drawn on the frame as taken: each cube's wireframe through
that camera, the edges it shows solid, those its faces hide faint, an X under its centre. Writes
out/edges-only-<time>.jpg (the whole frame) and out/edges-only-<time>-<colour>.jpg (each cube, 3x).

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/edges-only-draw.py
"""
import itertools, json
from pathlib import Path
import cv2
import numpy as np
from applib.camera import cubes, edges, fit as F
from applib.camera.model import BASE_DIAGONALS, Pinhole, cube_corners, hidden_edges, least_squares, visible_edges

FRAMES = {"1730": Path("config"), "1031": Path("../work/config-jackal-1110-1720")}
OUT = Path("agenttools/fitcheck/out")
YELLOW, SHIFT, ZOOM = (0, 255, 255), 4, 3


def edges_only(frame, points):
    """The camera's numbers and the cubes (true mm, turn) from the edges alone; the edges found; the fit's misses."""
    starts = [(*F._start(frame, n, points), n) for n in cubes.namings(cubes.find_blobs(frame))]
    _, params, naming = max(starts, key=lambda s: s[0])
    params = F._fit_outlines(naming, points, params)
    step3, _, found = F._fit_edges(frame, naming, points, params)
    centres = F._centres(step3, points)
    g, bl, rd = (F.COLOURS.index(c) for c in ("green", "blue", "red"))
    axis = (centres[bl] - centres[g]) / np.linalg.norm(centres[bl] - centres[g])
    normal = np.array((-axis[1], axis[0]))

    def layout(p):
        cs = [None] * 3
        cs[g], cs[bl], cs[rd] = centres[g], centres[g] + p[10] * axis, centres[g] + p[12] * axis + p[13] * normal
        return cs, {g: p[9], bl: p[11], rd: p[14]}

    def residuals(p):
        cam = Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9])
        cs, turns = layout(p)
        parts = [edges.signed_distances(cam.project(cube_corners(cs[k], turns[k])), *found[k]) for k in range(3)]
        parts.append((p[1:3] - np.array([frame.shape[1] / 2, frame.shape[0] / 2])) / (F.CENTRE_SPREAD * frame.shape[1]))
        return np.concatenate(parts)

    rel = centres[rd] - centres[g]
    start = np.concatenate((step3[:9], [step3[F.TURNS][g], np.linalg.norm(centres[bl] - centres[g]),
                                        step3[F.TURNS][bl], rel @ axis, rel @ normal, step3[F.TURNS][rd]]))
    steps = np.concatenate((F.STEPS[:9], [1e-5, 0.01, 1e-5, 0.01, 0.01, 1e-5]))
    p, r, _ = least_squares(residuals, start, steps, huber=F.EDGE_HUBER_PX, iterations=200)
    cam = Pinhole(p[0], (p[1], p[2]), p[3:6], p[6:9])
    cs, turns = layout(p)
    return cam, [(cs[k], turns[k]) for k in range(3)], found, np.abs(r[:-2])


def at(q, origin=(0, 0), zoom=1):
    return (int(round(((q[0] - origin[0] + 0.5) * zoom - 0.5) * (1 << SHIFT))),
            int(round(((q[1] - origin[1] + 0.5) * zoom - 0.5) * (1 << SHIFT))))


def draw(image, cam, cube, origin=(0, 0), zoom=1, width=1):
    centre, turn = cube
    pix = cam.project(cube_corners(centre, turn))
    seen = visible_edges(centre, turn, cam.position)
    faint = image.copy()
    for a, b in [*hidden_edges(seen), *BASE_DIAGONALS]:
        cv2.line(faint, at(pix[a], origin, zoom), at(pix[b], origin, zoom), YELLOW, width, cv2.LINE_AA, SHIFT)
    image[:] = cv2.addWeighted(faint, 0.35, image, 0.65, 0)
    for e in seen:
        cv2.line(image, at(pix[e.start], origin, zoom), at(pix[e.end], origin, zoom), YELLOW, width, cv2.LINE_AA, SHIFT)
    return pix


def label(image, lines, y0=40, scale=1.0):
    for i, text in enumerate(lines):
        y = y0 + int(40 * scale) * i
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


for stamp, config in FRAMES.items():
    frame = cv2.imread(str(config / "camera-frame.png"))
    rec = json.loads((config / "camera-points.json").read_text())["cubes"]
    cam, placed, found, misses = edges_only(frame, {c: (p["x"], p["y"]) for c, p in rec.items()})
    spacing = ", ".join(f"{F.COLOURS[a]}-{F.COLOURS[b]} {np.linalg.norm(placed[a][0] - placed[b][0]):.1f}"
                        for a, b in itertools.combinations(range(3), 2))
    head = [f"{stamp[:2]}:{stamp[2:]} CALIBRATE's frame, the cubes by their edges alone (no arm numbers): 40 mm wireframes",
            f"edges {np.median(misses):.2f} px typically, {np.percentile(misses, 90):.2f} at 90%; focal {cam.focal:.0f} px, "
            f"aim ({cam.centre[0]:.0f}, {cam.centre[1]:.0f}), the lens {cam.position[2]:.0f} mm over the desk",
            f"the cubes' centres apart, mm: {spacing}", "solid: edges the camera sees; faint: hidden ones, X under the centre"]
    whole = frame.copy()
    for cube in placed:
        draw(whole, cam, cube, width=2)
    label(whole, head)
    cv2.imwrite(str(OUT / f"edges-only-{stamp}.jpg"), whole, [cv2.IMWRITE_JPEG_QUALITY, 92])
    for k, cube in enumerate(placed):
        pix = cam.project(cube_corners(*cube))
        lo, hi = pix.min(axis=0) - 40, pix.max(axis=0) + 40
        x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
        x1, y1 = min(int(hi[0]), frame.shape[1]), min(int(hi[1]), frame.shape[0])
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        draw(crop, cam, cube, origin=(x0, y0), zoom=ZOOM, width=1)
        e = np.abs(edges.signed_distances(cam.project(cube_corners(*cube)), *found[k]))
        label(crop, [f"{F.COLOURS[k]}, {stamp[:2]}:{stamp[2:]}, 3x: edges {np.median(e):.2f} px typically, "
                     f"{np.percentile(e, 90):.2f} at 90%"], scale=0.9)
        cv2.imwrite(str(OUT / f"edges-only-{stamp}-{F.COLOURS[k]}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(stamp, head[1], "|", spacing)
