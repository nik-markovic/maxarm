"""The red cube in each frame given, as FIND RED CUBE finds it through camera.json, and how far the frame's own edges
sit from the wireframe drawn there, edge by edge: outward (the cube looks bigger) or inward, and towards the camera
(down the frame) or away. Writes a close-up of the cube, 3 pixels a pixel, the wireframe over it, beside each frame.
No arm.

    ../.venv/bin/python agenttools/live-res/cube-outline.py /tmp/formats/yuyv-avg5.png /tmp/formats/mjpg-avg5.png ...
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import cubes, edges, store  # noqa: E402
from camera.locate import find_red_cube  # noqa: E402

ZOOM = 3
fit = store.load_camera(ROOT / "config")
P = fit.desk_to_pixel


def to_pixels(points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ P.T
    return mapped[:, :2] / mapped[:, 2:]


rows = []
for name in sys.argv[1:]:
    frame = cv2.imread(name)
    found = find_red_cube(frame, fit)
    corners = to_pixels(found.corners)
    blobs = [blob for blob in cubes.find_blobs(frame, is_warm_only=True) if cubes.is_warm(blob)]
    blob = max(blobs, key=lambda b: b.size)
    crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
    starts, ends, targets = edges.search(crop, corners, found.seen, 0.02)
    middle = corners.mean(axis=0)
    # Each edge's offset: outward for an outline (away from the cube's middle), down the frame for an inner edge.
    per_edge, outward, toward = [], [], []
    px_per_mm = np.mean([np.linalg.norm(corners[4 + i] - corners[4 + (i + 1) % 4]) for i in range(4)]) / 40.0
    for edge in found.seen:
        mine = (starts == edge.start) & (ends == edge.end)
        if not mine.any():
            continue
        a, b = corners[edge.start], corners[edge.end]
        normal = np.array([a[1] - b[1], b[0] - a[0]]) / np.linalg.norm(b - a)
        if edge.is_outline:
            normal = normal if normal @ ((a + b) / 2 - middle) > 0 else -normal
        else:
            normal = normal if normal[1] > 0 else -normal
        offset = float(np.median((targets[mine] - a) @ normal))
        per_edge.append((edge.start, edge.end, edge.is_outline, offset, int(mine.sum())))
        if edge.is_outline:
            outward.append(offset)
            toward.append(offset * normal[1])          # how much of it points down the frame, to the camera
    rows.append((Path(name).stem, found, px_per_mm, per_edge, outward))
    # The close-up.
    x0, y0 = np.floor(corners.min(axis=0) - 25).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0) + 25).astype(int)
    close = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    drawn = ((corners - [x0, y0]) * ZOOM + (ZOOM - 1) / 2) * 16
    for start, end in [(e.start, e.end) for e in found.seen]:
        cv2.line(close, tuple(np.round(drawn[start]).astype(int)), tuple(np.round(drawn[end]).astype(int)),
                 (255, 255, 0), 1, cv2.LINE_AA, shift=4)
    cv2.putText(close, f"{Path(name).stem}: wireframe = the 40 mm cube where FIND RED CUBE puts it", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(close, f"{Path(name).stem}: wireframe = the 40 mm cube where FIND RED CUBE puts it", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(Path(name).with_name(f"cube-{Path(name).stem}.png")), close)

for stem, found, px_per_mm, per_edge, outward in rows:
    edges_text = "  ".join(f"{s}-{e}{'o' if o else 'i'} {off:+.2f}" for s, e, o, off, n in per_edge)
    print(f"{stem:12s} desk ({found.x:7.2f}, {found.y:7.2f}) edges {found.edge_px:.2f} px; {px_per_mm:.2f} px/mm; "
          f"outline outward median {np.median(outward):+.2f} px = {np.median(outward) / px_per_mm:+.2f} mm a side")
    print(f"             per edge (o outline, i inner; px): {edges_text}")
