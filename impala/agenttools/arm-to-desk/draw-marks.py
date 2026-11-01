#!/usr/bin/env python3
"""ARM TO DESK's marks as the tests page's live view draws them (record-marks.mjs), phase by phase, painted on the
calibration's frame through config/camera.json: the arm sent, landed, read, and sent elsewhere; the arm's reading
made up, a little off the point and a lot off it.

    .venv/bin/python impala/agenttools/arm-to-desk/draw-marks.py [x y]     # desk mm, default 60 -180

Writes phases.png beside this file: each phase close up, the reading near (top row) and far (bottom row).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from camera import store                                                   # noqa: E402

VIEW = (1440, 1080)                                     # the full view on a large window, canvas pixels
LIMITS = {"reach": 270, "reach_min": 95, "fan": 120,
          "tests": {"touch": {"cup_radius": 10.5, "down": 2, "over": 14}, "find": {"cube": 40, "down": 42},
                    "refine": {"off": [1, 2]}}}
CLOSE_PX = 70                                           # half the close-up's side, canvas pixels


def bgr(ink):
    ink = ink.lstrip("#")
    if ink.startswith("rgba"):
        return (255, 255, 255)
    return tuple(int(ink[i:i + 2], 16) for i in (4, 2, 0))


def paint(image, records):
    for record in records:
        layer = image.copy()
        alpha = record["alpha"]
        if record["kind"] == "text":
            cv2.putText(layer, record["text"], (int(record["at"][0]), int(record["at"][1] + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr(record["ink"]), 1, cv2.LINE_AA)
        for path in record.get("paths", []):
            points = np.round(np.array(path) * 4).astype(np.int32)
            if record["kind"] == "stroke":
                cv2.polylines(layer, [points], False, bgr(record["ink"]), max(1, round(record["width"])),
                              cv2.LINE_AA, shift=2)
            elif len(path) > 2:
                cv2.fillPoly(layer, [points], bgr(record["ink"]), cv2.LINE_AA, shift=2)
        cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, dst=image)
    return image


def main():
    x, y = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (60.0, -180.0)
    fit = store.load_camera(IMPALA / "config")
    frame = cv2.resize(store.load_frame(IMPALA / "config"), VIEW, interpolation=cv2.INTER_AREA)
    shares = fit.desk_to_shares
    seen = shares @ (x, y, 0.0, 1.0)
    u, v = seen[:2] / seen[2]
    rows = []
    for name, says in (("read 0.4 mm off, z 2.2", [x + 0.3, y - 0.3, 2.2]),
                       ("read 3 mm off, z 6", [x + 2.5, y - 1.7, 6.0])):
        scene = {"matrix": shares.tolist(), "deskToTrue": fit.desk_to_true.tolist(), "size": VIEW, "limits": LIMITS,
                 "touch": {"u": u, "v": v, "desk": [x, y], "status": "done"}, "says": says,
                 "phases": [{"phase": "sent", "target": [x, y, 14.0]}, {"phase": "landed", "target": [x, y, 2.0]},
                            {"phase": "read", "target": [x, y, 2.0]}, {"phase": "left", "target": [x + 40, y, 60.0]}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(scene, handle)
        recorded = json.loads(subprocess.run(
            ["node", "--experimental-default-type=module", str(HERE / "record-marks.mjs"), handle.name],
            check=True, capture_output=True, text=True).stdout)
        Path(handle.name).unlink()
        centre = np.array([u * VIEW[0], v * VIEW[1]]).round().astype(int)
        cells = []
        for phase, records in recorded.items():
            painted = paint(frame.copy(), records)
            left, top = centre - CLOSE_PX
            cell = cv2.resize(painted[top:top + 2 * CLOSE_PX, left:left + 2 * CLOSE_PX], None, fx=3.4, fy=3.4,
                              interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(cell, (0, 0), (cell.shape[1], 26), (20, 20, 20), -1)
            cv2.putText(cell, f"{phase} ({name})" if phase == "read" else phase, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)
            inks = sorted({record["ink"] for record in records if record["kind"] != "text"})
            print(f"{phase:7s} {name}: {len(records)} drawn, inks {inks}")
        rows.append(np.hstack(cells))
    cv2.imwrite(str(HERE / "phases.png"), np.vstack(rows))
    print(f"desk ({x}, {y}) is at {u:.4f}, {v:.4f} of the view; wrote {HERE / 'phases.png'}")


if __name__ == "__main__":
    main()
