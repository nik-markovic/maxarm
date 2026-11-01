#!/usr/bin/env python3
"""The misses on CALIBRATE's frame: at each Fine Tune point an arrow to where the cup comes down when sent to the
cube there (the miss SAVE leaves, x20), today's points and 10/30 noon's; the grid paper of 10:55 (lens test); the
frame's middle column; the arm's base and its bearings. Writes twist.jpg beside this file.

    .venv/bin/python impala/agenttools/twist/draw.py
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCALE = 20
PAPER = [(2253, 1458), (1958, 556), (767, 609), (589, 1477)]       # lens/straight.py on now-quality.png


def main():
    camera = json.loads((ROOT / "config" / "camera.json").read_text())
    to_pixel = np.array(camera["desk_to_pixel"])
    frame = cv2.imread(str(ROOT / "config" / "camera-frame.png"))
    frame = (frame * 0.55).astype(np.uint8)

    def px(x, y):
        p = to_pixel @ (x, y, 0.0, 1.0)
        return int(round(p[0] / p[2])), int(round(p[1] / p[2]))

    cv2.polylines(frame, [np.array(PAPER, np.int32)], True, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "grid paper 10:55", (PAPER[2][0] + 10, PAPER[2][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (200, 200, 200), 2, cv2.LINE_AA)
    cv2.line(frame, (1152, 0), (1152, 1535), (160, 160, 160), 1, cv2.LINE_AA)
    base = px(0, 0)
    for bearing in (-60, -30, 0, 30, 60):
        b = math.radians(bearing)
        end = px(280 * math.sin(b), -280 * math.cos(b))
        cv2.line(frame, px(60 * math.sin(b), -60 * math.cos(b)), end, (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{bearing:+d}", end, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 255), 2, cv2.LINE_AA)
    for reach in (150, 200, 250):
        ring = [px(reach * math.sin(math.radians(b)), -reach * math.cos(math.radians(b))) for b in range(-70, 71, 5)]
        cv2.polylines(frame, [np.array(ring, np.int32)], False, (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{reach}", ring[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 255), 2, cv2.LINE_AA)
    for name, colour in (("old1030", (0, 165, 255)), ("now", (80, 255, 80))):
        for row in json.loads((HERE / f"twist-{name}.json").read_text())["rows"]:
            x, y = row["arm"]
            miss = np.array(row["miss"])
            start, end = px(x, y), px(x + SCALE * miss[0], y + SCALE * miss[1])
            cv2.circle(frame, start, 7, colour, 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, start, end, colour, 3, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(frame, f"{row['id']} {math.hypot(*miss):.1f}", (start[0] + 10, start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    legend = ["arrow: where the cup comes down when sent to the cube there, x20 (the miss SAVE leaves), mm",
              "green: today 15:55-16:38 (18 points)   orange: 10/30 noon (20 points)",
              "red lines: the arm's bearings and reaches   grey: the frame's middle column, the 10:55 grid paper"]
    for index, text in enumerate(legend):
        cv2.putText(frame, text, (20, 40 + 40 * index), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(HERE / "twist.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(HERE / "twist.jpg", "base at", base)


if __name__ == "__main__":
    main()
