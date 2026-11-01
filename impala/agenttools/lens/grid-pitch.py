#!/usr/bin/env python3
"""The sheet's printed grid through camera.json: how far apart its lines are, in the camera's millimetres, along the
sheet's two sides. The sheet as sheet-size.py finds it, looked at from straight above (each sheet millimetre taken
to its pixel through camera.json), the lines' spacing from how they repeat across the whole sheet.

    .venv/bin/python impala/agenttools/lens/grid-pitch.py FRAME.png [OUT.png]
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[1] / "config"
spec = importlib.util.spec_from_file_location("size", HERE / "sheet-size.py")
size = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size)
straight = size.straight
PX_PER_MM = 8.0
MARGIN_MM = 6.0


def pitch(profile, mm_per_px):
    """The spacing that repeats best along `profile`: the FFT's strongest peak, refined over the whole length."""
    signal = profile - np.convolve(profile, np.ones(81) / 81, mode="same")
    signal = signal[60:-60]
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=16 * len(signal)))
    freqs = np.fft.rfftfreq(16 * len(signal), d=mm_per_px)
    usable = (freqs > 1 / 15.0) & (freqs < 1 / 2.0)              # spacings 2-15 mm
    peak = np.argmax(np.where(usable, spectrum, 0))
    best = freqs[peak]
    trials = np.linspace(0.97 * best, 1.03 * best, 3001)          # the phase-coherent fit over the whole sheet
    positions = np.arange(len(signal)) * mm_per_px
    power = [abs(np.sum(signal * np.exp(-2j * np.pi * f * positions))) for f in trials]
    return 1 / trials[int(np.argmax(power))]


def main():
    frame = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matrix = np.array(json.loads((CONFIG / "camera.json").read_text())["desk_to_pixel"])
    plane = matrix[:, [0, 1, 3]]
    corners = straight.paper_corners(gray)
    lines = [size.line_through(straight.border_points(gray, corners[i], corners[(i + 1) % 4])) for i in range(4)]
    pixels = [size.crossing(lines[i - 1], lines[i]) for i in range(4)]
    desk = []
    for pixel in pixels:
        x, y, w = np.linalg.solve(plane, (*pixel, 1.0))
        desk.append(np.array([x / w, y / w]))
    origin, along_end, across_end = desk[1], desk[0], desk[2]          # the sheet's right-hand corner, its two sides
    u_len, v_len = np.linalg.norm(along_end - origin), np.linalg.norm(across_end - origin)
    u, v = (along_end - origin) / u_len, (across_end - origin) / v_len
    us = np.arange(MARGIN_MM, u_len - MARGIN_MM, 1 / PX_PER_MM)
    vs = np.arange(MARGIN_MM, v_len - MARGIN_MM, 1 / PX_PER_MM)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    points = origin + grid_u[..., None] * u + grid_v[..., None] * v
    homogeneous = np.concatenate((points, np.ones(points.shape[:2] + (1,))), axis=2) @ plane.T
    map_x = (homogeneous[..., 0] / homogeneous[..., 2]).astype(np.float32)
    map_y = (homogeneous[..., 1] / homogeneous[..., 2]).astype(np.float32)
    top_down = cv2.remap(gray.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)       # rows along u
    along_u, along_v = top_down.mean(axis=1), top_down.mean(axis=0)
    print(f"sheet sides through the camera: {u_len:.2f} (u) x {v_len:.2f} (v) mm")
    for name, profile, length in (("u", along_u, u_len), ("v", along_v, v_len)):
        spacing = pitch(profile, 1 / PX_PER_MM)
        print(f"  along {name}: lines every {spacing:.4f} mm (camera); {length / spacing:.2f} spacings across "
              f"the whole side; against 5 mm {100 * (spacing / 5 - 1):+.2f}%, 1/4 inch {100 * (spacing / 6.35 - 1):+.2f}%, "
              f"1/5 inch {100 * (spacing / 5.08 - 1):+.2f}%")
    print("  each ninth of the sheet (rows: from the corner at origin along v; columns along u), mm a square, u / v:")
    for row in range(3):
        cells = []
        for column in range(3):
            part = top_down[column * len(us) // 3:(column + 1) * len(us) // 3, row * len(vs) // 3:(row + 1) * len(vs) // 3]
            middle_px = homogeneous[(2 * column + 1) * len(us) // 6, (2 * row + 1) * len(vs) // 6]
            cells.append(f"{pitch(part.mean(axis=1), 1 / PX_PER_MM):.3f} / {pitch(part.mean(axis=0), 1 / PX_PER_MM):.3f}"
                         f" (px {middle_px[0] / middle_px[2]:.0f},{middle_px[1] / middle_px[2]:.0f})")
        print("    " + "   ".join(cells))
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], np.clip(top_down, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    main()
