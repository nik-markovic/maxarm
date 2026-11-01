#!/usr/bin/env python3
"""The owner's grid paper (5 mm squares, flat on the desk) through camera.json: does the calibration's desk (z 0) lie
on the paper's printed grid all over the sheet? No arm.

    .venv/bin/python impala/agenttools/cube-grid/paper.py frame.png

The sheet is looked at from straight above through camera.json (each desk mm taken to its frame pixel), its printed
lines found in strips 10 mm wide, both ways. A perfect 5 mm grid is laid on them (its place and turn free, and its
pitch, to see the calibration's scale). Prints the misses by region and the squares' size along the sheet; writes
lines.json (each line point: family, line number, desk mm as the calibration sees it, frame pixel).
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PX_PER_MM = 8.0
STRIP_MM = 10.0
SQUARE_MM = 5.0


def line_positions(profile, mm_per_px):
    """The dark lines' places along `profile`, mm, each to a fraction of a sample (agenttools/lens/grid-lines.py)."""
    signal = profile - np.convolve(profile, np.ones(81) / 81, mode="same")
    places = []
    strong = np.abs(signal[45:-45]).max() if len(signal) > 90 else 0
    for index in range(45, len(signal) - 45):
        window = signal[index - 12:index + 13]
        if signal[index] == window.min() and signal[index] < -0.25 * strong:
            before, here, after = signal[index - 1:index + 2]
            shift = 0.5 * (before - after) / (before - 2 * here + after)
            places.append((index + shift) * mm_per_px)
    return np.array(places)


def main():
    frame = cv2.imread(str(HERE / sys.argv[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    which = sys.argv[2] if len(sys.argv) > 2 else "desk_to_pixel"         # or true_to_pixel: the camera's own mm
    matrix = np.array(json.loads((ROOT / "config" / "camera.json").read_text())[which])
    plane = matrix[:, [0, 1, 3]]

    def to_pixel(desk):                                   # (..., 2) desk mm at z 0 -> (..., 2) pixels
        h = np.concatenate((desk, np.ones(desk.shape[:-1] + (1,))), axis=-1) @ plane.T
        return h[..., :2] / h[..., 2:]

    # the sheet: the largest bright patch of the desk seen from above, coarse
    xs, ys = np.arange(-400, 400, 1.0), np.arange(-450, 100, 1.0)
    mesh = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    pix = to_pixel(mesh).astype(np.float32)
    coarse = cv2.remap(gray, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderValue=0)
    bright = (coarse > np.percentile(coarse[coarse > 0], 80)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contour = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    centre = np.array([xs[0] + cx, ys[0] + cy])
    turn = math.radians(angle)
    u = np.array([math.cos(turn), math.sin(turn)])
    v = np.array([-u[1], u[0]])
    print(f"sheet seen {w:.0f} x {h:.0f} mm (in view), centre ({centre[0]:.0f}, {centre[1]:.0f}), turn {angle:.1f} deg")
    # fine top-down view along the sheet's sides
    us = np.arange(-w / 2 + 4, w / 2 - 4, 1 / PX_PER_MM)
    vs = np.arange(-h / 2 + 4, h / 2 - 4, 1 / PX_PER_MM)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    desk = centre + grid_u[..., None] * u + grid_v[..., None] * v
    pix = to_pixel(desk).astype(np.float32)
    inside = (pix[..., 0] > 2) & (pix[..., 0] < width - 3) & (pix[..., 1] > 2) & (pix[..., 1] < height - 3)
    top = cv2.remap(gray, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderValue=0)
    paper = inside & (top > np.percentile(top[inside], 30))
    found = []                                            # family, along-mm (line's place), across-mm (strip middle)
    strip = int(STRIP_MM * PX_PER_MM)
    for family, (along, across, image, mask) in enumerate(((us, vs, top, paper), (vs, us, top.T, paper.T))):
        for first in range(0, len(across) - strip + 1, strip):
            block = image[:, first:first + strip]
            ok = mask[:, first:first + strip].all(axis=1)
            if ok.sum() < 200:
                continue
            # the longest run of rows wholly on the paper
            runs, start = [], None
            for index, good in enumerate(np.append(ok, False)):
                if good and start is None:
                    start = index
                elif not good and start is not None:
                    runs.append((start, index))
                    start = None
            a, b = max(runs, key=lambda run: run[1] - run[0])
            if b - a < 200:
                continue
            places = line_positions(block[a:b].mean(axis=1), 1 / PX_PER_MM) + along[a]
            middle = across[first:first + strip].mean()
            found += [(family, place, middle) for place in places]
    found = np.array(found)
    # each line point back to desk mm (as the calibration has it) and to the frame
    sheet = np.where(found[:, :1] == 0, found[:, [1, 2]], found[:, [2, 1]])        # (u, v) on the sheet
    desk_pts = centre + sheet[:, :1] * u + sheet[:, 1:2] * v
    pixels = to_pixel(desk_pts)
    # a perfect grid: family 0 lines at u' = s0 + p n, family 1 at v' = s1 + p n, (u', v') the sheet turned by a
    # little; solved by rounding line numbers, then least squares on shift, turn, pitch
    family = found[:, 0].astype(int)
    p, s, small = SQUARE_MM, np.zeros(2), 0.0
    for _ in range(10):
        rot = np.array([[1, -small], [small, 1]])
        turned = sheet @ rot.T
        coord = np.where(family == 0, turned[:, 0], turned[:, 1])
        number = np.round((coord - s[family]) / p)
        # linear in (s0, s1, p, small): coord + small * other - s_f - p n = 0
        other = np.where(family == 0, -sheet[:, 1], sheet[:, 0])
        rows = np.column_stack(((family == 0) * 1.0, (family == 1) * 1.0, number, -other))
        target = np.where(family == 0, sheet[:, 0], sheet[:, 1])
        solution = np.linalg.lstsq(rows, target, rcond=None)[0]
        s, p, small = solution[:2], solution[2], solution[3]
    miss = target - rows @ solution
    print(f"{len(found)} line points; the calibration's squares {p:.4f} mm on average; misses from that grid "
          f"rms {math.sqrt(np.mean(miss ** 2)):.3f} mm, 95% within {np.percentile(np.abs(miss), 95):.3f}")
    # local square size: spacing of neighbouring lines in each strip, by where it is
    local = []
    for f in (0, 1):
        for middle in np.unique(found[family == f, 2]):
            sel = (family == f) & (found[:, 2] == middle)
            places = np.sort(found[sel, 1])
            gaps = np.diff(places)
            for place, gap in zip(places[:-1], gaps):
                if 0.6 * SQUARE_MM < gap < 1.4 * SQUARE_MM:
                    mid = (place + gap / 2, middle) if f == 0 else (middle, place + gap / 2)
                    local.append((f, *mid, gap))
    local = np.array(local)
    json.dump({"centre": centre.tolist(), "u": u.tolist(), "v": v.tolist(), "pitch": p,
               "points": [{"family": int(f), "sheet": list(map(float, sh)), "desk": list(map(float, d)),
                           "pixel": list(map(float, px)), "miss": float(m)}
                          for f, sh, d, px, m in zip(family, sheet, desk_pts, pixels, miss)],
               "local": local.tolist()}, open(HERE / "lines.json", "w"))
    # by region of the frame: thirds across and down
    # the squares' size along the sheet, by how far down the frame (toward the camera)
    lp = to_pixel(centre + local[:, 1:2] * u + local[:, 2:3] * v)
    print("the squares' size by how far down the frame (v px): along u (lines of family 0) / along v (family 1)")
    for lo in range(500, 1536, 200):
        sel = (lp[:, 1] >= lo) & (lp[:, 1] < lo + 200)
        cells = [f"{np.median(local[sel & (local[:, 0] == f), 3]):.3f} ({np.sum(sel & (local[:, 0] == f)):3d})"
                 if np.any(sel & (local[:, 0] == f)) else "   -        " for f in (0, 1)]
        print(f"   v {lo:4d}-{lo + 200:4d}: {cells[0]} / {cells[1]}")
    for f in (0, 1):
        sel = local[:, 0] == f
        coef = np.polyfit(lp[sel, 1], local[sel, 3], 1)
        print(f"   family {f}: size = {coef[1]:.3f} + {coef[0] * 1000:.4f} per 1000 px down")
    print("misses by where in the frame (rms mm, points) and the squares' size there (mm, both ways):")
    for row in range(3):
        cells = []
        for col in range(3):
            sel = ((pixels[:, 0] // (width / 3)) == col) & ((pixels[:, 1] // (height / 3)) == row)
            lp = to_pixel(centre + local[:, 1:2] * u + local[:, 2:3] * v)
            lsel = ((lp[:, 0] // (width / 3)) == col) & ((lp[:, 1] // (height / 3)) == row)
            if sel.sum() < 10:
                cells.append("        -        ")
                continue
            sizes = [np.median(local[lsel & (local[:, 0] == f), 3]) if np.any(lsel & (local[:, 0] == f)) else np.nan
                     for f in (0, 1)]
            cells.append(f"{math.sqrt(np.mean(miss[sel] ** 2)):.2f} ({sel.sum():4d}) {sizes[0]:.3f}/{sizes[1]:.3f}")
        print("   " + " | ".join(cells))


if __name__ == "__main__":
    main()
