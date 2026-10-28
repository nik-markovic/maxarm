"""Grid paper measured line by line: each of the paper's lines followed along its whole length, through camera.json's
camera alone (no arm). Darkness across a line is averaged along 10 mm of it, so a line's place is found to a
few hundredths of a mm even where the lines are faint.

python paper-lines.py FRAME.png [config-dir]   -> the squares, how straight, even and square the lines are;
                                                   FRAME-lines.npz with every line's places
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
frame_path = Path(sys.argv[1])
config = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "config"
frame = cv2.imread(str(frame_path))
camera = json.loads((config / "camera.json").read_text())
P = np.array(camera["true_to_pixel"])
H = P[:, [0, 1, 3]]                       # true mm on the desk (z = 0) to the frame
H_inv = np.linalg.inv(H)

BAND_MM = 10.0                            # darkness averaged along this much of a line
STEP_MM = 0.04                            # across a line, the profile's step
ALONG_STEP_MM = 0.25


def to_frame(points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ H.T
    return mapped[:, :2] / mapped[:, 2:]


def to_true(pixels):
    mapped = np.column_stack((pixels, np.ones(len(pixels)))) @ H_inv.T
    return mapped[:, :2] / mapped[:, 2:]


def sample(image, pixels):
    count = len(pixels)
    padded = np.zeros((-(-count // 1024) * 1024, 2), np.float32)
    padded[:count] = pixels
    maps = padded.reshape(-1, 1024, 2)
    return cv2.remap(image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderValue=np.nan).ravel()[:count]


# The sheet: the largest white region (bright, colourless), a little in from its edges.
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8)
white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
sheet = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
sheet = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
rows, cols = np.nonzero(sheet)

darkest = frame.min(axis=2).astype(np.float32)
darkness = cv2.GaussianBlur(darkest, (0, 0), 5.0) - cv2.GaussianBlur(darkest, (0, 0), 0.7)
darkness[sheet == 0] = np.nan

outline = to_true(np.column_stack((cols, rows)).astype(float)[::50])
low, high = outline.min(axis=0), outline.max(axis=0)
middle = np.median(outline, axis=0)
print(f"sheet in true mm: x {low[0]:.0f}..{high[0]:.0f}, y {low[1]:.0f}..{high[1]:.0f}, middle {middle.round(1)}")
PX = 8.0
half = 256
gx, gy = np.meshgrid(np.arange(-half, half) / PX, np.arange(-half, half) / PX)
patch = sample(darkness, to_frame(np.column_stack((gx.ravel() + middle[0], gy.ravel() + middle[1])))).reshape(gx.shape)
patch = np.nan_to_num(patch - np.nanmean(patch))
window = np.outer(np.hanning(2 * half), np.hanning(2 * half))
spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch * window)))
fy, fx = np.mgrid[-half:half, -half:half]
radius = np.hypot(fx, fy)
spectrum[(radius < 2 * half / (12 * PX)) | (radius > 2 * half / (3 * PX)) | (fy < 0)] = 0
peak_y, peak_x = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
turn0 = math.atan2(peak_y - half, peak_x - half)
pitch0 = 2 * half / PX / math.hypot(peak_x - half, peak_y - half)
span = float(np.linalg.norm(high - low))


def band_profiles(angle):
    """Darkness across the lines whose normal is `angle`, averaged along each 10 mm band: offsets, {band: profile}."""
    normal = np.array([math.cos(angle), math.sin(angle)])
    along = np.array([-normal[1], normal[0]])
    offsets = np.arange(-span / 2, span / 2, STEP_MM)
    out = {}
    start = -span / 2
    while start + BAND_MM <= span / 2:
        alongs = np.arange(start, start + BAND_MM, ALONG_STEP_MM)
        points = (middle[None, None, :] + offsets[:, None, None] * normal[None, None, :]
                  + alongs[None, :, None] * along[None, None, :])
        values = sample(darkness, to_frame(points.reshape(-1, 2))).reshape(len(offsets), len(alongs))
        seen = np.isfinite(values)
        good = seen.sum(axis=1) >= 0.8 * len(alongs)
        with np.errstate(all="ignore"):
            profile = np.where(good, np.nanmean(np.where(seen, values, np.nan), axis=1), np.nan)
        out[start + BAND_MM / 2] = profile
        start += BAND_MM
    return normal, along, offsets, out


def local_peaks(offsets, profile, pitch):
    """Every dark line in one profile: local maxima at least half a square apart, darker than a third of the band's
    typical line, the centre of darkness round each."""
    reach = int(0.5 * pitch / STEP_MM)
    places = []
    finite = np.isfinite(profile)
    if finite.sum() < 4 * reach:
        return np.array(places)
    tops = [profile[index] for index in range(reach, len(profile) - reach)
            if finite[index - reach:index + reach + 1].all()
            and profile[index] >= profile[index - reach:index + reach + 1].max()]
    if not tops:
        return np.array(places)
    floor = 0.33 * np.percentile(tops, 75)
    for index in range(reach, len(profile) - reach):
        window = profile[index - reach:index + reach + 1]
        if not finite[index - reach:index + reach + 1].all() or profile[index] < window.max() or profile[index] < floor:
            continue
        inner = slice(index - reach // 2, index + reach // 2 + 1)
        values = profile[inner] - np.percentile(window, 20)
        weights = np.clip(values, 0, None)
        places.append(float((weights * offsets[inner]).sum() / weights.sum()))
    return np.array(places)


def track(offsets, bands, pitch):
    """Lines followed band to band from the middle band out: {number: {band: place}}, numbered across the sheet."""
    keys = sorted(bands)
    found = {band: local_peaks(offsets, bands[band], pitch) for band in keys}
    start = max(keys, key=lambda band: len(found[band]))
    lines = {}
    reference = np.sort(found[start])
    spacing = np.median(np.diff(reference))
    for place in reference:
        lines[int(round((place - reference[0]) / spacing))] = {start: place}
    for direction in (1, -1):
        order = [band for band in keys if (band - start) * direction > 0]
        order.sort(key=lambda band: abs(band - start))
        for band in order:
            for number, places in lines.items():
                known = sorted(places, key=lambda b: abs(b - band))[:3]
                if len(known) >= 2:
                    slope = np.polyfit(known, [places[b] for b in known], 1)
                    guess = np.polyval(slope, band)
                else:
                    guess = places[known[0]]
                near = found[band][np.abs(found[band] - guess) < 0.3 * pitch]
                if len(near) == 1:
                    places[band] = float(near[0])
    return lines


def measure(angle):
    normal, along, offsets, bands = band_profiles(angle)
    lines = track(offsets, bands, pitch0)
    return normal, along, lines


families = {}
for name, angle in (("a", turn0), ("b", turn0 + math.pi / 2)):
    for _ in range(3):                     # again along the lines' own direction
        normal, along, lines = measure(angle)
        slopes = [np.polyfit(list(places), list(places.values()), 1)[0] for places in lines.values() if len(places) >= 5]
        angle -= math.atan(float(np.median(slopes)))
    families[name] = (angle, normal, along, lines)
    print(f"family {name}: normal at {math.degrees(angle) % 360:.3f} deg, {len(lines)} lines, "
          f"{sum(len(places) for places in lines.values())} places")

out = {"middle": middle}
for name, (angle, normal, along, lines) in families.items():
    out[f"{name}_angle"] = angle
    out[f"{name}_rows"] = np.array([(number, band, place) for number, places in lines.items()
                                    for band, place in places.items()])
np.savez(frame_path.with_name(frame_path.stem + "-lines.npz"), **out)
print("saved", frame_path.with_name(frame_path.stem + "-lines.npz"))


def fit_family(rows, label, extra=False):
    """Lines evenly spaced and parallel: place = start + k * pitch + slant * along (and, with `extra`, the terms a
    camera turned wrong would add: spacing changing across the sheet and along it, lines spreading). Left over."""
    k, band, place = rows.T
    columns = [np.ones_like(k), k, band] + ([k * k, k * band, band * band] if extra else [])
    design = np.column_stack(columns)
    keep = np.ones(len(place), bool)
    for _ in range(4):
        solution, *_ = np.linalg.lstsq(design[keep], place[keep], rcond=None)
        left = place - design @ solution
        keep = np.abs(left) < max(4 * np.median(np.abs(left[keep])), 0.05)
    return solution, left, keep


import json as _json
summary = {}
for name, (angle, normal, along, lines) in families.items():
    rows = np.array([(number, band, place) for number, places in lines.items() for band, place in places.items()])
    for extra in (False, True):
        solution, left, keep = fit_family(rows, name, extra)
        tag = "with spreading terms" if extra else "straight, parallel, even"
        print(f"family {name} {tag}: pitch {solution[1]:.4f} mm, slant {math.degrees(math.atan(solution[2])):+.3f} deg;"
              f" left over median {np.median(np.abs(left[keep])):.3f} mm, 95% {np.percentile(np.abs(left[keep]), 95):.3f},"
              f" max {np.abs(left[keep]).max():.3f}; {keep.sum()} of {len(keep)} places, lines {int(rows[:, 0].min())}..{int(rows[:, 0].max())}")
        if extra:
            k, band = rows[keep, 0], rows[keep, 1]
            terms = np.column_stack((k * k, k * band, band * band)) @ solution[3:]
            print(f"   the spreading terms move lines by up to {np.abs(terms - terms.mean()).max():.3f} mm across the measured part")
    summary[name] = {"angle_deg": math.degrees(angle), "pitch": float(fit_family(rows, name)[0][1]),
                     "slant": float(fit_family(rows, name)[0][2])}
a, b = summary["a"], summary["b"]
between = (b["angle_deg"] + math.degrees(math.atan(b["slant"]))) - (a["angle_deg"] + math.degrees(math.atan(a["slant"])))
print(f"squares: {a['pitch']:.4f} x {b['pitch']:.4f} mm; the two directions {between % 180:.3f} deg apart")
