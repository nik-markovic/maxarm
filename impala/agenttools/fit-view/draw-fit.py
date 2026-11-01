"""How the arm is placed on its points: the picture. For the saved calibration (config/, only read): each point where
the arm said the cup was, and where the camera saw the cube, through the camera fitted to the cubes; placed on the
three cubes alone (CALIBRATE) and on the cubes and the Fine Tune points together (SAVE CALIBRATION); and how far
off the points are on average as the placement is nudged away from the best. No arm, no camera.

    ../.venv/bin/python agenttools/fit-view/draw-fit.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import refine, store  # noqa: E402

OUT = Path(__file__).resolve().parent / "fit-points.png"
MAGNIFY = 20
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
ARM, SEEN = "#2a78d6", "#eb6834"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


config = ROOT / "config"
saved = store.load_camera(config)
alone = saved.cube_fit()
points = store.load_refine_points(config)
cubes = refine.cube_sightings(saved)
names = list(saved.cubes)
sightings = cubes + [point.sighting for point in points]
labels = names + [str(point.id) for point in points]
together = refine.refine(saved, [point.sighting for point in points])
arm = np.array([(s.x, s.y) for s in sightings])


def seen(fit):
    return np.array([refine.seen_desk(fit, s) for s in sightings])


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


placements = [("CALIBRATE: the arm placed on the 3 cubes alone", alone, 3),
              ("SAVE CALIBRATION: placed on the 3 cubes and the 15 points", together, len(sightings))]

image = Image.new("RGB", (1930, 1540), SURFACE)
draw = ImageDraw.Draw(image)
draw.text((30, 22), "How the arm is placed on its points: the \"best\" turn and shift", fill=INK, font=font(30, True))
draw.text((30, 66), "Blue: where the arm said the cup was (its servos).  Orange: where the camera saw that cube's centre. "
                    f"Each blue-to-orange offset drawn {MAGNIFY}x longer.  Seen from above; desk x grows to the left.",
          fill=INK2, font=font(18))

# --- the two desk panels --------------------------------------------------
SCALE, U0, U1, Y0, Y1 = 2.0, -235.0, 205.0, -285.0, 40.0
panel_w, panel_h = int((U1 - U0) * SCALE), int((Y1 - Y0) * SCALE)


def to_screen(origin, xy):
    return origin[0] + (-xy[0] - U0) * SCALE, origin[1] + (Y1 - xy[1]) * SCALE


for index, (title, fit, used) in enumerate(placements):
    origin = (30 + index * (panel_w + 80), 196)
    draw.text((origin[0], origin[1] - 74), title, fill=INK, font=font(19, True))
    draw.rectangle([origin[0], origin[1], origin[0] + panel_w, origin[1] + panel_h], outline=GRID, width=1)
    for value in range(-200, 201, 50):
        a, b = to_screen(origin, (value, Y0)), to_screen(origin, (value, Y1))
        if U0 <= -value <= U1:
            draw.line([a, b], fill=GRID, width=1)
            draw.text((a[0] - 12, origin[1] + panel_h + 4), f"{value}", fill=MUTED, font=font(13))
    for value in range(-250, 31, 50):
        a, b = to_screen(origin, (-U0, value)), to_screen(origin, (-U1, value))
        draw.line([a, b], fill=GRID, width=1)
        draw.text((origin[0] - 2 + panel_w + 6, a[1] - 8), f"{value}", fill=MUTED, font=font(13))
    draw.text((origin[0] + panel_w / 2 - 40, origin[1] + panel_h + 22), "desk x, mm", fill=MUTED, font=font(13))
    base = to_screen(origin, (0, 0))
    draw.line([(base[0] - 9, base[1]), (base[0] + 9, base[1])], fill=INK, width=2)
    draw.line([(base[0], base[1] - 9), (base[0], base[1] + 9)], fill=INK, width=2)
    draw.text((base[0] + 10, base[1] - 22), "arm's base (0, 0)", fill=INK2, font=font(14))
    camera = saved.camera_position
    where = to_screen(origin, camera[:2])
    draw.text((min(max(where[0], origin[0] + 4), origin[0] + panel_w - 120), min(where[1], origin[1] + panel_h - 22)),
              f"camera ({camera[0]:.0f}, {camera[1]:.0f}),\n{camera[2]:.0f} mm up", fill=INK2, font=font(13))
    there = seen(fit)
    offs = np.linalg.norm(there - arm, axis=1)
    for k, (start, end, label) in enumerate(zip(arm, arm + MAGNIFY * (there - arm), labels)):
        a, b = to_screen(origin, start), to_screen(origin, end)
        taking_part = k < used
        draw.line([a, b], fill=SEEN, width=2)
        draw.ellipse([b[0] - 5, b[1] - 5, b[0] + 5, b[1] + 5], fill=SEEN, outline=SURFACE, width=2)
        if k < 3:
            draw.rectangle([a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7], fill=ARM if taking_part else SURFACE,
                           outline=ARM, width=2)
        else:
            draw.ellipse([a[0] - 6, a[1] - 6, a[0] + 6, a[1] + 6], fill=ARM if taking_part else SURFACE,
                         outline=ARM, width=2)
        away = np.array(a) - np.array(b)
        away = away / (np.linalg.norm(away) or 1.0)
        text = f"{label} {offs[k]:.1f}"
        box = draw.textbbox((0, 0), text, font=font(13))
        # On the far side of the blue mark from the offset; to its left when the offset points right.
        tx = a[0] + away[0] * 14 - (box[2] if away[0] < -0.3 else box[2] / 2 if abs(away[0]) <= 0.3 else 0)
        ty = a[1] + away[1] * 14 - box[3] / 2
        draw.text((tx, ty), text, fill=INK, font=font(13))
    # The legend for the magnified offsets: what 1 mm looks like.
    bar = (origin[0], origin[1] - 30)
    draw.line([bar, (bar[0] + MAGNIFY * SCALE, bar[1])], fill=SEEN, width=3)
    draw.text((bar[0] + MAGNIFY * SCALE + 8, bar[1] - 9), "= 1 mm off", fill=INK2, font=font(14))
    took = ("filled: taken into the placement; hollow: not (Fine Tune points, shown against it)" if used == 3
            else "all 18 filled: all taken into the placement, each counting the same")
    draw.text((bar[0] + MAGNIFY * SCALE + 110, bar[1] - 8), took, fill=INK2, font=font(14))
    draw.text((origin[0], origin[1] + panel_h + 46),
              f"off, all 18: {rms(offs):.2f} mm rms;  the 15 points: {rms(offs[3:]):.2f} mm rms, worst "
              f"{offs[3:].max():.2f}", fill=INK, font=font(15, True))
    draw.text((origin[0], origin[1] + panel_h + 70),
              f"the 3 cubes: {', '.join(f'{n} {o:.2f}' for n, o in zip(names, offs[:3]))} mm", fill=INK, font=font(15))

# --- what "best" means: the average offset as the placement is nudged --------
there = seen(together)
moved = arm - there.mean(axis=0)


def score(turn_deg, dx, dy):
    """rms of the 18 offsets with the arm's points turned about the base and shifted, the camera's left as fitted."""
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    placed = arm @ np.array([[c, s], [-s, c]]) + (dx, dy)
    return rms(np.linalg.norm(placed - there, axis=1))


row = 196 + panel_h + 170
draw.text((30, row - 44), "What \"best\" means: SAVE's placement, nudged.  Each curve: the 18 offsets' rms (mm) as one of the "
                          "three numbers is moved off its best value, the other two held.", fill=INK, font=font(19, True))
charts = [("turn about the arm's base, degrees", np.linspace(-1.0, 1.0, 201), lambda v: score(v, 0, 0)),
          ("shift along desk x, mm", np.linspace(-3.0, 3.0, 201), lambda v: score(0, v, 0)),
          ("shift along desk y, mm", np.linspace(-3.0, 3.0, 201), lambda v: score(0, 0, v))]
chart_w, chart_h, low, high = 500, 250, 1.0, 4.0
for index, (name, values, measure) in enumerate(charts):
    left, top = 70 + index * (chart_w + 110), row
    costs = np.array([measure(v) for v in values])
    sx = lambda v: left + (v - values[0]) / (values[-1] - values[0]) * chart_w  # noqa: E731
    sy = lambda c: top + chart_h - (c - low) / (high - low) * chart_h  # noqa: E731
    for level in np.arange(low, high + 0.01, 0.5):
        draw.line([(left, sy(level)), (left + chart_w, sy(level))], fill=GRID, width=1)
        draw.text((left - 34, sy(level) - 8), f"{level:.1f}", fill=MUTED, font=font(13))
    for tick in np.linspace(values[0], values[-1], 5):
        draw.line([(sx(tick), top + chart_h), (sx(tick), top + chart_h + 5)], fill=MUTED, width=1)
        draw.text((sx(tick) - 14, top + chart_h + 8), f"{tick:+.1f}" if tick else "0", fill=MUTED, font=font(13))
    draw.line([(left, top + chart_h), (left + chart_w, top + chart_h)], fill=MUTED, width=1)
    draw.line([(sx(v), sy(c)) for v, c in zip(values, costs)], fill=ARM, width=2, joint="curve")
    best = int(np.argmin(costs))
    bx, by = sx(values[best]), sy(costs[best])
    draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=ARM, outline=SURFACE, width=2)
    draw.text((bx + 8, by - 26), f"best {costs[best]:.2f} mm, at {values[best]:+.2f}", fill=INK, font=font(14))
    draw.text((left, top + chart_h + 30), name, fill=INK2, font=font(15))

# --- the rule, in words ------------------------------------------------------
note = ("The camera stays as the cubes' shapes fitted it. Through it each point's pixel is a place on the desk (orange).\n"
        "The arm's points (blue) are moved as one rigid set: one turn about the base and one shift, 3 numbers for all.\n"
        "Best = the smallest sum of the squared blue-to-orange distances, solved in one step (no search). Every point\n"
        "counts the same, cubes and Fine Tune points alike, and no point is set aside. The distances between the arm's\n"
        "points are never changed: no stretch, so what is left over is each point's own disagreement.")
draw.multiline_text((30, row + chart_h + 70), note, fill=INK, font=font(17), spacing=6)
image.save(OUT)
print("written", OUT)

# The numbers behind the picture: how much of what is left lies along the line from the base, and how much across.
offsets = there - arm
radial = np.array([o @ (a / np.linalg.norm(a)) for o, a in zip(offsets, arm)])
print(f"SAVE leaves: rms {rms(np.linalg.norm(offsets, axis=1)):.2f} mm; along the line from the base: mean "
      f"{radial.mean():+.2f}, rms {rms(radial):.2f}; mean offset ({offsets[:, 0].mean():+.2f}, {offsets[:, 1].mean():+.2f})")
for label, a, o, r in zip(labels, arm, offsets, radial):
    print(f"  {label:6s} arm ({a[0]:7.1f}, {a[1]:7.1f}) reach {np.linalg.norm(a):5.1f}: camera - arm "
          f"({o[0]:+.2f}, {o[1]:+.2f}), outward {r:+.2f}")
