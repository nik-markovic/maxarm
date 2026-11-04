#!/usr/bin/env python3
"""The arm's error fitted from the points, against taking the library's x, y as given.

The camera is SAVE's, from the cube pictures alone; each point's cube is where it puts it (true mm). The arm side
starts from the firmware's own reading, recovered from the stored x, y with the library as it was when the point was
recorded (reach + 6.6 - 0.118 min(h, 44), the turning wave). Each model lays the arm on the cubes; each point is then
predicted from the others. mm, typically / worst.

    ../.venv/bin/python agenttools/systemic/arm-model-fit.py config_dir
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import store                                                     # noqa: E402
from applib.camera.fit import link_arm                                              # noqa: E402
from camera import refine                                                           # noqa: E402

WAVE_SIN, WAVE_COS = -1.36, 0.33


def wave(bearing):
    return math.radians(WAVE_SIN * math.sin(2 * bearing) + WAVE_COS * (math.cos(2 * bearing) - 1))


def old_raw_xy(x, y, height):
    """The firmware's x, y for a stored point, by the library as it was."""
    reach, bearing = math.hypot(x, y), math.atan2(x, -y)
    raw_bearing = bearing
    for _ in range(8):
        raw_bearing = bearing - wave(raw_bearing)
    reach -= 6.6 - 0.118 * min(max(height, 0.0), 44.0)
    return reach * math.sin(raw_bearing), -reach * math.cos(raw_bearing)


config = Path(sys.argv[1])
fit = store.load_camera(config)
points = [p.sighting for p in store.load_refine_points(config, fit)]
heights = [p.z for p in store.load_refine_points(config, fit)]
plane = json.loads((config / "desk.json").read_text())["desk_to_arm"][2] if (config / "desk.json").exists() else None
seen = np.array([refine._seen(refine.in_frame(fit.true_to_pixel, fit.frame_size, p), p) for p in points])
firmware = []
for p, z in zip(points, heights):
    arm_z = z + (plane[0] * p.x + plane[1] * p.y + plane[3] if plane else 44.0)
    firmware.append(old_raw_xy(p.x, p.y, arm_z - 44.0))
firmware = np.array(firmware)


def polar_to_true(params, rows, with_wave, with_scale):
    turn, sx, sy, offset = params[:4]
    rest = list(params[4:])
    scale = 1.0 + rest.pop(0) if with_scale else 1.0
    w1, w2 = (rest.pop(0), rest.pop(0)) if with_wave else (0.0, 0.0)
    out = []
    for x, y in rows:
        r, b = math.hypot(x, y), math.atan2(x, -y)
        r = scale * r + offset
        b = b + math.radians(w1 * math.sin(2 * b) + w2 * (math.cos(2 * b) - 1))
        px, py = -r * math.sin(b), -r * math.cos(b)                  # the board's mirror: x negated
        c, s = math.cos(turn), math.sin(turn)
        out.append((c * px - s * py + sx, s * px + c * py + sy))
    return np.array(out)


def fit_polar(rows, targets, with_wave, with_scale):
    params = np.zeros(4 + with_scale + 2 * with_wave)
    a = polar_to_true(params, rows, with_wave, with_scale)
    params[1:3] = (targets - a).mean(0)
    for _ in range(50):
        r = (polar_to_true(params, rows, with_wave, with_scale) - targets).ravel()
        J = np.empty((len(r), len(params)))
        for i in range(len(params)):
            q = params.copy()
            q[i] += 1e-6
            J[:, i] = ((polar_to_true(q, rows, with_wave, with_scale) - targets).ravel() - r) / 1e-6
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        params += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return params


def library_xy(rows):
    """The library now: reach + 1.6, the turning wave."""
    out = []
    for x, y in rows:
        r, b = math.hypot(x, y), math.atan2(x, -y)
        r, b = r + 1.6, b + wave(b)
        out.append((r * math.sin(b), -r * math.cos(b)))
    return np.array(out)


def stretch_model(arm_rows):
    def train(rows, targets):
        return link_arm(rows, targets)

    def predict(model, rows):
        turn, shift, desk_to_true = model
        return (np.asarray(rows) @ desk_to_true.T) @ turn.T + shift
    return train, predict


def polar_model(with_wave, with_scale):
    def train(rows, targets):
        return fit_polar(rows, targets, with_wave, with_scale)

    def predict(model, rows):
        return polar_to_true(model, rows, with_wave, with_scale)
    return train, predict


def score(name, rows, model, free):
    train, predict = model
    whole = np.linalg.norm(predict(train(rows, seen), rows) - seen, axis=1)
    left = []
    for i in range(len(rows)):
        keep = [k for k in range(len(rows)) if k != i]
        m = train(rows[keep], seen[keep])
        left.append(np.linalg.norm(predict(m, rows[i:i + 1])[0] - seen[i]))
    print(f"   {name:58s} {free} numbers: fitted {np.median(whole):.2f} / {whole.max():.2f}   "
          f"each from the others {np.median(left):.2f} / {max(left):.2f}")
    return train(rows, seen)


print(f"{config}: {len(points)} points")
score("library's x, y (reach + 1.6, wave) + turn, shift, stretch (SAVE)", library_xy(firmware), stretch_model(None), 6)
score("firmware's x, y + turn, shift, stretch", firmware, stretch_model(None), 6)
score("firmware's: turn, shift, reach offset", firmware, polar_model(False, False), 4)
score("firmware's: turn, shift, reach offset and scale", firmware, polar_model(False, True), 5)
m = score("firmware's: turn, shift, reach offset, turning wave", firmware, polar_model(True, False), 6)
print(f"      reach + {m[3]:.2f} mm; wave {m[4]:+.2f} sin 2b {m[5]:+.2f} (cos 2b - 1) deg  (gecko: +1.6 at the notch; "
      f"{WAVE_SIN:+.2f}, {WAVE_COS:+.2f})")
m = score("firmware's: turn, shift, reach offset and scale, turning wave", firmware, polar_model(True, True), 7)
print(f"      reach x {1 + m[4]:.4f} + {m[3]:.2f} mm; wave {m[5]:+.2f} sin 2b {m[6]:+.2f} (cos 2b - 1) deg")

# Where the two disagree: the same firmware reading, laid by SAVE's way and by the arm's own error fitted.
save_train, save_predict = stretch_model(None)
save = save_train(library_xy(firmware), seen)
arm_train, arm_predict = polar_model(True, True)
arm = arm_train(firmware, seen)
covered = [math.degrees(math.atan2(x, -y)) for x, y in firmware]
print(f"   the points cover bearings {min(covered):+.0f} to {max(covered):+.0f}; the two apart, mm (rows: out from the base):")
print("   reach  " + " ".join(f"{b:>5d}" for b in range(-100, 101, 20)))
for reach in (110, 150, 190, 230, 260):
    rows = np.array([(reach * math.sin(math.radians(b)), -reach * math.cos(math.radians(b))) for b in range(-100, 101, 20)])
    apart = np.linalg.norm(save_predict(save, library_xy(rows)) - arm_predict(arm, rows), axis=1)
    print(f"   {reach:5d}  " + " ".join(f"{d:5.1f}" for d in apart))
