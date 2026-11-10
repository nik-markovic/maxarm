#!/usr/bin/env python3
"""The arm's millimetres made another way: every reading taken back to the firmware's own (raw.py), then made true
again with other reach and bearing terms, and the same fits run: today's point 4 from the three cubes, the 9-10 Nov
green from blue, red and the 10 Nov point, each point left out of 4 Nov's twelve. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/arm-mm.py
"""
import dataclasses, json, math
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine
from maxarm import raw

# reach on the desk, less per mm of height, bearing sin 2b, cos 2b - 1 (deg)
MODELS = {"library now": (6.26, 0.110, -1.36, 0.33), "no bearing wave": (6.26, 0.110, 0.0, 0.0),
          "impala's fitted wave": (6.26, 0.110, -1.96, 0.23), "hyena's fitted wave": (6.26, 0.110, -1.41, 0.65),
          "firmware alone": (0.0, 0.0, 0.0, 0.0)}


def remade(p, model):
    rx, ry, _ = raw.raw_position((p.x, p.y, p.z + raw.DESK_Z_MM))
    out, less, s2, c2 = model
    h = max(p.z, 0.0)
    r, b = math.hypot(rx, ry), math.atan2(rx, -ry)
    r += out - less * min(h, 44.0)
    b += math.radians(s2 * math.sin(2 * b) + c2 * (math.cos(2 * b) - 1))
    return r * math.sin(b), -r * math.cos(b)


def load(camera_file, points_file):
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = json.loads(Path(points_file).read_text())["points"]
    return fit, stored


def sightings(stored, model):
    out = []
    for p in stored:
        rp = RefinePoint(**p)
        x, y = remade(rp, model)
        out.append(dataclasses.replace(rp.sighting, x=x, y=y))
    return out


def camera(fit, chosen, start):
    on_arm = refine._OnArm(fit.frame_size, chosen, start)
    n = least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=200, derivatives=on_arm.derivatives)[0]
    return refine._pinhole(n).matrix


def miss(m, s):
    return float(np.linalg.norm(HANDEDNESS @ refine._seen(m, s) - (s.x, s.y)))


today = load("/tmp/c1110/camera.json", "/tmp/c1110/camera-refine-points.json")
nov9 = load("/tmp/c1110/camera.json", "/tmp/c1110/camera-refine-points-previous.json")
nov4 = load("../work/config-jackal-1104/camera.json", "../work/config-jackal-1104/camera-refine-points.json")
for name, model in MODELS.items():
    fit, st = today
    s = sightings(st, model)
    first = fit.cube_fit().true_to_pixel
    m = camera(fit, s[:3], first)
    t4 = miss(m, s[3])
    fit9, st9 = nov9
    s9 = sightings(st9, model)
    g = miss(camera(fit9, [s9[0], s9[2], s9[3]], first), s9[1])
    fit4, st4 = nov4
    s4 = sightings(st4, model)
    f4 = fit4.cube_fit().true_to_pixel
    every = camera(fit4, s4, f4)
    lo = [miss(camera(fit4, [o for j, o in enumerate(s4) if j != k], every), s4[k]) for k in range(len(s4))]
    print(f"{name:22}: today point 4 {t4:4.2f} | 9 Nov green {g:4.2f} | 4 Nov twelve left out median "
          f"{np.median(lo):4.2f}, worst {max(lo):4.2f}")
