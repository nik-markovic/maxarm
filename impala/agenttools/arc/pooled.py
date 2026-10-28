"""Both sessions' points together: one arm error model in the firmware's x, y (shared), each session's camera frame
placed by its own turn and shift (and scale). Each place predicted from a fit without it. No arm.

Arm models (firmware reach r, bearing b, degrees):
  library: today's raw.py correction, nothing fitted
  wave: bearing + s1 sin2b + c1 (cos2b - 1); reach + d
  wave4: wave plus sin4b, cos4b terms on the bearing
  wave4r: wave4 plus the reach error varying with the bearing (sin b, cos b)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import load  # noqa: E402

sessions = []
for cfg in ("/var/b/shared/work/maxarm/work/config-old", "/var/b/shared/work/maxarm/impala/config"):
    fit, rows = load(cfg)
    sessions.append(dict(raw=np.array([r["raw"] for r in rows]), lib=np.array([(r["x"], r["y"]) for r in rows]),
                         cam=np.array([r["cam"] for r in rows]),
                         keys=[(round(r["x"], 1), round(r["y"], 1)) for r in rows]))

TERMS = {
    "wave": ["s2", "c2", "d"],
    "wave4": ["s2", "c2", "s4", "c4", "d"],
    "wave4r": ["s2", "c2", "s4", "c4", "d", "ds1", "dc1"],
    "wave6r": ["s2", "c2", "s4", "c4", "s6", "c6", "d", "ds1", "dc1", "ds2", "dc2"],
}


def corrected(xy, names, p):
    q = dict(zip(names, p))
    r = np.hypot(xy[:, 0], xy[:, 1])
    b = np.arctan2(xy[:, 0], -xy[:, 1])
    db = sum(q.get(f"s{k}", 0) * np.sin(k * b) + q.get(f"c{k}", 0) * (np.cos(k * b) - 1) for k in (2, 4, 6))
    dr = q.get("d", 0) + sum(q.get(f"ds{k}", 0) * np.sin(k * b) + q.get(f"dc{k}", 0) * (np.cos(k * b) - 1)
                             for k in (1, 2))
    b2, r2 = b + np.radians(db), r + dr
    return np.column_stack((r2 * np.sin(b2), -r2 * np.cos(b2)))


def place(src, dst, scale):
    """Similarity (or rigid) with the board's mirror, least squares; returns the mapped points and the map."""
    s = (src * [-1, 1]) @ [1, 1j]
    d = dst @ [1, 1j]
    sm, dm = s.mean(), d.mean()
    a = np.vdot(s - sm, d - dm) / np.vdot(s - sm, s - sm)
    if not scale:
        a = a / abs(a)
    b = dm - a * sm
    f = lambda pts: (lambda z: np.column_stack((z.real, z.imag)))(a * ((pts * [-1, 1]) @ [1, 1j]) + b)
    return f(src), f


def total_residual(names, p, use, scale):
    out = []
    for si, s in enumerate(sessions):
        idx = use[si]
        src = corrected(s["raw"][idx], names, p)
        mapped, _ = place(src, s["cam"][idx], scale)
        out.append((mapped - s["cam"][idx]).ravel())
    return np.concatenate(out)


def fit(names, use, scale, start=None):
    p = np.zeros(len(names)) if start is None else start.copy()
    for _ in range(50):
        r0 = total_residual(names, p, use, scale)
        J = np.column_stack([(total_residual(names, p + d, use, scale) - r0) / 1e-5 for d in np.eye(len(p)) * 1e-5])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-7:
            break
    return p


def evaluate(label, names, scale, library=False):
    every = [np.arange(len(s["raw"])) for s in sessions]
    if library:
        own, loo = [], []
        for s in sessions:
            m, f = place(s["lib"], s["cam"], scale)
            own += list(np.linalg.norm(m - s["cam"], axis=1))
            for key in set(s["keys"]):
                g = [k for k, kk in enumerate(s["keys"]) if kk == key]
                rest = [k for k in range(len(s["keys"])) if k not in g]
                _, f2 = place(s["lib"][rest], s["cam"][rest], scale)
                loo += list(np.linalg.norm(f2(s["lib"][g]) - s["cam"][g], axis=1))
        print(f"{label:46s} fitted typ {np.median(own):.2f} worst {max(own):.2f} | from the others typ "
              f"{np.median(loo):.2f} 90% {np.percentile(loo, 90):.2f} worst {max(loo):.2f}")
        return
    p = fit(names, every, scale)
    own = np.abs(total_residual(names, p, every, scale)).reshape(-1, 2)
    own = np.linalg.norm(own, axis=1)
    loo = []
    for si, s in enumerate(sessions):
        for key in set(s["keys"]):
            g = [k for k, kk in enumerate(s["keys"]) if kk == key]
            use = [e if sj != si else np.array([k for k in e if k not in g]) for sj, e in enumerate(every)]
            p2 = fit(names, use, scale, p)
            src_rest = corrected(s["raw"][use[si]], names, p2)
            _, f = place(src_rest, s["cam"][use[si]], scale)
            loo += list(np.linalg.norm(f(corrected(s["raw"][g], names, p2)) - s["cam"][g], axis=1))
    print(f"{label:46s} fitted typ {np.median(own):.2f} worst {own.max():.2f} | from the others typ "
          f"{np.median(loo):.2f} 90% {np.percentile(loo, 90):.2f} worst {max(loo):.2f}")
    print("      " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, p)))


for scale in (False, True):
    tag = "turn+scale+shift" if scale else "turn+shift"
    evaluate(f"library correction, {tag}", None, scale, library=True)
    for model, names in TERMS.items():
        evaluate(f"{model}, {tag}", names, scale)
