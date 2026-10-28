"""Do the misses come from the gap between the cup and the cube's top when the cube was centred by eye? No arm.

A cube centred by eye under a cup held `gap` mm over its top, seen from an eye at E, ends up displaced away from
the eye by (P - E_xy) * gap / (E_z - cup height). Fitted: the 4-number map plus one eye position for every point;
then each place predicted from the others.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from points import load  # noqa: E402

CUBE_MM = 40.0


def similarity_fit(src, dst):
    s = (src * [-1, 1]) @ [1, 1j]
    d = dst @ [1, 1j]
    (a, b), *_ = np.linalg.lstsq(np.column_stack((s, np.ones(len(s)))), d, rcond=None)
    return a, b


def apply_sim(a, b, src):
    z = a * ((src * [-1, 1]) @ [1, 1j]) + b
    return np.column_stack((z.real, z.imag))


def with_eye(arm, z, eye):
    return arm + (arm - eye[:2]) * ((z - CUBE_MM) / (eye[2] - z))[:, None]


def fit_eye(arm, z, cam, starts):
    best = None
    for e0 in starts:
        e = np.array(e0, float)
        for _ in range(60):
            def res(ee):
                moved = with_eye(arm, z, ee)
                return (apply_sim(*similarity_fit(moved, cam), moved) - cam).ravel()
            r0 = res(e)
            J = np.column_stack([(res(e + d) - r0) / 1e-3 for d in np.eye(3) * 1e-3])
            step = np.clip(np.linalg.lstsq(J, -r0, rcond=None)[0], -100, 100)
            e = e + step
            e[2] = max(e[2], 80.0)
            if np.abs(step).max() < 1e-4:
                break
        cost = (res(e) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, e)
    return best[1]


STARTS = [(x, y, h) for x in (-400, 0, 400) for y in (-900, -500) for h in (250, 450, 700)]
for cfg in sys.argv[1:]:
    fit, rows = load(cfg)
    arm = np.array([(r['x'], r['y']) for r in rows])
    z = np.array([r['z'] for r in rows])
    cam = np.array([r['cam'] for r in rows])
    places = {}
    for k, p in enumerate(arm):
        places.setdefault((round(p[0], 1), round(p[1], 1)), []).append(k)
    eye = fit_eye(arm, z, cam, STARTS)
    loo_plain, loo_eye = np.zeros(len(rows)), np.zeros(len(rows))
    for g in places.values():
        rest = [k for k in range(len(rows)) if k not in g]
        loo_plain[g] = np.linalg.norm(apply_sim(*similarity_fit(arm[rest], cam[rest]), arm[g]) - cam[g], axis=1)
        e_ = fit_eye(arm[rest], z[rest], cam[rest], [tuple(eye)])
        moved = with_eye(arm[rest], z[rest], e_)
        loo_eye[g] = np.linalg.norm(apply_sim(*similarity_fit(moved, cam[rest]), with_eye(arm[g], z[g], e_)) - cam[g],
                                    axis=1)
    print(f"{cfg}: gaps {np.round(z - CUBE_MM, 1)} mm")
    print(f"   from the others: 4 numbers typ {np.median(loo_plain):.2f}; with an eye typ {np.median(loo_eye):.2f}, "
          f"the eye at desk ({eye[0]:.0f}, {eye[1]:.0f}), {eye[2]:.0f} mm up")
