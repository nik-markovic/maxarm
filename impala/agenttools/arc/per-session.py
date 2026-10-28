"""Per session: each place predicted from the others, by the map camera.json could hold: 3 numbers (turn, shift),
4 (and a scale), 6 (any stretch). And the first calibration's three cubes alone, the same three ways. No arm."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import load  # noqa: E402


def mapper(src, dst, kind):
    if kind == "6 numbers":
        A = np.column_stack((src, np.ones(len(src))))
        sol = np.linalg.lstsq(A, dst, rcond=None)[0]
        return lambda p: np.column_stack((p, np.ones(len(p)))) @ sol
    s = (src * [-1, 1]) @ [1, 1j]
    d = dst @ [1, 1j]
    sm, dm = s.mean(), d.mean()
    a = np.vdot(s - sm, d - dm) / np.vdot(s - sm, s - sm)
    if kind == "3 numbers":
        a /= abs(a)
    b = dm - a * sm
    return lambda p: (lambda z: np.column_stack((z.real, z.imag)))(a * ((p * [-1, 1]) @ [1, 1j]) + b)


KINDS = ("3 numbers", "4 numbers", "6 numbers")
for cfg in sys.argv[1:]:
    fit, rows = load(cfg)
    arm = np.array([(r["x"], r["y"]) for r in rows])
    cam = np.array([r["cam"] for r in rows])
    keys = [(round(x, 1), round(y, 1)) for x, y in arm]
    print(cfg)
    for kind in KINDS:
        loo = []
        for key in set(keys):
            g = [k for k, kk in enumerate(keys) if kk == key]
            rest = [k for k in range(len(keys)) if k not in g]
            loo += list(np.linalg.norm(mapper(arm[rest], cam[rest], kind)(arm[g]) - cam[g], axis=1))
        whole = mapper(arm, cam, kind)
        own = np.linalg.norm(whole(arm) - cam, axis=1)
        three = mapper(arm[:3], cam[:3], kind)
        later = [k for k in range(3, len(rows)) if keys[k] not in keys[:3]]
        e3 = np.linalg.norm(three(arm[later]) - cam[later], axis=1)
        print(f"   {kind}: all points fitted typ {np.median(own):.2f} worst {own.max():.2f}; each from the others typ "
              f"{np.median(loo):.2f} worst {max(loo):.2f} | three cubes alone -> later points typ {np.median(e3):.2f} "
              f"worst {e3.max():.2f}")
