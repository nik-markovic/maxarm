"""How far the owner's points sit from any smooth map, and from physical arm models. No arm.

Each model maps the arm's x, y onto the camera's true mm. 'From the others' predicts each place from a fit to the
rest (the two places recorded twice are left out together).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import fit_affine, load  # noqa: E402

fit, rows = load()
arm = np.array([(r["x"], r["y"]) for r in rows])
raw = np.array([r["raw"] for r in rows])
cam = np.array([r["cam"] for r in rows])
names = [r["name"] for r in rows]
place = {}
for k, (x, y) in enumerate(arm):
    place.setdefault((round(x, 1), round(y, 1)), []).append(k)
groups = list(place.values())


def poly_terms(p, degree):
    x, y = p[:, 0] / 200, p[:, 1] / 200
    cols = [np.ones(len(p))]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.column_stack(cols)


def fit_poly(src, dst, degree, test=None):
    A = poly_terms(src, degree)
    sol = np.linalg.lstsq(A, dst, rcond=None)[0]
    return (poly_terms(test if test is not None else src, degree) @ sol)


def polar_model(src, params):
    """Firmware x, y -> corrected: bearing gets s1*b + a*sin2b + c*(cos2b-1); reach gets s2*(r-200) + d."""
    s1, a, c, s2, d = params
    r = np.hypot(src[:, 0], src[:, 1])
    b = np.arctan2(src[:, 0], -src[:, 1])
    b2 = b + s1 * b + np.radians(a) * np.sin(2 * b) + np.radians(c) * (np.cos(2 * b) - 1)
    r2 = r + s2 * (r - 200) + d
    return np.column_stack((r2 * np.sin(b2), -r2 * np.cos(b2)))


def fit_polar(src, dst, free, test=None):
    p = np.zeros(5)

    def cost(pp):
        full = p.copy()
        full[free] = pp
        mapped, _ = fit_affine(polar_model(src, full), dst, "rigid")
        return (mapped - dst).ravel()

    x = np.zeros(len(free))
    for _ in range(30):
        r0 = cost(x)
        J = np.empty((len(r0), len(x)))
        for k in range(len(x)):
            dx = np.zeros(len(x))
            dx[k] = 1e-5
            J[:, k] = (cost(x + dx) - r0) / 1e-5
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        x = x + step
        if np.abs(step).max() < 1e-8:
            break
    full = p.copy()
    full[free] = x
    if test is None:
        mapped, _ = fit_affine(polar_model(src, full), dst, "rigid")
        return mapped, full
    # map test points through the same rigid placement
    src_c = polar_model(src, full)
    mapped, (mirror, scale, turn) = fit_affine(src_c, dst, "rigid")
    s = src_c * ([-1, 1] if mirror else [1, 1])
    R = np.array([[math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
                  [math.sin(math.radians(turn)), math.cos(math.radians(turn))]])
    t = (mapped - s @ R.T).mean(0)
    st = polar_model(test, full) * ([-1, 1] if mirror else [1, 1])
    return st @ R.T + t, full


def report(label, predict):
    own = predict(np.arange(len(rows)), np.arange(len(rows)))
    e = np.linalg.norm(own - cam, axis=1)
    loo = np.zeros(len(rows))
    for g in groups:
        rest = np.array([k for k in range(len(rows)) if k not in g])
        out = predict(rest, np.array(g))
        loo[g] = np.linalg.norm(out - cam[g], axis=1)
    print(f"{label:58s} {np.median(e):5.2f} {e.max():5.2f} | from the others {np.median(loo):5.2f} {loo.max():5.2f}")


if __name__ == "__main__":
  print(f"{'model (arm x,y -> camera mm)':58s} typ  worst | from the others typ worst")
  for src_label, src in (("library", arm), ("firmware", raw)):
      for kind in ("rigid", "similarity", "affine"):
          def predict(train, test, kind=kind, src=src):
              mapped, prm = fit_affine(src[train], cam[train], kind)
              # rebuild the map for test points
              if kind == "affine":
                  return np.column_stack((src[test], np.ones(len(test)))) @ prm
              mirror, scale, turn = prm
              s = src[train] * ([-1, 1] if mirror else [1, 1])
              R = np.array([[math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
                            [math.sin(math.radians(turn)), math.cos(math.radians(turn))]]) * scale
              t = (mapped - s @ R.T).mean(0)
              return (src[test] * ([-1, 1] if mirror else [1, 1])) @ R.T + t
          report(f"{src_label}: {kind}", predict)
      for degree in (2, 3):
          report(f"{src_label}: polynomial degree {degree}",
                 lambda train, test, d=degree, src=src: fit_poly(src[train], cam[train], d, src[test]))
  for free, label in (([3, 4], "reach scale+offset"), ([0, 1, 2], "bearing scale+wave"),
                      ([0, 1, 2, 3, 4], "bearing scale+wave, reach scale+offset"), ([1, 2, 4], "wave + reach offset")):
      report(f"firmware, polar: {label}", lambda train, test, f=free: fit_polar(raw[train], cam[train], f, raw[test])[0])
      _, full = fit_polar(raw, cam, free)
      print(f"      params: bearing scale {full[0]*100:+.2f}%, sin2b {full[1]:+.2f} deg, cos2b {full[2]:+.2f} deg, "
            f"reach scale {full[3]*100:+.2f}%, reach offset {full[4]:+.2f} mm")
