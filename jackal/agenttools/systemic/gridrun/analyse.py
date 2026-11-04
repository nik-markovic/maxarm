#!/usr/bin/env python3
"""run.py's points against the owner's 5 mm grid paper: where the cup really was, in the paper's own millimetres,
beside where the arm said it was.

1. The paper: its grid lines found in the bare-paper frames, each line numbered, and the paper's millimetres
   (5 mm a square) tied to the frame's pixels by the line crossings (a homography: the paper is flat). camera.json
   only helps find the lines; the millimetres are the paper's.
2. The cup: its rim (21 mm) fitted to each point's frame through camera.json at the height the arm read, and the
   paper straight under the rim's centre (camera.json's vertical, which the cubes' 40 mm fix) read off the paper.
3. The arm's readings (raw, the firmware's) against those paper positions, with a turn and shift only, then with a
   scale, then as reach and bearing about a base axis: how the arm's reach measures against reality.

    analyse.py [--pictures]        writes analysis.json; --pictures: fits/ for checking by eye
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAMERA = HERE.parents[2] / "config" / "camera.json"
RIM_MM = 10.5
S = 10                                   # rectified px per mm
PITCH = 5.0


def load():
    cam = json.loads(CAMERA.read_text())
    P = np.array(cam["desk_to_pixel"])
    T2D = np.linalg.inv(np.array(cam["desk_to_true"]))
    run = json.loads((HERE / "points.json").read_text())
    return P, T2D, run


def project(P, points):
    h = np.c_[points, np.ones(len(points))] @ P.T
    return h[:, :2] / h[:, 2:]


# --- 1. the paper ---------------------------------------------------------------------------------------------------

def rectify(gray, P, x0, x1, y0, y1):
    X, Y = np.arange(x0, x1, 1 / S), np.arange(y0, y1, 1 / S)
    dx, dy = np.meshgrid(X, Y)
    den = P[2, 0] * dx + P[2, 1] * dy + P[2, 3]
    u = ((P[0, 0] * dx + P[0, 1] * dy + P[0, 3]) / den).astype(np.float32)
    v = ((P[1, 0] * dx + P[1, 1] * dy + P[1, 3]) / den).astype(np.float32)
    return X, Y, cv2.remap(gray, u, v, cv2.INTER_LINEAR, borderValue=0)


def minima(profile, axis):
    """Sub-pixel positions (mm) of the thin dark lines along a profile."""
    p = profile - np.convolve(profile, np.ones(61) / 61, "same")
    found = []
    for i in range(15, len(p) - 15):
        if p[i] == p[i - 15:i + 16].min() and p[i] < -1.5:
            a, b, c = p[i - 1], p[i], p[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            found.append(axis[0] + (i + d) / S)
    return np.array(found)


def line_samples(rect, X, Y, across):
    """(position along the line family's normal, position along the line) of every line sample: `across` 'x' finds
    lines of constant x in thin bands every 4 mm of y."""
    samples = []
    if across == "x":
        for yb in np.arange(Y[0] + 3, Y[-1] - 3, 4.0):
            rows = slice(int((yb - 1.5 - Y[0]) * S), int((yb + 1.5 - Y[0]) * S))
            band = rect[rows]
            ok = (band > 60).mean(0) > 0.9                  # the paper, not the desk or the arm
            prof = np.where(ok, band.mean(0), np.nan)
            for seg in _segments(ok):
                for x in minima(np.nan_to_num(prof[seg], nan=float(np.nanmean(prof[seg]))), X[seg]):
                    samples.append((x, yb))
    else:
        for xb in np.arange(X[0] + 3, X[-1] - 3, 4.0):
            cols = slice(int((xb - 1.5 - X[0]) * S), int((xb + 1.5 - X[0]) * S))
            band = rect[:, cols]
            ok = (band > 60).mean(1) > 0.9
            prof = np.where(ok, band.mean(1), np.nan)
            for seg in _segments(ok):
                for y in minima(np.nan_to_num(prof[seg], nan=float(np.nanmean(prof[seg]))), Y[seg]):
                    samples.append((y, xb))
    return np.array(samples)


def _segments(ok, least=200):
    runs, start = [], None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        if not good and start is not None:
            if i - start >= least:
                runs.append(slice(start, i))
            start = None
    return runs


def number_lines(samples):
    """Each sample's line number: across = a + PITCH k + t along, k whole, fitted and numbered in turn."""
    a, t = samples[0, 0], 0.0
    for _ in range(6):
        k = np.round((samples[:, 0] - a - t * samples[:, 1]) / PITCH)
        A = np.c_[np.ones(len(k)), PITCH * k, samples[:, 1]]
        sol, *_ = np.linalg.lstsq(A, samples[:, 0], rcond=None)
        a, scale, t = sol
        resid = samples[:, 0] - A @ sol
        keep = np.abs(resid) < 0.6
        sol, *_ = np.linalg.lstsq(A[keep], samples[keep, 0], rcond=None)
        a, scale, t = sol
    k = np.round((samples[:, 0] - a - t * samples[:, 1]) / (PITCH * scale))
    resid = samples[:, 0] - (a + PITCH * scale * k + t * samples[:, 1])
    return k.astype(int), resid, scale, a


def paper(P, frames):
    """The homography from paper mm (5 mm squares; columns along desk x, rows along desk y) to frame pixels."""
    X0, X1, Y0, Y1 = -115.0, 115.0, -255.0, -60.0
    pix, mm = [], []
    for name in frames:
        gray = cv2.cvtColor(cv2.imread(str(HERE / "frames" / name)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        X, Y, rect = rectify(gray, P, X0, X1, Y0, Y1)
        xs, ys = line_samples(rect, X, Y, "x"), line_samples(rect, X, Y, "y")
        kx, rx, sx, ax = number_lines(xs)
        ky, ry, sy, ay = number_lines(ys)
        # each sample lies on its line: a pixel with one known paper coordinate
        for (x, y), k, r in zip(xs, kx, rx):
            if abs(r) < 0.4:
                pix.append(project(P, np.array([[x, y, 0.0]]))[0]); mm.append((PITCH * k, np.nan))
        for (y, x), k, r in zip(ys, ky, ry):
            if abs(r) < 0.4:
                pix.append(project(P, np.array([[x, y, 0.0]]))[0]); mm.append((np.nan, PITCH * k))
        print(f"{name}: {len(xs)} column samples on {len(set(kx))} lines (spread {np.std(rx[np.abs(rx) < 0.6]):.2f} mm),"
              f" {len(ys)} row samples on {len(set(ky))} lines ({np.std(ry[np.abs(ry) < 0.6]):.2f} mm);"
              f" squares {PITCH * sx:.3f} x {PITCH * sy:.3f} desk mm")
        break                                # the first frame numbers the lines; others would renumber them
    pix, mm = np.array(pix), np.array(mm)
    # Hi (pixel -> paper mm): a column sample says h1.p = X h3.p, a row sample h2.p = Y h3.p -- linear in Hi
    centre, spread = pix.mean(0), pix.std()
    p = np.c_[(pix - centre) / spread, np.ones(len(pix))]          # pixels conditioned
    rows = []
    for q, (X, Y) in zip(p, mm):
        if not np.isnan(X):
            rows.append(np.r_[q, np.zeros(3), -X * q])
        else:
            rows.append(np.r_[np.zeros(3), q, -Y * q])
    _, _, vt = np.linalg.svd(np.array(rows))
    Hc = vt[-1].reshape(3, 3)
    Hi = Hc @ np.array([[1 / spread, 0, -centre[0] / spread], [0, 1 / spread, -centre[1] / spread], [0, 0, 1]])
    Hi /= Hi[2, 2]
    q = np.c_[pix, np.ones(len(pix))] @ Hi.T
    q = q[:, :2] / q[:, 2:]
    r = np.where(np.isnan(mm[:, 0]), q[:, 1] - np.nan_to_num(mm[:, 1]), q[:, 0] - np.nan_to_num(mm[:, 0]))
    print(f"paper fit: {len(r)} samples, {np.median(np.abs(r)):.3f} mm typically, "
          f"{np.percentile(np.abs(r), 95):.3f} at 95%")
    return Hi, float(np.median(np.abs(r)))


def to_paper(Hi, pixel):
    q = Hi @ np.array([pixel[0], pixel[1], 1.0])
    return q[:2] / q[2]


# --- 2. the cup -------------------------------------------------------------------------------------------------------

def rim_fit(im, P, T2D, x0, y0, z):
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    turns = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ring = (T2D @ np.vstack([RIM_MM * np.cos(turns), RIM_MM * np.sin(turns)])).T
    centre0 = project(P, np.array([[x0, y0, z]]))[0]
    up = project(P, np.array([[x0, y0, z + 10.0]]))[0] - centre0
    up /= np.linalg.norm(up)

    def rim(x, y):
        return project(P, np.c_[ring + (x, y), np.full(len(ring), z)])

    def score(x, y):
        pts = rim(x, y)
        centre = project(P, np.array([[x, y, z]]))[0]
        normal = pts - centre
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        keep = normal @ up < 0.2                       # the side away from the cup's body
        u, v = pts[keep, 0].astype(np.float32).reshape(1, -1), pts[keep, 1].astype(np.float32).reshape(1, -1)
        sx = cv2.remap(gx, u, v, cv2.INTER_LINEAR)[0]
        sy = cv2.remap(gy, u, v, cv2.INTER_LINEAR)[0]
        return float(np.mean(sx * normal[keep, 0] + sy * normal[keep, 1]))

    best = (-1e9, x0, y0)
    for step, span in ((0.5, 7.0), (0.1, 1.0), (0.025, 0.15)):
        cx, cy = best[1], best[2]
        for x in np.arange(cx - span, cx + span + 1e-9, step):
            for y in np.arange(cy - span, cy + span + 1e-9, step):
                s = score(x, y)
                if s > best[0]:
                    best = (s, x, y)
    return best, rim


# --- 3. the arm against the paper -------------------------------------------------------------------------------------

def rigid(A, B, scale=False):
    """B ~ s R A + t, least squares; returns the residuals, s and the turn (deg)."""
    a, b = A - A.mean(0), B - B.mean(0)
    U, sig, Vt = np.linalg.svd(a.T @ b)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    s = (sig.sum() / (a ** 2).sum()) if scale else 1.0
    fitted = s * a @ R.T + B.mean(0)
    return B - fitted, s, math.degrees(math.atan2(R[1, 0], R[0, 0]))


def polar(raw_xy, paper_xy):
    """paper = turn and shift of the arm's point at reach s R + c, bearing b + w1 sin 2b + w2 (cos 2b - 1)."""
    R = np.hypot(raw_xy[:, 0], raw_xy[:, 1])
    b = np.arctan2(raw_xy[:, 0], -raw_xy[:, 1])

    def model(p, with_wave):
        s, c, th, tx, ty, w1, w2 = p
        bb = b + (np.radians(w1 * np.sin(2 * b) + w2 * (np.cos(2 * b) - 1)) if with_wave else 0.0)
        rr = s * R + c
        x, y = rr * np.sin(bb), -rr * np.cos(bb)
        ct, st = math.cos(th), math.sin(th)
        return np.c_[ct * x - st * y + tx, st * x + ct * y + ty]

    out = {}
    for with_wave in (False, True):
        p = np.array([1.0, 0.0, 0.0, *(paper_xy.mean(0) - raw_xy.mean(0)), 0.0, 0.0])
        free = list(range(7 if with_wave else 5))
        for _ in range(50):                                   # Gauss-Newton, numeric Jacobian
            r = (model(p, with_wave) - paper_xy).ravel()
            J = np.empty((len(r), len(free)))
            for j, i in enumerate(free):
                step = np.zeros(7); step[i] = 1e-6
                J[:, j] = ((model(p + step, with_wave) - paper_xy).ravel() - r) / 1e-6
            delta = np.linalg.lstsq(J, -r, rcond=None)[0]
            p[free] += delta
            if np.abs(delta).max() < 1e-9:
                break
        out["wave" if with_wave else "plain"] = (p, model(p, with_wave) - paper_xy)
    return out


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--pictures", action="store_true")
    a = args.parse_args()
    P, T2D, run = load()
    Hi, paper_px = paper(P, run["paper"])
    rows = []
    for point in run["points"]:
        im = cv2.imread(str(HERE / "frames" / point["frame"]))
        dx, dy, dz = point["desk"]
        (score, cx, cy), rim = rim_fit(im, P, T2D, dx, dy, dz)
        under = project(P, np.array([[cx, cy, 0.0]]))[0]
        on_paper = to_paper(Hi, under)
        rows.append({**point, "rim_desk": [round(cx, 3), round(cy, 3)], "rim_score": round(score, 1),
                     "paper": [round(float(on_paper[0]), 3), round(float(on_paper[1]), 3)]})
        if a.pictures:
            (HERE / "fits").mkdir(exist_ok=True)
            c = project(P, np.array([[cx, cy, dz]]))[0].astype(int)
            at = np.array([c[0] - 90, c[1] - 70])
            crop = cv2.resize(im[at[1]:at[1] + 140, at[0]:at[0] + 180], None, fx=4, fy=4,
                              interpolation=cv2.INTER_CUBIC)
            cv2.polylines(crop, [np.round((rim(cx, cy) - at) * 4).astype(np.int32)], True, (0, 140, 255), 2,
                          cv2.LINE_AA)
            cv2.imwrite(str(HERE / "fits" / point["frame"].replace(".jpg", ".png")), crop)
    raw_xy = np.array([r["raw"][:2] for r in rows])
    true_xy = np.array([r["true"][:2] for r in rows])
    paper_xy = np.array([r["paper"] for r in rows])
    result = {"paper_fit_mm": paper_px, "points": rows, "fits": {}}
    for name, arm in (("raw", raw_xy), ("library", true_xy)):
        for scale in (False, True):
            r, s, turn = rigid(arm, paper_xy, scale)
            d = np.hypot(r[:, 0], r[:, 1])
            result["fits"][f"{name}, turn and shift{', scale' if scale else ''}"] = {
                "scale": round(s, 5), "turn_deg": round(turn, 3), "typical_mm": round(float(np.median(d)), 2),
                "worst_mm": round(float(d.max()), 2), "residuals": np.round(r, 2).tolist()}
    for kind, (p, r) in polar(raw_xy, paper_xy).items():
        d = np.hypot(r[:, 0], r[:, 1])
        result["fits"][f"raw, reach scale and offset, {kind}"] = {
            "reach_scale": round(p[0], 5), "reach_offset_mm": round(p[1], 2), "turn_deg": round(math.degrees(p[2]), 3),
            "wave_deg": [round(p[5], 3), round(p[6], 3)], "typical_mm": round(float(np.median(d)), 2),
            "worst_mm": round(float(d.max()), 2), "residuals": np.round(r, 2).tolist()}
    (HERE / "analysis.json").write_text(json.dumps(result, indent=1))
    for name, fit in result["fits"].items():
        print(name, {k: v for k, v in fit.items() if k != "residuals"})


if __name__ == "__main__":
    main()
