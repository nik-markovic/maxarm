#!/usr/bin/env python3
"""Where the camera is, from the cubes' edges alone (edges-only-draw.py's fit, no arm numbers), and how sure: each
frame's own spread (the fit's covariance, from its edge scatter), and the two frames against each other -- 10:31
and 17:30, the same camera unmoved (0.6 px on the bare desk), different cube layouts. The two fits' desks are laid
on each other by where they put the same pixels (a turn and shift on the desk); then the two lens positions, their
heights, tilts and focal lengths, and how far apart the two desk maps put the same pixel over the cubes' area.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/camera-place.py
"""
import importlib.util, json, math
from pathlib import Path
import cv2
import numpy as np
from applib.camera import edges, fit as F
from applib.camera.model import Pinhole, cube_corners, least_squares

spec = importlib.util.spec_from_file_location("eod", "agenttools/fitcheck/edges-only-draw.py")
src = Path("agenttools/fitcheck/edges-only-draw.py").read_text().split("\nfor stamp, config in FRAMES.items():")[0]
eod = {}
exec(compile(src, "edges-only-draw", "exec"), eod)
FRAMES = eod["FRAMES"]


def tilt_deg(cam):
    """How far the camera's axis is from straight down."""
    axis = cv2.Rodrigues(np.asarray(cam.rotation, float))[0][2]       # the optical axis in true mm
    return math.degrees(math.acos(abs(axis[2])))


def desk_of(cam, pixels):
    m = cam.matrix
    h = np.column_stack((m[:, 0], m[:, 1], m[:, 3]))
    q = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(h).T
    return q[:, :2] / q[:, 2:]


fits = {}
for stamp, config in FRAMES.items():
    frame = cv2.imread(str(config / "camera-frame.png"))
    rec = json.loads((config / "camera-points.json").read_text())["cubes"]
    cam, placed, found, misses = eod["edges_only"](frame, {c: (p["x"], p["y"]) for c, p in rec.items()})
    # each frame's own spread: the whole fit's covariance (camera and cubes together, the desk's origin and direction
    # held as the fit holds them), from the edges' scatter; 400 draws from it
    g, bl, rd = (F.COLOURS.index(c) for c in ("green", "blue", "red"))
    cg = placed[g][0]
    axis = (placed[bl][0] - cg) / np.linalg.norm(placed[bl][0] - cg)
    normal = np.array((-axis[1], axis[0]))
    rel = placed[rd][0] - cg
    p0 = np.concatenate(([cam.focal], cam.centre, cam.rotation, cam.shift,
                         [placed[g][1], np.linalg.norm(placed[bl][0] - cg), placed[bl][1], rel @ axis, rel @ normal,
                          placed[rd][1]]))

    def unpack(n):
        c = Pinhole(n[0], (n[1], n[2]), n[3:6], n[6:9])
        cubes_ = {g: (cg, n[9]), bl: (cg + n[10] * axis, n[11]), rd: (cg + n[12] * axis + n[13] * normal, n[14])}
        return c, cubes_

    def res(n):
        c, cubes_ = unpack(n)
        parts = [edges.signed_distances(c.project(cube_corners(*cubes_[k])), *found[k]) for k in range(3)]
        parts.append((n[1:3] - np.array([frame.shape[1] / 2, frame.shape[0] / 2])) / (F.CENTRE_SPREAD * frame.shape[1]))
        return np.concatenate(parts)

    steps = np.concatenate((F.STEPS[:9], [1e-5, 0.01, 1e-5, 0.01, 0.01, 1e-5]))
    r0 = res(p0)
    jac = np.column_stack([(res(p0 + np.eye(len(p0))[i] * steps[i]) - r0) / steps[i] for i in range(len(p0))])
    w = np.where(np.abs(r0) <= F.EDGE_HUBER_PX, 1.0, F.EDGE_HUBER_PX / np.maximum(np.abs(r0), 1e-12))
    sigma2 = float(np.sum(w * r0[:-2] ** 2 if False else w * r0 ** 2) / (len(r0) - len(p0)))
    cov = sigma2 * np.linalg.pinv(jac.T @ (jac * w[:, None]))
    draws = np.random.default_rng(1).multivariate_normal(p0, cov, 400)
    positions = np.array([unpack(d)[0].position for d in draws])
    tilts = np.array([tilt_deg(unpack(d)[0]) for d in draws])
    pos = cam.position
    print(f"{stamp[:2]}:{stamp[2:]}: lens at ({pos[0]:.1f}, {pos[1]:.1f}) mm across, {pos[2]:.1f} mm up "
          f"(+- {positions[:, 0].std():.1f}, {positions[:, 1].std():.1f}, {positions[:, 2].std():.1f} from the edges' "
          f"scatter); axis {tilt_deg(cam):.2f} deg from straight down (+- {tilts.std():.2f}); focal {cam.focal:.0f} px "
          f"(+- {math.sqrt(cov[0, 0]):.0f}), aim ({cam.centre[0]:.0f}, {cam.centre[1]:.0f})")
    fits[stamp] = (cam, placed, frame.shape)

(a, (cam_a, placed_a, shape)), (b, (cam_b, placed_b, _)) = list(fits.items())[1], list(fits.items())[0]
# pixels over the desk where the cubes were, both frames: a grid in the picture, kept where the desk is near them
us, vs = np.meshgrid(np.linspace(150, shape[1] - 150, 40), np.linspace(150, shape[0] - 150, 24))
pixels = np.column_stack((us.ravel(), vs.ravel()))
da, db = desk_of(cam_a, pixels), desk_of(cam_b, pixels)
centres_a = np.array([c for c, _ in placed_a])
near = np.min(np.linalg.norm(da[:, None, :] - centres_a[None], axis=2), axis=1) < 160
centres_b = np.array([c for c, _ in placed_b])
near &= np.min(np.linalg.norm(db[:, None, :] - centres_b[None], axis=2), axis=1) < 160 + 1e9   # a's area is enough
pa, pb = da[near], db[near]
ma, mb = pa.mean(axis=0), pb.mean(axis=0)
u, s, vt = np.linalg.svd((pa - ma).T @ (pb - mb))
rot = u @ vt
if np.linalg.det(rot) < 0:
    u[:, -1] *= -1
    rot = u @ vt
left = np.linalg.norm(pa - (ma + (pb - mb) @ rot.T), axis=1)
scale = s.sum() / ((pb - mb) ** 2).sum()
lens_b = ma + (cam_b.position[:2] - mb) @ rot.T
print(f"\nthe two fits' desks laid on each other ({near.sum()} pixels over the 10:31 cubes' area): the same pixel "
      f"{np.median(left):.1f} mm apart typically, {left.max():.1f} at worst; their scales differ x{scale:.4f}")
print(f"lens, 10:31 fit: ({cam_a.position[0]:.1f}, {cam_a.position[1]:.1f}, {cam_a.position[2]:.1f}); 17:30 fit in the "
      f"same desk: ({lens_b[0]:.1f}, {lens_b[1]:.1f}, {cam_b.position[2]:.1f}): "
      f"{np.linalg.norm(lens_b - cam_a.position[:2]):.1f} mm apart across, {cam_b.position[2] - cam_a.position[2]:+.1f} up; "
      f"tilt {tilt_deg(cam_a):.2f} / {tilt_deg(cam_b):.2f} deg; focal {cam_a.focal:.0f} / {cam_b.focal:.0f} px")
