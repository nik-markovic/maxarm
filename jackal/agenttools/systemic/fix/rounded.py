#!/usr/bin/env python3
"""The cube's picture, modelled closer to a real cube: its edges rounded, radius r (the owner: under 0.25 mm), so an
outline is seen where the line of sight grazes the rounding -- inside the sharp edge by r (cos a + sin a - 1), a the
view's angle to the faces -- and an edge between two lit faces where the rounding turns, its middle; and with or
without the base outline, where the cube meets the desk (a contact shadow would sit there). For each: the camera
fitted to the 4 Nov points' edges alone, each cube placed freely (the app's SAVE), read against today's grid sheets
(which it does not see), and the arm laid on the cubes by a turn and a shift as a check; and how well the same edges
fit the camera that matches the sheets.

    rounded.py
"""
import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pinned", HERE / "pinned.py")
pn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pn)
onedim, edges, cube_corners = pn.onedim, pn.edges, pn.cube_corners
from applib.camera.model import _FACES  # noqa: E402

FACES = tuple(_FACES) + (((0, 1, 2, 3), (0.0, 0.0, -1.0)),)
ROOT2 = math.sqrt(2.0)


def lens_of(camera):
    return -cv2.Rodrigues(np.array(camera[4:7], float))[0].T @ np.array(camera[7:10], float)


def residual(camera, cube, data, r, base):
    """One cube's found edge points against its edges drawn through `camera`, the edges rounded by r mm."""
    starts, ends, found = data
    c3 = cube_corners(cube[:2], cube[2])
    cs, sn = math.cos(cube[2]), math.sin(cube[2])
    lens = lens_of(camera)
    normals, visible = [], []
    for idx, (nx, ny, nz) in FACES:
        n = np.array((cs * nx - sn * ny, sn * nx + cs * ny, nz))
        normals.append(n)
        visible.append(nz >= 0 and (lens - c3[list(idx)].mean(0)) @ n > 0)
    keys = sorted(set(zip(starts.tolist(), ends.tolist())))
    drawn, where, keep = [], {}, np.ones(len(starts), bool)
    for a, b in keys:
        if not base and a < 4 and b < 4:
            keep &= ~((starts == a) & (ends == b))
            continue
        both = [k for k, (idx, _) in enumerate(FACES) if a in idx and b in idx]
        n1, n2 = normals[both[0]], normals[both[1]]
        shift = np.zeros(3)
        if r > 0 and visible[both[0]] and visible[both[1]]:
            shift = -(ROOT2 - 1) * r * (n1 + n2) / np.linalg.norm(n1 + n2)
        elif r > 0 and (visible[both[0]] or visible[both[1]]):
            e = c3[b] - c3[a]; e /= np.linalg.norm(e)
            v = (c3[a] + c3[b]) / 2 - lens; v -= (v @ e) * e; v /= np.linalg.norm(v)
            m = np.cross(e, v); m /= np.linalg.norm(m)
            if m @ (n1 + n2) < 0:
                m = -m
            shift = -max(0.0, r * (m @ n1 + m @ n2 - 1.0)) * m
        where[(a, b)] = len(drawn)
        drawn += [c3[a] + shift, c3[b] + shift]
    if not drawn:
        return np.zeros(0)
    pix = onedim.project(camera, np.array(drawn))
    s = np.array([where[(a, b)] for a, b in zip(starts[keep].tolist(), ends[keep].tolist())])
    return edges.signed_distances(pix, s, s + 1, found[keep])


def solve(camera, cubes, datas, r, base, free_camera, prior=(2560, 1472), iterations=40):
    """Camera and cubes to the edges, least squares, the cubes' blocks apart."""
    x_cam, x_cubes = camera.astype(float).copy(), cubes.astype(float).copy()
    n = len(datas)

    def blocks(cam, cub):
        return [residual(cam, cub[i], datas[i], r, base) for i in range(n)]

    for _ in range(iterations):
        parts = blocks(x_cam, x_cubes)
        rows = np.cumsum([0] + [len(p) for p in parts])
        res = np.concatenate(parts + ([np.array([x_cam[2] - prior[0] / 2, x_cam[3] - prior[1] / 2]) / (0.1 * prior[0])]
                                      if free_camera else []))
        J = np.zeros((len(res), len(free_camera) + 3 * n))
        for k, i in enumerate(free_camera):
            h = 1e-6 * max(1.0, abs(x_cam[i]))
            cam = x_cam.copy(); cam[i] += h
            moved = np.concatenate(blocks(cam, x_cubes) + [np.array([cam[2] - prior[0] / 2, cam[3] - prior[1] / 2])
                                                           / (0.1 * prior[0])])
            J[:, k] = (moved - res) / h
        for i in range(n):
            for j, h in enumerate((1e-3, 1e-3, 1e-6)):
                cub = x_cubes[i].copy(); cub[j] += h
                J[rows[i]:rows[i + 1], len(free_camera) + 3 * i + j] = \
                    (residual(x_cam, cub, datas[i], r, base) - parts[i]) / h
        w = np.where(np.abs(res) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(res), 1e-12))
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)) + 1e-12 * np.eye(len(A)), -(J.T @ (w * res)),
                               rcond=None)[0]
        x_cam[free_camera] += step[:len(free_camera)]
        x_cubes += step[len(free_camera):].reshape(n, 3)
        if np.max(np.abs(step)) < 1e-7:
            break
    e = np.concatenate(blocks(x_cam, x_cubes))
    return x_cam, x_cubes, float(np.sqrt(np.mean(np.minimum(np.abs(e), 3) ** 2)))


def main():
    rig = onedim.Rig(Path("/tmp/c1104"))
    both = pn.pickle.load(open("/tmp/today-joint.pkl", "rb"))["both"]
    sheets_camera = np.array([both[0], 1.0, both[1], both[2], *both[3:9], 0.0, 0.0])
    start = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
    cubes0 = np.array(rig.starts, float)
    then = [p.edges for p in rig.points]
    moved = [(s, e, np.c_[f, np.ones(len(f))] @ pn.A.T) for s, e, f in then]
    T = pn.tj.matrix(pn.as9(sheets_camera))
    cubes_s0 = []
    for p, c in zip(rig.points, cubes0):
        w = np.linalg.solve(T[:, [0, 1, 3]], np.array([*(pn.A @ [*p.pixel, 1.0]), 1.0]))
        cubes_s0.append((w[0] / w[2], w[1] / w[2], c[2]))
    cubes_s0 = np.array(cubes_s0)
    free_camera = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    for name, r, base in (("sharp, every edge (the app's)", 0.0, True), ("sharp, no base outline", 0.0, False),
                          ("rounded 0.25 mm, every edge", 0.25, True), ("rounded 0.25 mm, no base outline", 0.25, False),
                          ("rounded 0.5 mm, no base outline", 0.5, False)):
        camera, cubes, edge = solve(start, cubes0, then, r, base, free_camera)
        _, lo = rig.arm_misses(cubes[:, :2])
        print(f"{name:34s} fitted alone: edges {edge:.3f} px; {pn.describe(camera)}\n"
              f"{'':34s} {pn.sheets_read(camera)}; the arm by a turn and a shift, left out {np.median(lo):.2f} typ, "
              f"{lo.max():.2f} worst", flush=True)
        _, cubes_s, edge_s = solve(sheets_camera, cubes_s0, moved, r, base, [])
        _, lo = rig.arm_misses(cubes_s[:, :2])
        print(f"{'':34s} the same edges through the sheets' camera: {edge_s:.3f} px; the arm, left out "
              f"{np.median(lo):.2f} typ, {lo.max():.2f} worst", flush=True)


if __name__ == "__main__":
    main()
