#!/usr/bin/env python3
"""The camera fit against scenes drawn through a known camera (`cube_scene.py`): no camera, no arm.

What is checked is whether the fit finds the camera that drew the scene, takes in
the arm's stretch, keeps the arm's green body out of it, and refuses what it
cannot fit, saying why. The drawn scenes say nothing about how close a real
camera gets: that is in work/STATUS-gecko.md.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

from maxarm import Space, Spaces                               # noqa: E402
from maxarm.path import REACH_MAX_MM, REACH_MIN_MM             # noqa: E402
from camera import check as check_image                        # noqa: E402
from camera import store                                       # noqa: E402
from camera.cubes import CubesNotFound                         # noqa: E402
from camera.fit import CameraFit, CameraNotFound, fit_camera   # noqa: E402
import cube_scene                                              # noqa: E402
from harness import check                                      # noqa: E402

STRETCH = ((0.965, 0.0), (0.0, 1.02))          # the arm's x short, its y long: as measured on the owner's arm


def in_reach(heights=(0.0, 40.0)) -> np.ndarray:
    """Desk points every 20 mm within the arm's reach, at each height."""
    return np.array([(x, y, z) for x in range(-280, 281, 20) for y in range(-280, 281, 20) for z in heights
                     if REACH_MIN_MM <= math.hypot(x, y) <= REACH_MAX_MM
                     and abs(math.degrees(math.atan2(x, -y))) <= 120], float)


def pixels(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def apart_px(fit: CameraFit, scene: cube_scene.Scene) -> np.ndarray:
    """How far the fit puts each desk point in view from where the drawing camera does, pixels."""
    points = in_reach()
    truth = pixels(scene.desk_to_pixel, points)
    width, height = cube_scene.SIZE
    seen = (truth[:, 0] >= 0) & (truth[:, 0] < width) & (truth[:, 1] >= 0) & (truth[:, 1] < height)
    return np.linalg.norm(pixels(fit.desk_to_pixel, points[seen]) - truth[seen], axis=1)


def test_the_camera_that_drew_it() -> bool:
    print("The fit finds the camera that drew the scene, and the arm's stretch:")
    scene = cube_scene.render(stretch=STRETCH)
    fit = fit_camera(scene.frame, scene.points)
    apart = apart_px(fit, scene)
    x_scale, y_scale, skew = fit.arm_scale()
    focal = 0.74 * cube_scene.SIZE[0]
    return all([
        check("every desk point in reach and view, on the desk and 40 mm up, within 2 px of where it was drawn",
              apart.max() < 2.0, f"median {np.median(apart):.2f}, worst {apart.max():.2f} px"),
        check("the cubes' edges on the frame's, half the time within 0.2 px", fit.edge_px[0] < 0.2,
              f"{fit.edge_px[0]:.2f} px, {fit.edges_used} points"),
        check("the focal length within 1%", abs(fit.focal / focal - 1) < 0.01, f"{fit.focal:.1f} for {focal:.1f}"),
        check("the arm's stretch within 1%: x short, y long",
              abs(x_scale - 0.965) < 0.01 and abs(y_scale - 1.02) < 0.01 and abs(skew) < 0.5,
              f"x {x_scale:.4f}, y {y_scale:.4f}, skew {skew:+.2f} deg"),
        check("the camera where it stood", np.linalg.norm(fit.camera_position - _camera_desk(scene)) < 10.0,
              f"{np.round(fit.camera_position, 1)} for {np.round(_camera_desk(scene), 1)}"),
    ])


def test_the_arms_green_body() -> bool:
    print("A green shape as green as the green cube, round the arm's base, is not taken for it:")
    with_decoy = fit_camera(cube_scene.render(decoy=True).frame, cube_scene.SPOTS)
    without = fit_camera(cube_scene.render(decoy=False).frame, cube_scene.SPOTS)
    return check("the same camera with it as without it",
                 np.abs(pixels(with_decoy.desk_to_pixel, in_reach()) - pixels(without.desk_to_pixel, in_reach()))
                 .max() < 1.0)


def test_green_and_blue_by_where_they_stand() -> bool:
    print("Green and blue are told apart by where the arm recorded them, if the camera mixes their colours:")
    scene = cube_scene.render(stretch=STRETCH)
    swapped = dict(scene.points, green=scene.points["blue"], blue=scene.points["green"])
    fit = fit_camera(scene.frame, swapped)
    return check("each recorded point gets the cube standing there: the same camera",
                 apart_px(fit, scene).max() < 2.0, f"worst {apart_px(fit, scene).max():.2f} px")


def test_refusals() -> bool:
    print("What the fit refuses, and what it says:")
    two_cubes = cube_scene.render(spots={colour: spot for colour, spot in cube_scene.SPOTS.items()
                                         if colour != "red"}, decoy=False)
    moved = cube_scene.render(spots=dict(cube_scene.SPOTS, red=(160.0, -45.0)))
    missing = _refusal(lambda: fit_camera(two_cubes.frame, cube_scene.SPOTS))
    shifted = _refusal(lambda: fit_camera(moved.frame, cube_scene.SPOTS))
    return all([
        check("a cube out of the picture: what the frame does show",
              isinstance(missing, CubesNotFound) and "2 whole, cube-shaped" in str(missing), str(missing)),
        check("a cube moved 45 mm after RECORD: not where the arm recorded it",
              isinstance(shifted, CameraNotFound) and "moved after RECORD" in str(shifted), str(shifted)),
    ])


def test_camera_json() -> bool:
    print("camera.json: what the library reads, and what the page reads back:")
    scene = cube_scene.render(stretch=STRETCH)
    fit = fit_camera(scene.frame, scene.points)
    with tempfile.TemporaryDirectory() as scratch:
        config = Path(scratch)
        store.save_camera(config, fit)
        stored = json.loads((config / store.CAMERA_FILE).read_text())
        again = store.load_camera(config)
        spaces = Spaces(config)
        red = scene.points["red"]
        top = pixels(scene.desk_to_pixel, np.array([[red[0], red[1], 40.0]]))[0]
        seen = spaces.convert((top[0], top[1], 40.0), Space.CAMERA, Space.DESK)
        summary = store.describe_calibration(config)["summary"]
    other_size = np.zeros((480, 640, 3), np.uint8)
    return all([
        check("desk_to_pixel and the frame size, and the fit behind them",
              np.allclose(stored["desk_to_pixel"], fit.desk_to_pixel) and stored["frame_size"] == [1296, 972]),
        check("read back as written", np.allclose(again.desk_to_pixel, fit.desk_to_pixel)
              and again.cubes == fit.cubes),
        check("the library's camera space: the red cube's top centre is the desk point under the cup",
              math.dist(seen[:2], red) < 0.5, f"{np.round(seen, 2)} for {red}"),
        check("the page's line says how well it fits and the arm's error",
              "cube edges within" in summary and "short" in summary and "long" in summary, summary),
        check("a frame of another size is refused for it", check_image.frame_size_problem(other_size, fit)
              is not None and check_image.frame_size_problem(scene.frame, fit) is None),
        check("two calibrations of the same scene put the desk in the same place",
              check_image.difference_mm(fit, again)[1] < 1e-6),
    ])


def _camera_desk(scene: cube_scene.Scene) -> np.ndarray:
    _, _, vectors = np.linalg.svd(scene.desk_to_pixel)
    return vectors[-1][:3] / vectors[-1][3]


def _refusal(action):
    try:
        action()
    except (CubesNotFound, CameraNotFound) as error:
        return error
    return None


TESTS = [test_the_camera_that_drew_it, test_the_arms_green_body, test_green_and_blue_by_where_they_stand,
         test_refusals, test_camera_json]
