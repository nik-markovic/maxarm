#!/usr/bin/env python3
"""Coordinate spaces: round trips, the camera, directions, and what elephant's calibration says.

Arithmetic only. The camera checks use elephant's `camera.json` (`harness.BASELINE`),
a real 3x4, so a point at any height and its pixel can be compared exactly.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import Camera, Space, Spaces                      # noqa: E402
from maxarm.spaces import Affine, angle_from_desk, angle_to_desk  # noqa: E402
from harness import BASELINE, check                            # noqa: E402

# condor's cube bases: arm x, y, and the board z of the bare desk there, as the
# owner measured them. Desk height 0 must land on all three.
CUBE_DESK_HEIGHTS = [(100.0, -254.0, 38.0), (-100.0, -254.0, 36.0), (0.0, -90.0, 48.0)]


def test_affine_round_trip() -> bool:
    print("a 4x4 space:")
    turn = math.radians(30.0)
    matrix = np.array([[math.cos(turn), -math.sin(turn), 0.0, 40.0],
                       [math.sin(turn), math.cos(turn), 0.0, -200.0],
                       [0.0, 0.0, 1.0, 12.0], [0.0, 0.0, 0.0, 1.0]])
    space = Affine(matrix)
    point = (25.0, 60.0, 4.0)
    desk = space.to_desk(point)
    back = space.from_desk(desk)
    return all([
        check("onto desk space by the matrix", math.dist(desk, matrix[:3] @ (*point, 1.0)) < 1e-9),
        check("and back", math.dist(back, point) < 1e-9, f"{math.dist(back, point):.1e} mm"),
        check("a direction turns with it", abs(angle_to_desk(space, point, 10.0) - 40.0) < 1e-9),
    ])


def test_camera() -> bool:
    print("camera space, the shipped camera:")
    camera = Camera.load(BASELINE / "camera.json")
    worst = 0.0
    for x in (-120.0, 0.0, 110.0):
        for y in (-300.0, -200.0, -100.0):
            for height in (0.0, 4.0, 40.0):
                pixel = camera.from_desk((x, y, height))
                worst = max(worst, math.dist(camera.to_desk(pixel), (x, y, height)))
    pixel = camera.from_desk((10.0, -200.0, 4.0))
    on_desk = camera.to_desk((pixel[0], pixel[1], 0.0))
    centre = camera.position()
    through = camera.matrix @ (*centre, 1.0)
    return all([
        check("desk to pixel and back, 27 points at three heights", worst < 1e-9,
              f"worst {worst:.1e} mm"),
        check("a pixel is a ray: the same pixel at height 0 is elsewhere on the desk",
              math.dist(on_desk[:2], (10.0, -200.0)) > 2.0,
              f"{math.dist(on_desk[:2], (10.0, -200.0)):.1f} mm away"),
        check("the camera centre projects to nothing",
              np.linalg.norm(through) < 1e-12 * np.linalg.norm(camera.matrix),
              f"{np.linalg.norm(through) / np.linalg.norm(camera.matrix):.1e} of the matrix"),
        check("and it stands above the desk", centre[2] > 100.0, f"z {centre[2]:.0f} mm"),
        check("plane_to_pixel agrees with to_pixels",
              np.allclose(camera.to_pixels(np.array([[10.0, -200.0, 4.0]]))[0], pixel[:2])),
    ])


def test_directions_are_exact() -> bool:
    """A direction is converted by both ends of a step; lines stay lines, so it is exact."""
    print("directions through the camera:")
    camera = Camera.load(BASELINE / "camera.json")
    worst = 0.0
    for angle in range(-180, 180, 15):
        desk_point = (-40.0, -230.0, 4.0)
        pixel = camera.from_desk(desk_point)
        image_angle = angle_from_desk(camera, desk_point, float(angle))
        back = angle_to_desk(camera, pixel, image_angle)
        worst = max(worst, abs((back - angle + 180.0) % 360.0 - 180.0))
    return check("desk to image and back, 24 directions", worst < 1e-6, f"worst {worst:.1e} deg")


def test_shipped_desk_calibration() -> bool:
    print("elephant's desk.json:")
    spaces = Spaces(BASELINE)
    heights = [spaces.arm.to_desk(cube)[2] for cube in CUBE_DESK_HEIGHTS]
    desk = spaces.get(Space.DESK)
    return all([
        check("is there", spaces.is_desk_calibrated),
        check("desk height 0 on the bare desk under all three cubes",
              max(abs(h) for h in heights) < 1e-6, f"{[round(h, 6) for h in heights]}"),
        check("desk space is the identity onto itself",
              math.dist(desk.to_desk((1.0, 2.0, 3.0)), (1.0, 2.0, 3.0)) < 1e-12),
        check("convert() goes through desk space",
              math.dist(spaces.convert((100.0, -254.0, 38.0), Space.ARM, Space.DESK)[2:],
                        (0.0,)) < 1e-6),
    ])


def test_missing_and_wrong_files() -> bool:
    print("what a config directory without the files does:")
    with tempfile.TemporaryDirectory() as empty:
        spaces = Spaces(Path(empty))
        refusals = []
        for space in (Space.DESK, Space.CAMERA, Space.CUSTOM):
            try:
                spaces.get(space)
            except FileNotFoundError as error:
                refusals.append(space if "desk.json" in str(error) else None)
        arm = spaces.get(Space.ARM)
        (Path(empty) / "desk.json").write_text(json.dumps({"desk_to_arm": np.eye(4).tolist()}))
        (Path(empty) / "custom.json").write_text(json.dumps({"custom_to_desk": np.eye(3).tolist()}))
        try:
            Spaces(Path(empty)).get(Space.CUSTOM)
            wrong_shape = False
        except ValueError:
            wrong_shape = True
        try:
            Spaces(Path(empty)).get(Space.CAMERA)
            no_camera = False
        except FileNotFoundError as error:
            no_camera = "camera.json" in str(error)
    return all([
        check("no desk.json: not calibrated, and arm space is the identity",
              not spaces.is_desk_calibrated
              and math.dist(arm.to_desk((5.0, -200.0, 40.0)), (5.0, -200.0, 40.0)) < 1e-12),
        check("every other space refuses, naming desk.json", refusals == [
            Space.DESK, Space.CAMERA, Space.CUSTOM], f"{refusals}"),
        check("a 3x3 where a 4x4 belongs is refused", wrong_shape),
        check("camera space without camera.json is refused, naming it", no_camera),
    ])


TESTS = [test_affine_round_trip, test_camera, test_directions_are_exact,
         test_shipped_desk_calibration, test_missing_and_wrong_files]
