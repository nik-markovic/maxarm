#!/usr/bin/env python3
"""Offline checks for the camera half. No camera, no arm, no motion.

    ../.venv/bin/python agenttools/test-calibration.py

`calibration/detect.py`, `mapping.py` and the parts of `calibrate.py` that do not
need a device. `calibration/arm.py` is the other half and `test-scene.py` has it.

Three kinds of check. The arithmetic ones build a synthetic camera -- a known
perspective transform -- project the arm's coordinates through it, and ask the
fit to find its way back; the answer is known exactly, which is the only way to
tell a mapping that is right from one that merely reproduces its own input. Some
run the real detector over `files/cubes1.jpeg` and the real solver over what it
finds, which is what catches the two files drifting apart; the detector on its
own is `test-detect.py`. The last few stand a
fake V4L2 device in front of the capture code, which is the one part of this that
cannot be checked against a camera that is not plugged in.

Its name has a hyphen in it, so `run-tests.py` cannot import it. Run it directly.
"""

import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

import calibrate                                                      # noqa: E402
import detect                                                         # noqa: E402
from arm import CUBE_POSITIONS, CUBE_SIZE_MM                  # noqa: E402
from harness import check, run                                        # noqa: E402
from maxarm.config import Limits                                      # noqa: E402
from mapping import DeskMapping, DeskPlane, Observation, fit          # noqa: E402

# A real camera rather than a plane-to-plane transform: 3x4, so a point can have
# a height and the pixel moves when it does. Placed where the owner's is --
# behind the desk, 220 mm up and 520 mm back, which is 23 degrees of elevation.
# The numbers do not matter; that it is a projection and not a homography does,
# because that is the only way a test can know what height should do to a pixel.
CAMERA_AT = np.array([0.0, -700.0, 220.0])
CAMERA_LOOKS_AT = np.array([0.0, -180.0, 0.0])
FOCAL_PX, PRINCIPAL_PX = 1600.0, (960.0, 540.0)


def _camera() -> np.ndarray:
    forward = CAMERA_LOOKS_AT - CAMERA_AT
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))          # world -> camera
    intrinsics = np.array([[FOCAL_PX, 0.0, PRINCIPAL_PX[0]],
                           [0.0, FOCAL_PX, PRINCIPAL_PX[1]],
                           [0.0, 0.0, 1.0]])
    return intrinsics @ np.column_stack((rotation, -rotation @ CAMERA_AT))


CAMERA = _camera()


def project(x: float, y: float, height_mm: float = 0.0) -> tuple:
    """AACS millimetres to a pixel, through the synthetic camera.

    `height_mm` is height above the desk, so `project(x, y, 0)` is a footprint
    and `project(x, y, 40)` is the top of a cube standing on it -- a different
    pixel, which is the whole subject of the height tests.
    """
    point = CAMERA @ np.array([x, y, height_mm, 1.0])
    return (float(point[0] / point[2]), float(point[1] / point[2]))


def desk_homography() -> np.ndarray:
    """What the camera does to the z=0 plane alone: pixel <- AACS x, y."""
    return CAMERA[:, [0, 1, 3]]


def synthetic(positions, lift_mm: float = 0.0) -> list:
    """Point pairs for these places, optionally with a cube top over each one."""
    observations = [Observation(str(index), project(x, y), (x, y, z), 0.0)
                    for index, (x, y, z) in enumerate(positions)]
    if lift_mm:
        observations += [Observation(str(index), project(x, y, lift_mm), (x, y, z), lift_mm)
                         for index, (x, y, z) in enumerate(positions)]
    return observations


def test_three_points_are_affine_and_exact() -> bool:
    """Three pairs: an affine fit, exact at the three, and honest about it."""
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0), (0.0, -90.0, 48.0)]
    mapping = fit(synthetic(points))
    is_ok = check("three point pairs give an affine fit", mapping.model == "affine")
    is_ok &= check("it passes through all three", max(mapping.residuals_mm()) < 1e-6,
                   f"worst {max(mapping.residuals_mm()):.2e} mm")
    is_ok &= check("and reports no held-out error, having none to report",
                   mapping.held_out_mm() is None)
    return is_ok


def test_affine_cannot_follow_the_perspective() -> bool:
    """The reason four pairs matter, as a number rather than an assertion.

    Fit three points through a genuinely perspective camera, then ask the fit
    where a fourth known point is. The error is what an affine mapping costs at
    this camera's angle, and it is tens of millimetres.
    """
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0), (0.0, -90.0, 48.0)]
    mapping = fit(synthetic(points))
    held = (120.0, -180.0)
    error = float(np.hypot(*(np.array(mapping.pixel_to_ground(project(*held))) - np.array(held))))
    return check("an affine fit misses a fourth point by millimetres it cannot see",
                 error > 1.0, f"{error:.1f} mm away from the three it was fitted through")


def test_three_cubes_with_their_tops_are_a_camera() -> bool:
    """Base and top centres are points at two heights, which is enough for a camera.

    Six points, eleven freedoms: the fit is a full 3x4 projection, so three cubes
    give a perspective mapping -- the fourth point that an affine fit misses by
    millimetres is exact here, on the desk and a tile's height above it.
    """
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0), (0.0, -90.0, 48.0)]
    mapping = fit(synthetic(points, lift_mm=CUBE_SIZE_MM))
    is_ok = check("three cubes and their tops fit a camera", mapping.model == "camera")
    worst = 0.0
    for x, y in ((120.0, -180.0), (-80.0, -120.0), (0.0, -300.0)):
        for height in (0.0, 4.0, CUBE_SIZE_MM):
            found = mapping.pixel_to_ground(project(x, y, height), height)
            worst = max(worst, float(np.hypot(found[0] - x, found[1] - y)))
    is_ok &= check("points it never saw come back where they are, at any height",
                   worst < 1e-6, f"worst {worst:.1e} mm")
    in_a_line = [(0.0, -100.0, 48.0), (0.0, -180.0, 42.0), (0.0, -260.0, 36.0)]
    try:
        fit(synthetic(in_a_line, lift_mm=CUBE_SIZE_MM))
        is_ok &= check("three cubes in a line are refused, tops or not", False, "it fitted")
    except (ValueError, np.linalg.LinAlgError) as error:
        is_ok &= check("three cubes in a line are refused, tops or not", True, str(error))
    return is_ok


def test_four_points_recover_the_camera() -> bool:
    """Four pairs: a perspective fit, and it finds the camera it was projected through."""
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0),
              (0.0, -90.0, 48.0), (120.0, -180.0, 44.0)]
    mapping = fit(synthetic(points))
    is_ok = check("four point pairs give a perspective fit", mapping.model == "perspective")
    # The fit runs pixel -> AACS, so it should be the inverse of what the camera
    # does to the desk plane. Homographies carry an arbitrary scale, so compare
    # them normalised.
    wanted = np.linalg.inv(desk_homography())
    is_ok &= check("which is the camera it was projected through",
                   np.allclose(mapping.matrix / mapping.matrix[2, 2],
                               wanted / wanted[2, 2], atol=1e-6))
    # Five points, so there is one to hold out.
    five = fit(synthetic(points + [(-60.0, -150.0, 45.0)]))
    held_out = five.held_out_mm()
    is_ok &= check("and a fifth point is predicted, not fitted",
                   held_out is not None and max(held_out) < 1e-3,
                   f"worst {max(held_out):.2e} mm")
    return is_ok


def test_round_trip_and_hull() -> bool:
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0),
              (0.0, -90.0, 48.0), (120.0, -180.0, 44.0)]
    mapping = fit(synthetic(points))
    pixel = project(0.0, -200.0)
    is_ok = check("ground_to_pixel undoes pixel_to_ground",
                  np.allclose(mapping.ground_to_pixel(mapping.pixel_to_ground(pixel)), pixel, atol=1e-6))
    is_ok &= check("a point among the cubes is inside the fitted patch",
                   mapping.is_inside(project(0.0, -200.0)))
    is_ok &= check("one well outside it is not", not mapping.is_inside(project(600.0, -600.0)))
    return is_ok


def test_aacs_zero_plane() -> bool:
    """AACS zero is interpolated across the three measured surface heights, exactly."""
    observations = synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()])
    mapping = fit(observations)
    is_ok = True
    for observation in observations:
        x, y, z = observation.arm
        is_ok &= check(f"AACS zero passes through ({x:.0f},{y:.0f},{z:.0f})",
                       abs(mapping.board_z(x, y, 0.0) - z) < 1e-9)
    is_ok &= check("and reads lower at reach, as the arm's z error demands",
                   mapping.board_z(0, -254, 0.0) < mapping.board_z(0, -90, 0.0),
                   f"{mapping.board_z(0, -254, 0.0):.1f} mm at y=-254 against "
                   f"{mapping.board_z(0, -90, 0.0):.1f} at y=-90")
    is_ok &= check("a pixel is on the surface: AACS z is 0 there by definition",
                   mapping.pixel_to_aacs(project(0.0, -200.0))[2] == 0.0)
    is_ok &= check("and board z converts back to the AACS height it came from",
                   abs(mapping.board_to_aacs(mapping.aacs_to_board((0.0, -200.0, 17.0)))[2]
                       - 17.0) < 1e-9)
    return is_ok


def test_a_piece_is_gripped_at_its_own_height() -> bool:
    """AACS z = the piece's height, and `pixel_to_board` is what the arm is given.

    The scene is the check: the arm sets the far cubes down with the cup at 78
    and `CUBE_POSITIONS` records the surface there at 38. A mapping that sent the
    cup to 38 would drive it into the desk.
    """
    observations = synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()])
    mapping = fit(observations)
    is_ok = True
    for colour, (x, y, z) in CUBE_POSITIONS.items():
        pixel = project(x, y)
        cup = mapping.pixel_to_board(pixel, CUBE_SIZE_MM)
        is_ok &= check(f"the cup grips the {colour} cube at board z={z + CUBE_SIZE_MM:.0f}",
                       abs(cup[2] - (z + CUBE_SIZE_MM)) < 1e-9
                       and np.allclose(cup[:2], (x, y), atol=1e-6),
                       f"({cup[0]:.1f},{cup[1]:.1f},{cup[2]:.1f})")
    # Whatever the app's pieces turn out to be, the height is its argument.
    thin = mapping.pixel_to_board(project(0.0, -200.0), 4.0)
    cube = mapping.pixel_to_board(project(0.0, -200.0), CUBE_SIZE_MM)
    is_ok &= check("a 4 mm piece is gripped 36 mm lower than a cube on the same spot",
                   abs((cube[2] - thin[2]) - 36.0) < 1e-9)
    is_ok &= check("and the arm can be sent there -- the library floor allows it",
                   thin[2] > Limits().z_floor,
                   f"board z {thin[2]:.0f} against a floor of {Limits().z_floor:.0f}")
    return is_ok


def test_the_desk_plane_alone_converts_aacs_to_board() -> bool:
    """AACS to board with no camera anywhere near it, on the scene's own numbers.

    The app needs this half on its own: it knows it wants the cup 40 mm above the
    desk at the green cube's place, and `arm.py` already holds the three measured
    surface heights that answer it. The check is the scene -- `SCENE` releases
    green at board z 78, and AACS 40 there has to come out as 78.
    """
    desk = DeskPlane.through(list(CUBE_POSITIONS.values()))
    is_ok = True
    for colour, (x, y, surface) in CUBE_POSITIONS.items():
        board = desk.aacs_to_board((x, y, CUBE_SIZE_MM))
        is_ok &= check(f"AACS {CUBE_SIZE_MM:.0f} at the {colour} cube is board "
                       f"{surface + CUBE_SIZE_MM:.0f}",
                       np.allclose(board, (x, y, surface + CUBE_SIZE_MM), atol=1e-9),
                       f"({board[0]:.0f}, {board[1]:.0f}, {board[2]:.1f})")
    is_ok &= check("and AACS 0 is the surface the owner measured",
                   all(abs(desk.board_z(x, y) - z) < 1e-9
                       for x, y, z in CUBE_POSITIONS.values()))
    is_ok &= check("board to AACS is its inverse anywhere in between",
                   np.allclose(desk.board_to_aacs(desk.aacs_to_board((40.0, -200.0, 4.0))),
                               (40.0, -200.0, 4.0), atol=1e-9))
    # The scene's own release heights, which is where these numbers came from.
    is_ok &= check("which is what the scene commands at the green drop-off",
                   abs(desk.aacs_to_board((100.0, -254.0, 40.0))[2] - 78.0) < 0.05,
                   f"board z {desk.aacs_to_board((100.0, -254.0, 40.0))[2]:.1f}")
    is_ok &= check("two heights are not a plane", _refuses(
        lambda: DeskPlane.through(list(CUBE_POSITIONS.values())[:2])))
    return is_ok


def test_the_mapping_carries_the_same_plane() -> bool:
    """A solved mapping answers AACS-to-board identically to the plane alone."""
    mapping = fit(synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()]))
    desk = DeskPlane.through(list(CUBE_POSITIONS.values()))
    pose = (40.0, -200.0, 4.0)
    return check("the camera mapping and the bare desk plane agree",
                 np.allclose(mapping.aacs_to_board(pose), desk.aacs_to_board(pose), atol=1e-9)
                 and mapping.plane == desk)


def _refuses(call) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def test_a_ray_is_straight() -> bool:
    """The height model, against a camera that really has perspective.

    Two solved planes and a linear interpolation between them is not an
    approximation: a pixel is a ray, a ray is a straight line, so where it
    crosses `z = h` is linear in `h`. If that is right, a feature at *any*
    height -- 4 mm, 17 mm, 40 mm, 120 mm well above both planes -- comes back at
    the exact millimetre it was projected from.
    """
    # Four per plane, so both planes get a perspective fit rather than an affine
    # one; with three each, the affine error swamps what is being checked here.
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0),
              (0.0, -90.0, 48.0), (120.0, -180.0, 44.0)]
    mapping = fit(synthetic(points, lift_mm=CUBE_SIZE_MM))
    is_ok = check("both planes are solved", mapping.lifted is not None
                  and mapping.lift_mm == CUBE_SIZE_MM, f"lift {mapping.lift_mm:.0f} mm")
    place = (40.0, -200.0)
    for height in (4.0, 17.0, CUBE_SIZE_MM, 120.0):
        found = mapping.pixel_to_ground(project(*place, height), height)
        error = float(np.hypot(*(np.array(found) - np.array(place))))
        # A hundredth of a micron. The fit is exact; what is left is the DLT's
        # own floating point, and it grows only where the height is far above
        # both planes and the interpolation has become an extrapolation.
        is_ok &= check(f"a feature {height:5.1f} mm up is placed where it really is",
                       error < 1e-2, f"{error:.2e} mm out")
    return is_ok


def test_height_changes_the_answer_enough_to_matter() -> bool:
    """How wrong ignoring height would be, in millimetres, for the app's tiles.

    The camera looks along the desk, so a 4 mm tile's printed face is several
    millimetres from the tile's own footprint. That is larger than the arm's
    own accuracy, which is what makes it a correction rather than a detail.
    """
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0),
              (0.0, -90.0, 48.0), (120.0, -180.0, 44.0)]
    mapping = fit(synthetic(points, lift_mm=CUBE_SIZE_MM))
    place = (40.0, -200.0)
    tile_pixel = project(*place, 4.0)
    uncorrected = mapping.pixel_to_ground(tile_pixel)              # as if it were flat
    corrected = mapping.pixel_to_ground(tile_pixel, 4.0)
    slip = float(np.hypot(*(np.array(uncorrected) - np.array(place))))
    is_ok = check("ignoring 4 mm of height misplaces a tile by millimetres",
                  slip > 3.0, f"{slip:.1f} mm out")
    is_ok &= check("correcting for it puts the tile where it is",
                   np.allclose(corrected, place, atol=1e-6))
    is_ok &= check("and parallax_mm reports that distance",
                   abs(mapping.parallax_mm(tile_pixel, 4.0) - slip) < 1e-3,
                   f"{mapping.parallax_mm(tile_pixel, 4.0):.1f} mm")
    return is_ok


def test_height_without_the_second_plane_is_refused() -> bool:
    """A mapping with no cube tops in it cannot place a tile, and says so.

    The alternative is answering anyway, several millimetres out, with nothing
    to show that it happened -- which is the same trap as using a cube's blob
    centre instead of its base.
    """
    mapping = fit(synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()]))
    is_ok = check("a desk-only fit maps the desk", mapping.lifted is None
                  and mapping.pixel_to_ground(project(0.0, -200.0)) is not None)
    try:
        mapping.pixel_to_ground(project(0.0, -200.0, 4.0), 4.0)
    except ValueError as error:
        return is_ok and check("but refuses a height it has no evidence for", True, str(error))
    return check("but refuses a height it has no evidence for", False, "it answered anyway")


def test_mixed_lift_heights_are_refused() -> bool:
    """One plane above the desk, not several. Two heights is a different solver."""
    points = [(x, y, z) for x, y, z in CUBE_POSITIONS.values()]
    observations = synthetic(points, lift_mm=CUBE_SIZE_MM)
    observations.append(Observation("odd", project(0.0, -150.0, 20.0), (0.0, -150.0, 44.0), 20.0))
    try:
        fit(observations)
    except ValueError as error:
        return check("point pairs at two different heights are refused",
                     "height" in str(error), str(error))
    return check("point pairs at two different heights are refused", False, "it fitted something")


def test_top_centre_sits_above_the_base_in_the_fixture() -> bool:
    """The detector's own top centre, on a real frame rather than a synthetic one.

    It should sit above the base centre in the image and roughly a cube-height
    away from it, scaled by what the detector measured for that cube.
    """
    image = cv2.imread(str(ROOT / "files" / "cubes1.jpeg"))
    is_ok = True
    for det in detect.find_cubes(image, ["red", "green", "blue"]):
        rise = det.base_center[1] - det.top_center[1]
        expected = CUBE_SIZE_MM / det.mm_per_px
        is_ok &= check(f"{det.colour}'s top centre is above its base, about a cube up",
                       rise > 0 and 0.5 * expected < rise < 1.5 * expected,
                       f"{rise:.0f} px against {expected:.0f} px for {CUBE_SIZE_MM:.0f} mm")
    return is_ok


def test_config_round_trip() -> bool:
    mapping = fit(synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        mapping.save(path)
        reloaded = DeskMapping.load(path)
        is_ok = check("the written config is readable JSON", isinstance(
            json.loads(path.read_text()), dict))
        is_ok &= check("and reloads to the same mapping",
                       np.allclose(reloaded.matrix, mapping.matrix)
                       and reloaded.plane == mapping.plane
                       and reloaded.observations == mapping.observations)
        pixel = project(0.0, -200.0)
        is_ok &= check("which sends the cup to the same place",
                       np.allclose(reloaded.pixel_to_board(pixel, CUBE_SIZE_MM),
                                   mapping.pixel_to_board(pixel, CUBE_SIZE_MM)))
    return is_ok


def test_too_few_points() -> bool:
    observations = synthetic([(0.0, -90.0, 48.0), (100.0, -254.0, 38.0)])
    try:
        fit(observations)
    except ValueError as error:
        return check("two point pairs are refused", "three" in str(error), str(error))
    return check("two point pairs are refused", False, "it fitted something")


def test_collinear_cubes() -> bool:
    """Three cubes in a line determine nothing, and should say so rather than fit."""
    observations = synthetic([(0.0, -100.0, 48.0), (0.0, -180.0, 42.0), (0.0, -260.0, 36.0)])
    try:
        fit(observations)
    except (ValueError, np.linalg.LinAlgError):
        return check("three collinear cubes are refused", True)
    return check("three collinear cubes are refused", False, "it fitted something")


def test_detector_still_finds_the_fixture() -> bool:
    """The real detector over the real frame, and the pixels it gives.

    Re-pinned when the detector learned perspective (`work/STATUS-condor.md`
    §10.6): the prototype's numbers were blue (1303.4, 192.4), green (1507.0,
    636.8) and red (407.5, 382.0) -- the midpoint of two corners rather than where
    the base's diagonals cross, and red's lower-left corner 25 px up its face.
    They are here to catch the detector moving; `test-detect.py` is where it is
    checked against a camera that knows the answer.
    """
    image = cv2.imread(str(ROOT / "files" / "cubes1.jpeg"))
    detections = {det.colour: det for det in detect.find_cubes(image, ["red", "green", "blue"])}
    expected = {"blue": (1299.5, 192.2), "green": (1497.3, 631.3), "red": (422.6, 388.7)}
    is_ok = check("all three cubes are found in cubes1.jpeg", len(detections) == 3)
    for colour, pixel in expected.items():
        if colour not in detections:
            is_ok = False
            continue
        found = detections[colour].base_center
        is_ok &= check(f"{colour} base is where it was", np.allclose(found, pixel, atol=1.0),
                       f"({found[0]:.1f},{found[1]:.1f})")
    return is_ok


def test_fit_over_the_fixture() -> bool:
    """The detector's own pixels through the solver, end to end.

    `cubes1.jpeg` was taken before the arm ever placed a cube, so its cubes are
    not at `CUBE_POSITIONS` and the millimetres this produces mean nothing. What
    it does check is that the two halves join up, and that the cube edge check
    -- the one number that judges a three-point fit -- comes out of it.
    """
    image = cv2.imread(str(ROOT / "files" / "cubes1.jpeg"))
    detections = detect.find_cubes(image, ["red", "green", "blue"])
    observations = [Observation(det.colour, det.base_center, CUBE_POSITIONS[det.colour])
                    for det in detections]
    mapping = fit(observations)
    is_ok = check("the fixture's three cubes fit", mapping.model == "affine")
    is_ok &= check("each detected base maps back to the coordinate it was paired with",
                   max(mapping.residuals_mm()) < 1e-6)
    for det in detections:
        lengths = calibrate.edge_lengths_mm(mapping, det)
        is_ok &= check(f"{det.colour} base edges measure through the mapping",
                       all(1.0 < length < 400.0 for length in lengths),
                       f"{lengths[0]:.1f} / {lengths[1]:.1f} mm")
    return is_ok


def test_cube_edges_are_40_mm_under_a_true_mapping() -> bool:
    """The check that judges a three-point fit, checked itself.

    A cube's base is 40 mm square wherever it stands, so a mapping that is right
    makes it measure 40 mm. Here the mapping *is* right -- the synthetic camera
    inverted exactly -- so the only way this fails is if the check is wrong.
    """
    points = [(-100.0, -254.0, 36.0), (100.0, -254.0, 38.0),
              (0.0, -90.0, 48.0), (120.0, -180.0, 44.0)]
    mapping = fit(synthetic(points))
    # A cube standing at (0, -200), turned so the camera sees a corner: the two
    # opposite base corners and the near one between them, half a diagonal out.
    half_diagonal = CUBE_SIZE_MM / math.sqrt(2)
    corners = [project(-half_diagonal, -200.0), project(0.0, -200.0 - half_diagonal),
               project(half_diagonal, -200.0)]
    detection = detect.CubeDetection(
        colour="red", base_corners=tuple(corners), top_corners=tuple(corners),
        far_base=project(0.0, -200.0 + half_diagonal), near_top=corners[1],
        silhouette_center=corners[0], hexagon=list(corners))
    lengths = calibrate.edge_lengths_mm(mapping, detection)
    return check("a 40 mm base measures 40 mm through a mapping that is right",
                 all(abs(length - CUBE_SIZE_MM) < 1e-4 for length in lengths),
                 f"{lengths[0]:.3f} / {lengths[1]:.3f} mm")


def test_placements_and_pairing() -> bool:
    """`--at` overrides one cube, and a cube that is not in the frame is dropped."""
    placements = calibrate.cube_placements(["red=120,-180,44"])
    is_ok = check("--at moves one cube and leaves the others alone",
                  placements["red"] == (120.0, -180.0, 44.0)
                  and placements["green"] == CUBE_POSITIONS["green"], str(placements["red"]))
    without_z = calibrate.cube_placements(["red=120,-180"])
    is_ok &= check("without a z it keeps the height arm.py measured",
                   without_z["red"] == (120.0, -180.0, CUBE_POSITIONS["red"][2]))
    image = cv2.imread(str(ROOT / "files" / "cubes1.jpeg"))
    detections = [det for det in detect.find_cubes(image, ["red", "green", "blue"])
                  if det.colour != "green"]
    pairs, missing = calibrate.pair_up(detections, dict(CUBE_POSITIONS))
    is_ok &= check("a cube that is not in the frame is named, not guessed at",
                   len(pairs) == 4 and missing == ["green"],
                   f"{len(pairs)} pair(s), missing {missing}")
    is_ok &= check("each cube gives two: its base on the desk and its top above it",
                   sorted(obs.height_mm for obs, _ in pairs) == [0.0, 0.0, 40.0, 40.0],
                   str(sorted(obs.height_mm for obs, _ in pairs)))
    is_ok &= check("and each pair carries the coordinate that colour was placed at",
                   all(obs.arm == CUBE_POSITIONS[obs.colour] for obs, _ in pairs))
    is_ok &= check("and the detection it came from, which is what measures its edges",
                   all((det.base_center if not obs.height_mm else det.top_center) == obs.pixel
                       for obs, det in pairs))
    return is_ok


def test_keeping_earlier_points_drops_repeats() -> bool:
    """`--keep` adds the placements that are new and only those.

    A cube that has not moved is detected again in the new frame at the same
    coordinate. Adding it twice would double its weight, and would let one copy
    predict the other in `held_out_mm()` -- a held-out error that flatters
    itself is worse than none at all.
    """
    earlier = fit(synthetic([(x, y, z) for x, y, z in CUBE_POSITIONS.values()]))
    this_frame = [Observation(colour, (10.0 * index, 20.0 * index), position)
                  for index, (colour, position) in enumerate(CUBE_POSITIONS.items())]
    this_frame.append(Observation("red", (500.0, 500.0), (120.0, -180.0, 44.0)))
    with tempfile.TemporaryDirectory() as directory:
        original = calibrate.CALIBRATION
        calibrate.CALIBRATION = Path(directory) / "calibration.json"
        try:
            earlier.save(calibrate.CALIBRATION)
            observations = quietly(calibrate.kept, this_frame)
        finally:
            calibrate.CALIBRATION = original
    is_ok = check("three cubes that have not moved add nothing, the fourth placement does",
                  len(observations) == 4, f"{len(observations)} point pairs")
    is_ok &= check("and the pair that is kept is the new placement",
                   observations[-1].arm == (120.0, -180.0, 44.0))
    is_ok &= check("which is enough for a perspective fit",
                   len({obs.ground for obs in observations}) == 4)
    return is_ok


def test_one_degenerate_subset_does_not_end_the_run() -> bool:
    """Holding a point out can leave a set that determines nothing. Skip, do not raise.

    These five are the ones that found this: the same cube claimed at two
    different pixels with one coordinate between them, which `kept()` now
    prevents but a hand-written `--at` still can. Three of the five subsets
    collapse. The other two answers are still worth having, and a calibration
    run should not end in a traceback over it.
    """
    observations = [
        Observation("blue", (1303.4, 192.4), (-100.0, -254.0, 36.0)),
        Observation("green", (1507.0, 636.8), (100.0, -254.0, 38.0)),
        Observation("red", (407.5, 382.0), (0.0, -90.0, 48.0)),
        Observation("green", (65.5, 508.0), (100.0, -254.0, 38.0)),
        Observation("red", (401.5, 382.5), (120.0, -180.0, 44.0)),
    ]
    held_out = fit(observations).held_out_mm()
    return check("the held-out pass reports what it could and skips what it could not",
                 held_out is not None and 0 < len(held_out) < len(observations),
                 f"{len(held_out) if held_out else 0} of {len(observations)} points")


def test_capture_takes_the_largest_frame() -> bool:
    """The capture code against a fake V4L2 device that clamps like a real one.

    Both cameras this has met max out in a format the other does not have, so
    what is being checked is that nothing here knows the name of a camera.
    """
    camera = FakeCamera({"YUYV": (1280, 720), "MJPG": (2592, 1944)})
    is_ok = check("the bigger frame wins even though it is the compressed one",
                  calibrate.largest_frame(camera) == ("MJPG", 2592, 1944),
                  str(calibrate.largest_frame(camera)))
    even = FakeCamera({"YUYV": (1920, 1080), "MJPG": (1920, 1080)})
    is_ok &= check("at the same size the uncompressed one wins",
                   calibrate.largest_frame(even) == ("YUYV", 1920, 1080))
    picky = FakeCamera({"YUYV": (2304, 1536), "MJPG": (1920, 1080)}, is_lying=True)
    is_ok &= check("a camera that agrees to a size and hands over 640x480 is believed "
                   "about the frame, not the size",
                   calibrate.largest_frame(picky) == ("YUYV", 640, 480),
                   str(calibrate.largest_frame(picky)))
    return is_ok


def test_capture_averages_frames() -> bool:
    """Several frames into one image, and a camera that stops is not silently short."""
    camera = FakeCamera({"YUYV": (640, 480)}, brightnesses=(10, 20, 30, 40, 50))
    calibrate._request(camera, "YUYV", 640, 480)
    averaged = calibrate.average_frames(camera)
    is_ok = check(f"{calibrate.AVERAGED_FRAMES} frames average to their mean",
                  averaged.shape == (480, 640, 3) and int(averaged[0, 0, 0]) == 30,
                  f"{averaged.shape}, value {int(averaged[0, 0, 0])}")
    stopping = FakeCamera({"YUYV": (640, 480)}, frames_before_failing=2)
    calibrate._request(stopping, "YUYV", 640, 480)
    try:
        calibrate.average_frames(stopping)
    except RuntimeError as error:
        return is_ok and check("a camera that stops mid-snapshot says so", True, str(error))
    return check("a camera that stops mid-snapshot says so", False, "it returned an image")


def quietly(function, *args):
    """Run it with its own narration swallowed -- the checks are the output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


class FakeCamera:
    """A V4L2 device: it clamps a request to a size it has, and may lie about it."""

    def __init__(self, modes, is_lying: bool = False, brightnesses=(), frames_before_failing=None):
        self.modes = modes
        self.is_lying = is_lying               # agrees, then hands over 640x480
        self.brightnesses = list(brightnesses)
        self.frames_before_failing = frames_before_failing
        self.fourcc, self.size, self.reads = "YUYV", (640, 480), 0

    def set(self, prop: int, value: float) -> None:
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = "".join(chr((int(value) >> (8 * shift)) & 0xFF) for shift in range(4))
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.size = (int(value), self.size[1])
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.size = (self.size[0], int(value))

    def read(self):
        self.reads += 1
        if self.frames_before_failing is not None and self.reads > self.frames_before_failing:
            return False, None
        if self.fourcc not in self.modes:
            return False, None
        width, height = (640, 480) if self.is_lying else (
            min(self.size[0], self.modes[self.fourcc][0]),
            min(self.size[1], self.modes[self.fourcc][1]))
        value = self.brightnesses.pop(0) if self.brightnesses else 0
        return True, np.full((height, width, 3), value, dtype=np.uint8)


TESTS = [
    test_three_points_are_affine_and_exact,
    test_affine_cannot_follow_the_perspective,
    test_three_cubes_with_their_tops_are_a_camera,
    test_four_points_recover_the_camera,
    test_round_trip_and_hull,
    test_aacs_zero_plane,
    test_a_piece_is_gripped_at_its_own_height,
    test_the_desk_plane_alone_converts_aacs_to_board,
    test_the_mapping_carries_the_same_plane,
    test_a_ray_is_straight,
    test_height_changes_the_answer_enough_to_matter,
    test_height_without_the_second_plane_is_refused,
    test_mixed_lift_heights_are_refused,
    test_top_centre_sits_above_the_base_in_the_fixture,
    test_config_round_trip,
    test_too_few_points,
    test_collinear_cubes,
    test_detector_still_finds_the_fixture,
    test_fit_over_the_fixture,
    test_cube_edges_are_40_mm_under_a_true_mapping,
    test_placements_and_pairing,
    test_keeping_earlier_points_drops_repeats,
    test_one_degenerate_subset_does_not_end_the_run,
    test_capture_takes_the_largest_frame,
    test_capture_averages_frames,
]


if __name__ == "__main__":
    sys.exit(run(TESTS, "the camera half"))
