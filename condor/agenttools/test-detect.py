#!/usr/bin/env python3
"""Offline checks for the cube detector alone. No camera, no arm, no motion.

    ../.venv/bin/python agenttools/test-detect.py

Two kinds. Cubes rendered through a real pinhole camera, where the base centre
is known exactly, at the owner's 40-50 degrees and at the old fixture's 23 --
including a cube turned nearly face-on, which is the one the detector used to
get wrong. And the owner's own frames: `files/cubes2.png` against a pinhole
camera fitted to it independently of the detector, and `files/red-cube-45.png`,
a cube with one face pale from glare.

`test-calibration.py` runs the detector through the mapping; this is the
detector on its own.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

import detect                                                         # noqa: E402
from harness import check, run                                        # noqa: E402

FRAME_PX = (2592, 1944)
FOCAL_PX = 1900.0
CAMERA_DISTANCE_MM = 380.0
DESK_BGR = (95, 105, 115)
# Lit top, lit side, shaded side -- red enough for the red window, and the
# shaded one as dark as the owner's (L*=71 here, 81 there).
FACE_BGR = ((60, 60, 235), (45, 45, 200), (30, 30, 125))
# In millimetres on the desk, because that is what a pickup needs; the arm's own
# repeatability is +/-2.6 mm. A cube turned near face-on has a corner where two
# silhouette edges meet almost in line, which slides a few pixels along them for
# a tiny change in the outline: 3.8 px, 0.8 mm, at 8 degrees of turn.
TOLERANCE_MM = 1.0
MM_PER_PX = CAMERA_DISTANCE_MM / FOCAL_PX      # at the cube, for this camera


def _camera(elevation_deg: float, look_at: tuple) -> np.ndarray:
    elevation = math.radians(elevation_deg)
    target = np.array(look_at, dtype=np.float64)
    position = target + CAMERA_DISTANCE_MM * np.array(
        [0.0, -math.cos(elevation), math.sin(elevation)])
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.vstack((right, np.cross(forward, right), forward))
    intrinsics = np.array([[FOCAL_PX, 0.0, FRAME_PX[0] / 2],
                           [0.0, FOCAL_PX, FRAME_PX[1] / 2], [0.0, 0.0, 1.0]])
    return intrinsics @ np.column_stack((rotation, -rotation @ position))


def _project(camera: np.ndarray, point) -> tuple:
    x, y, w = camera @ np.append(point, 1.0)
    return (x / w, y / w)


def render(elevation_deg: float, yaw_deg: float, look_at=(0.0, 0.0, 20.0), shades=FACE_BGR):
    """A 40 mm cube on a desk, through a pinhole camera, and its true centres.

    The cube stands on the desk at the origin turned `yaw_deg`. The top face
    takes the first shade and the two visible sides the others, nearer-to-camera
    side first -- so swapping `shades` round lights it from a different side.
    """
    half = detect.CUBE_MM / 2
    turn = math.radians(yaw_deg)
    spin = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    footprint = [spin @ corner for corner in ((-half, -half), (half, -half), (half, half), (-half, half))]
    camera = _camera(elevation_deg, look_at)
    eye = -np.linalg.inv(camera[:, :3]) @ camera[:, 3]

    sides = []
    for index in range(4):
        start, end = footprint[index], footprint[(index + 1) % 4]
        normal = np.append(spin @ [(0, -1), (1, 0), (0, 1), (-1, 0)][index], 0.0)
        middle = np.append((start + end) / 2, half)
        if normal @ (eye - middle) > 0:
            quad = [np.append(start, 0), np.append(end, 0),
                    np.append(end, detect.CUBE_MM), np.append(start, detect.CUBE_MM)]
            sides.append((float(np.linalg.norm(eye - middle)), quad))
    faces = [[np.append(corner, detect.CUBE_MM) for corner in footprint]]
    faces += [quad for _, quad in sorted(sides, key=lambda side: side[0])]

    image = np.empty((FRAME_PX[1], FRAME_PX[0], 3), np.uint8)
    image[:] = DESK_BGR
    for face, shade in zip(faces, shades):
        polygon = np.array([_project(camera, corner) for corner in face]) * 16
        cv2.fillConvexPoly(image, polygon.astype(np.int32), shade, cv2.LINE_AA, shift=4)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    noise = np.random.default_rng(1).normal(0.0, 3.0, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image, _project(camera, (0.0, 0.0, 0.0)), _project(camera, (0.0, 0.0, detect.CUBE_MM))


def _detect_rendered(label: str, elevation_deg: float, yaw_deg: float, **scene) -> bool:
    image, base, top = render(elevation_deg, yaw_deg, **scene)
    found = detect.find_cubes(image, ["red"])
    if not found:
        return check(label, False, "no cube found")
    det = found[0]
    base_error, top_error = math.dist(det.base_center, base), math.dist(det.top_center, top)
    base_error, top_error = base_error * MM_PER_PX, top_error * MM_PER_PX
    return check(label, base_error < TOLERANCE_MM and top_error < TOLERANCE_MM,
                 f"base {base_error:.2f} mm, top {top_error:.2f} mm out, "
                 f"margin {det.reading_margin:.1f}")


def test_a_steep_camera_at_every_turn() -> bool:
    """The owner's geometry: 45 degrees down, close, the cube off to one side.

    Near face-on one side face is a sliver -- the owner's red cube -- and the
    receding edges project as steeply as the vertical ones. This camera sees the
    cube face-on at 20.7 degrees of turn; 8 is a sliver on one side of that and
    30 on the other.
    """
    is_ok = True
    for yaw in (8, 30, 35, 45, 60, 75, 84):
        is_ok &= _detect_rendered(f"45 deg down, turned {yaw:2d} deg: base and top centres within a millimetre",
                                  45.0, yaw, look_at=(90.0, 30.0, 0.0))
    return is_ok


def test_a_cube_turned_face_on_is_not_reported() -> bool:
    """Two faces, so two base corners hidden, and nothing to read the third from."""
    image, _, _ = render(45.0, 20.7, look_at=(90.0, 30.0, 0.0))
    found = detect.find_cubes(image, ["red"])
    return check("a cube face-on to the camera is refused, not guessed at", not found,
                 f"margin {found[0].reading_margin:.2f}" if found else "")


def test_the_old_fixture_angle() -> bool:
    """23 degrees, the camera the detector was first written for, still works."""
    is_ok = True
    for yaw in (20, 45, 70):
        is_ok &= _detect_rendered(f"23 deg down, turned {yaw} deg", 23.0, yaw)
    return is_ok


def test_the_reading_does_not_need_the_top_lit() -> bool:
    """Which face is brightest is not what decides it; that the faces differ is."""
    darkest_on_top = (FACE_BGR[2], FACE_BGR[0], FACE_BGR[1])
    return _detect_rendered("the top face darkest of the three", 45.0, 30.0,
                            look_at=(-60.0, 0.0, 0.0), shades=darkest_on_top)


def test_a_cube_lit_alike_on_every_face_is_not_reported() -> bool:
    """No shading, no way to tell the reading, so no answer rather than a guess."""
    image, _, _ = render(45.0, 30.0, shades=(FACE_BGR[0],) * 3)
    found = detect.find_cubes(image, ["red"])
    return check("a cube with three identical faces is refused, not guessed at", not found,
                 f"margin {found[0].reading_margin:.2f}" if found else "")


# A pinhole camera fitted to each cube's six silhouette corners in the owner's
# frame (solvePnP, focal length scanned; 1855, 2011 and 1947 px came out for the
# three, which is one camera). It shares nothing with the detector's vanishing
# point construction. `work/STATUS-condor.md` §10.6 had red at (638, 978) -- the
# middle of its *front* bottom edge -- and the detector at the time said (586, 865).
OWNER_FRAME_BASES = {"red": (663.5, 875.5), "green": (2083.2, 1500.3), "blue": (1743.7, 463.3)}
OWNER_FRAME_TOLERANCE_PX = 4.0


def test_the_owners_frame() -> bool:
    image = cv2.imread(str(ROOT / "files" / "cubes2.png"))
    found = {det.colour: det for det in detect.find_cubes(image, ["red", "green", "blue"])}
    is_ok = check("all three cubes are found in cubes2.png", len(found) == 3, str(sorted(found)))
    for colour, expected in OWNER_FRAME_BASES.items():
        if colour not in found:
            continue
        det = found[colour]
        error = math.dist(det.base_center, expected)
        is_ok &= check(f"{colour} base centre agrees with the fitted camera",
                       error < OWNER_FRAME_TOLERANCE_PX,
                       f"({det.base_center[0]:.1f},{det.base_center[1]:.1f}), {error:.1f} px out, "
                       f"margin {det.reading_margin:.1f}")
    return is_ok


# A cube turned 45 degrees with one face catching glare off its own surface: it
# reads pink, about 0.4 of the way from the desk's colour to the cube's. A fixed
# halfway cut lost half of that face and a corner with it, and the base centre
# landed near (1170, 766). Confirmed by the owner by eye, and through the
# 2026-09-29 calibration its four base edges measure 40.2-41.6 mm.
GLARE_FRAME_BASE = (1191.5, 814.1)


def test_a_face_catching_glare_is_still_part_of_the_cube() -> bool:
    image = cv2.imread(str(ROOT / "files" / "red-cube-45.png"))
    found = detect.find_cubes(image, ["red"])
    if not found:
        return check("the red cube in red-cube-45.png is found", False)
    det = found[0]
    error = math.dist(det.base_center, GLARE_FRAME_BASE)
    return check("the glaring face is kept, and the base centre is where it was confirmed",
                 error < OWNER_FRAME_TOLERANCE_PX,
                 f"({det.base_center[0]:.1f},{det.base_center[1]:.1f}), {error:.1f} px out, "
                 f"margin {det.reading_margin:.0f}")


TESTS = [
    test_a_steep_camera_at_every_turn,
    test_a_cube_turned_face_on_is_not_reported,
    test_the_old_fixture_angle,
    test_the_reading_does_not_need_the_top_lit,
    test_a_cube_lit_alike_on_every_face_is_not_reported,
    test_the_owners_frame,
    test_a_face_catching_glare_is_still_part_of_the_cube,
]


if __name__ == "__main__":
    sys.exit(run(TESTS, "the cube detector"))
