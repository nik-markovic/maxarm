#!/usr/bin/env python3
"""Refine condor's calibration with grid paper: a true-millimetre camera, and the arm's own error.

condor solves its camera from six cube points by DLT, free to give it non-square
pixels, and it took them: 1994 px across, 1895 down. Tiles came out skewed up to
5 degrees and an O leaned 20. The grid shows why: the arm had put the cubes where
it did not say. Blue and green, sent to x = -100 and +100, stand 185.6 mm apart,
not 200 -- and a camera fitted through them has to bend to agree.

So the two questions are answered separately:

- **The camera, from the grid.** A sheet of square grid paper flat on the desk
  is aligned as a whole lattice (`grid.py`). That gives the desk plane exactly.
  A camera with square pixels has two freedoms left once that plane is known --
  its principal point, along a curve on which the grid's right angles and equal
  sides both hold -- and the cube tops settle it: through the right camera, each
  top lands straight above its base.
- **Where the arm goes, from the cubes.** The desk frame is turned and shifted
  onto the arm's through the three cube bases -- no stretch, so it stays true
  millimetres. What the arm gets wrong is kept apart, as the correction
  `desk_to_arm`: an affine through the same three cubes, exact where the arm put
  them, as condor's calibration was. Vision works in true millimetres; only what
  is sent to the arm goes through the correction.

    ./calibration/refine.py files/grid-1.png              # rewrite config/calibration.json
    ./calibration/refine.py files/grid-1.png --dry-run    # report only

Starts from `config/calibration-condor.json`, condor's calibration as it came.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import grid                             # noqa: E402
from mapping import DeskMapping, Observation, _plane_of  # noqa: E402

CONDOR = ROOT / "config" / "calibration-condor.json"
CALIBRATION = ROOT / "config" / "calibration.json"
CHECK = ROOT / "config" / "grid-check.jpg"
SQUARE_MM = 5.0                         # the owner's paper, measured
# Where to look for the principal point, and how finely.
PRINCIPAL_REACH_PX = 300
PRINCIPAL_STEP_PX = 2
# The two focal lengths the grid implies must agree this closely to count as one camera.
FOCAL_AGREEMENT = 1e-3


def cameras_on_curve(desk_to_pixel: np.ndarray, frame_size: Tuple[int, int]):
    """Every square-pixel camera consistent with the desk plane, near the frame's centre.

    With the principal point at (cx, cy), the plane's two columns h1, h2 must be
    perpendicular and equally long once K is undone. Each condition gives f in
    closed form; where they agree is one camera.
    """
    width, height = frame_size
    for cy in np.arange(height / 2 - PRINCIPAL_REACH_PX, height / 2 + PRINCIPAL_REACH_PX, PRINCIPAL_STEP_PX):
        for cx in np.arange(width / 2 - PRINCIPAL_REACH_PX, width / 2 + PRINCIPAL_REACH_PX, PRINCIPAL_STEP_PX):
            shifted = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ desk_to_pixel
            (a1, b1, c1), (a2, b2, c2) = shifted[:, 0], shifted[:, 1]
            square_perpendicular = -(a1 * a2 + b1 * b2) / (c1 * c2)
            square_equal = -(a1 * a1 + b1 * b1 - a2 * a2 - b2 * b2) / (c1 * c1 - c2 * c2)
            if square_perpendicular <= 0 or square_equal <= 0:
                continue
            f1, f2 = np.sqrt(square_perpendicular), np.sqrt(square_equal)
            if abs(f1 - f2) < FOCAL_AGREEMENT * f1:
                yield np.array([[(f1 + f2) / 2, 0, cx], [0, (f1 + f2) / 2, cy], [0, 0, 1.0]])


def camera_from_plane(intrinsics: np.ndarray, desk_to_pixel: np.ndarray) -> np.ndarray:
    """K [R | t] whose z = 0 plane is `desk_to_pixel`, with z pointing up at the camera."""
    columns = np.linalg.inv(intrinsics) @ desk_to_pixel
    columns /= (np.linalg.norm(columns[:, 0]) + np.linalg.norm(columns[:, 1])) / 2
    if columns[2, 2] < 0:                  # the desk is in front of the camera
        columns = -columns
    rotation = np.column_stack((columns[:, 0], columns[:, 1], np.cross(columns[:, 0], columns[:, 1])))
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    return intrinsics @ np.column_stack((rotation, columns[:, 2]))


def centre_of(camera: np.ndarray) -> np.ndarray:
    _, _, vectors = np.linalg.svd(camera)
    return vectors[-1][:3] / vectors[-1][3]


def on_plane(camera: np.ndarray, pixel, height: float) -> np.ndarray:
    ground = np.linalg.inv(_plane_of(camera, height)) @ np.array([pixel[0], pixel[1], 1.0])
    return ground[:2] / ground[2]


def cube_lean_mm(camera: np.ndarray, observations: Sequence[Observation]) -> np.ndarray:
    """How far each cube's top lands from straight above its base, through this camera."""
    bases = {obs.colour: obs for obs in observations if not obs.height_mm}
    return np.array([np.linalg.norm(on_plane(camera, obs.pixel, obs.height_mm)
                                    - on_plane(camera, bases[obs.colour].pixel, 0.0))
                     for obs in observations if obs.height_mm])


def rigid_onto(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The distance-keeping map (no stretch) taking source points closest to target: 3x3.

    It may mirror as well as turn, and here it does: the board negates x, so the
    arm's x, y with z up is left-handed, and paper seen from above is not.
    """
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_centre).T @ (target - target_centre))
    rotation = (left @ right).T
    transform = np.eye(3)
    transform[:2, :2] = rotation
    transform[:2, 2] = target_centre - rotation @ source_centre
    return transform


def in_plane(transform: np.ndarray) -> np.ndarray:
    """A 3x3 turn-and-shift of (x, y) as a 4x4 on (x, y, z, 1), z untouched."""
    lifted = np.eye(4)
    lifted[:2, :2] = transform[:2, :2]
    lifted[:2, 3] = transform[:2, 2]
    return lifted


def refine(frame: np.ndarray, condor: DeskMapping, square_mm: float = SQUARE_MM):
    lattice = grid.fit(frame, condor)
    desk_to_pixel = lattice.square_to_pixel @ np.diag([1 / square_mm, 1 / square_mm, 1.0])
    candidates = []
    for flip in (1.0, -1.0):               # the paper's v may run either way round
        plane = desk_to_pixel @ np.diag([1.0, flip, 1.0])
        for intrinsics in cameras_on_curve(plane, frame.shape[1::-1]):
            camera = camera_from_plane(intrinsics, plane)
            if centre_of(camera)[2] > 0:
                candidates.append((float(np.mean(cube_lean_mm(camera, condor.observations))), camera,
                                   intrinsics))
    if not candidates:
        raise ValueError("no square-pixel camera fits this grid near the frame's centre")
    lean, camera, intrinsics = min(candidates, key=lambda candidate: candidate[0])

    bases = [obs for obs in condor.observations if not obs.height_mm]
    seen = np.array([on_plane(camera, obs.pixel, 0.0) for obs in bases])
    arm = np.array([obs.ground for obs in bases])
    onto_arm = rigid_onto(seen, arm)
    desk_camera = camera @ np.linalg.inv(in_plane(onto_arm))
    desk_bases = np.array([on_plane(desk_camera, obs.pixel, 0.0) for obs in bases])
    correction = np.vstack((cv2.getAffineTransform(desk_bases.astype(np.float32), arm.astype(np.float32)),
                            (0.0, 0.0, 1.0)))
    return {"lattice": lattice, "intrinsics": intrinsics, "desk_camera": desk_camera,
            "desk_to_arm": correction, "cube_lean_mm": cube_lean_mm(desk_camera, condor.observations),
            "arm_misses_mm": np.linalg.norm(desk_bases - arm, axis=1), "bases": bases,
            "desk_bases": desk_bases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frame", type=Path, help="grid paper flat on the desk, camera untouched since")
    parser.add_argument("--square-mm", type=float, default=SQUARE_MM)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frame = cv2.imread(str(args.frame))
    condor = DeskMapping.load(CONDOR)
    result = refine(frame, condor, args.square_mm)

    lattice, intrinsics = result["lattice"], result["intrinsics"]
    f, cx, cy = intrinsics[0, 0], intrinsics[0, 2], intrinsics[1, 2]
    height, width = frame.shape[:2]
    fov = np.degrees(2 * np.arctan(np.array((width, height, np.hypot(width, height))) / 2 / f))
    print(f"grid: {lattice.squares[0]} x {lattice.squares[1]} squares aligned, correlation "
          f"{lattice.correlation:.3f}")
    print(f"camera: square pixels, f {f:.0f} px, principal point ({cx:.0f}, {cy:.0f}) -- "
          f"{fov[0]:.1f} x {fov[1]:.1f} degrees, {fov[2]:.1f} diagonal")
    print(f"        at {np.round(centre_of(result['desk_camera']), 1)} mm; cube tops land "
          f"{np.round(result['cube_lean_mm'], 1)} mm from straight above their bases")
    print("where the arm put the cubes, in true millimetres after the best turn and shift:")
    for obs, desk, miss in zip(result["bases"], result["desk_bases"], result["arm_misses_mm"]):
        print(f"  {obs.colour:5s} arm said ({obs.ground[0]:6.1f},{obs.ground[1]:7.1f}), "
              f"it is at ({desk[0]:6.1f},{desk[1]:7.1f}): {miss:.1f} mm")
    stretch = np.linalg.svd(result["desk_to_arm"][:2, :2], compute_uv=False)
    print(f"arm correction: stretch {stretch[0]:.3f} / {stretch[1]:.3f} (1 / 1 is an arm that "
          f"goes where it is sent)")
    if args.dry_run:
        return 0

    refined = json.loads(CONDOR.read_text())
    refined["model"] = "grid"
    desk_camera, correction = result["desk_camera"], result["desk_to_arm"]
    refined["pixel_to_arm"] = (correction @ np.linalg.inv(_plane_of(desk_camera, 0.0))).tolist()
    refined["pixel_to_arm_lifted"] = (correction @ np.linalg.inv(
        _plane_of(desk_camera, refined["lift_mm"]))).tolist()
    refined["desk_camera"] = desk_camera.tolist()
    refined["desk_to_arm"] = correction.tolist()
    refined["desk"] = ("desk_camera maps true millimetres on the desk (x, y, height) to pixels, "
                       "square pixels, from grid paper; desk_to_arm is where the arm must be sent "
                       "to reach a desk point, fitted through the three cubes")
    refined["grid"] = {"frame": args.frame.name, "square_mm": args.square_mm,
                       "focal_px": f, "principal_point": [cx, cy],
                       "correlation": lattice.correlation}
    CALIBRATION.write_text(json.dumps(refined, indent=2) + "\n")
    cv2.imwrite(str(CHECK), check_image(frame, lattice, DeskMapping.load(CALIBRATION)))
    print(f"wrote {CALIBRATION.relative_to(ROOT)} and {CHECK.relative_to(ROOT)}")
    return 0


def check_image(frame: np.ndarray, lattice: grid.Lattice, mapping: DeskMapping) -> np.ndarray:
    """Every tenth grid line as fitted (red), and the desk's own 20 mm grid (cyan), drawn thin."""
    drawn = frame.copy()
    centre = np.floor(lattice.to_squares(np.array([[frame.shape[1] / 2, frame.shape[0] / 2]]))[0])
    along = np.linspace(-40, 40, 400)
    for k in range(-40, 41, 10):
        for squares in (np.column_stack((np.full_like(along, k), along)),
                        np.column_stack((along, np.full_like(along, k)))):
            pixels = lattice.to_pixel(squares + centre)
            cv2.polylines(drawn, [np.round(pixels).astype(np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA)
    for value in np.arange(-160, 161, 20):
        for line in (np.column_stack((np.full(200, value), np.linspace(-320, -60, 200))),
                     np.column_stack((np.linspace(-160, 160, 200), np.full(200, -value - 190)))):
            pixels = mapping.world_to_pixel(np.column_stack((line, np.zeros(len(line)))))
            cv2.polylines(drawn, [np.round(pixels).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
    return drawn


if __name__ == "__main__":
    sys.exit(main())
