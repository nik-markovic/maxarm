"""CPCS to AACS: a pixel in the camera frame to a place on the arm's desk.

This is the thing the whole pilot exists to produce. `calibrate.py` solves one
and writes it to `config/calibration.json`; the demo loads it and asks it where
to send the arm for something it has just found in a frame.

    mapping = DeskMapping.load(Path("config/calibration.json"))
    x, y, z = mapping.pixel_to_board(tile.base_center, TILE_MM, seen_at_mm=TILE_MM)

## The two coordinate systems, and the third one in between

**CPCS**, camera pixel coordinates: what the detector reports. Pixels.

**AACS**, application arm coordinates: millimetres, x and y as the arm has them,
and **z measured up from the desk**. `z = 0` is the cup snug on the surface,
where it can seal against it; a 4 mm tile is gripped at `z = 4` and a 40 mm cube
at `z = 40`. This is the system the demo should think in, because it is the only
one in which "on the desk" is a number rather than a place-dependent reading.

**Board z**, which is what `MaxArm.move_to()` takes, is neither. The arm's
reported z drifts upward as it extends (`work/STATUS-condor.md` §8.2), so the
same desk reads 48 near the base and 36 at full reach.

**Those are two separate mappings and they are separate objects.** `DeskPlane`
is AACS to board and back -- three measured surface heights, no camera anywhere
in it, usable on its own by an app that already knows where it wants to go:

    desk = DeskPlane.through(CUBE_POSITIONS.values())
    desk.aacs_to_board((100.0, -254.0, 40.0))     # -> board z 78, the green drop

`DeskMapping` is the camera half, CPCS to AACS x and y, and it carries a
`DeskPlane` so that `pixel_to_board()` can do both steps at once. Nothing in
`maxarm/` knows AACS exists.

## What is fitted, and from what

**x and y** come from a camera: one 3x4 projection solved through every cube's
base centre *and* top centre, which are points at two known heights. Six of
them fix all eleven of its freedoms with one to spare, so three cubes give a
full perspective mapping. Without the tops, only a plane-to-plane transform is
possible, and three points make that affine -- see `fit()`.

**The AACS zero plane** comes from the heights in `CUBE_POSITIONS`: each one is
the owner's measurement of *the cup snug on the bare desk at that x and y*. Three
of them fix a plane exactly, so this interpolates the arm's z drift across the
work area rather than explaining it.

## Height, which is not just an offset

A pixel on its own does not name a point -- it names a *ray*. Where that ray
lands depends on how high the thing it shows is off the desk: the top face of a
4 mm tile is about 9 mm from its own base in this camera, at this elevation,
because the camera is looking along the desk rather than down at it.

So `pixel_to_ground()` takes the height of the feature being pointed at, and the
mapping is fitted on **two planes**: the cubes' base centres at AACS 0 and their
top centres at AACS 40. Between them it interpolates, and that interpolation is
exact rather than approximate -- the ray is a straight line, so where it crosses
`z = h` is linear in `h`. A 4 mm tile is a tenth of the way up that line, so what
it asks of the fit is an interpolation, not an extrapolation, and the error at
4 mm is a tenth of the error at 40.

Height enters twice and the two are different numbers:

- **where the feature is** -- 0 for a cube's base or a tile's outline on the
  desk, 4 for the top face of a 4 mm tile, which is what the camera actually
  sees when the tile is printed on top;
- **where the cup has to go** -- 40 to grip a cube, 4 to grip that tile.

`pixel_to_board(pixel, grip_height_mm, seen_at_mm)` takes both, neither with a
silently wrong default.

Nothing here talks to a camera or to the arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Pixel = Tuple[float, float]
Position = Tuple[float, float, float]

# Below this the fit is affine; at or above it, a perspective transform. A plane
# seen by a camera is a perspective transform and nothing less, but it has eight
# degrees of freedom and three point pairs supply six.
POINTS_FOR_PERSPECTIVE = 4

# A 3x4 camera has eleven freedoms; six points give twelve equations.
POINTS_FOR_CAMERA = 6


@dataclass(frozen=True)
class DeskPlane:
    """AACS to board z and back. The arm half of the calibration, no camera in it.

    `z = z0 + dz_dx * x + dz_dy * y` is where the cup is snug on the bare desk --
    AACS zero -- in the board's own millimetres. Everything AACS is that plus a
    height:

        desk = DeskPlane.through(CUBE_POSITIONS.values())
        arm.move_to(*desk.aacs_to_board((100.0, -254.0, 40.0)))    # board z 78

    which is the green cube's drop-off: 40 mm of cube standing on a surface that
    reads 38 there. **This needs no camera and no calibration run.** It is three
    heights the owner measured with the arm, and an app that already knows where
    it wants to go in AACS needs nothing else. The camera mapping in
    `DeskMapping` carries one of these because a pixel has to become x and y
    before this can be applied -- but the two are separable, and this is the half
    that does not depend on where the camera is standing.

    The tilt is mostly the arm's own z drift with reach rather than the desk's:
    the surface reads 36-38 under the two far cubes and 48 under the near one,
    which is 11 mm over 164 mm of reach on one flat desk (`work/STATUS-condor.md`
    §8.2). Three points cannot tell that apart from measurement noise, so treat
    it as an interpolation between heights that were each measured on the
    hardware -- not as a model of the arm.
    """

    z0: float
    dz_dx: float
    dz_dy: float

    def board_z(self, x: float, y: float, z_aacs: float = 0.0) -> float:
        """The board z for a height above the desk at this point of the work area.

        `board_z(x, y)` on its own is the surface: what the arm reads with the
        cup snug on the bare desk there.
        """
        return self.z0 + self.dz_dx * x + self.dz_dy * y + z_aacs

    def aacs_to_board(self, position: Position) -> Position:
        """(x, y, height above the desk) to what `MaxArm.move_to()` takes."""
        x, y, z_aacs = position
        return (x, y, self.board_z(x, y, z_aacs))

    def board_to_aacs(self, position: Position) -> Position:
        """The way back: a pose read off the arm, as a height above the desk."""
        x, y, z_board = position
        return (x, y, z_board - self.board_z(x, y))

    @classmethod
    def through(cls, positions: Sequence[Position]) -> "DeskPlane":
        """Fit it through measured surface heights: exact at three, lsq at more.

        Each position is `(x, y, board_z)` with the board z being the cup snug on
        the bare desk there -- which is exactly what `arm.CUBE_POSITIONS` holds.
        """
        if len(positions) < 3:
            raise ValueError(f"a plane needs three surface heights, have {len(positions)}")
        design = np.array([(1.0, x, y) for x, y, _ in positions])
        heights = np.array([z for _, _, z in positions])
        coefficients, *_ = np.linalg.lstsq(design, heights, rcond=None)
        return cls(float(coefficients[0]), float(coefficients[1]), float(coefficients[2]))

    def as_dict(self) -> dict:
        return {"z0": self.z0, "dz_dx": self.dz_dx, "dz_dy": self.dz_dy}

    @classmethod
    def from_dict(cls, data: dict) -> "DeskPlane":
        return cls(data["z0"], data["dz_dx"], data["dz_dy"])


@dataclass(frozen=True)
class Observation:
    """One point pair: a feature in CPCS, and where the arm knows it to be.

    `arm` is `(x, y, board_z)` -- the third number being the board z at which the
    cup is snug on the bare desk at that x and y, which is AACS zero there. It is
    not where the cube's top is and not where the cup went to place it.

    `height_mm` is how high off the desk the thing in the pixel is: 0 for a
    cube's base centre, one cube-height for its top centre. Same cube, same x and
    y, two different pixels -- which is the whole of how the mapping learns what
    height does to a pixel.
    """

    colour: str
    pixel: Pixel
    arm: Position
    height_mm: float = 0.0

    @property
    def ground(self) -> Tuple[float, float]:
        return (self.arm[0], self.arm[1])


@dataclass(frozen=True)
class DeskMapping:
    model: str                              # "camera", "perspective" or "affine"
    matrix: np.ndarray                      # 3x3, homogeneous CPCS -> AACS x, y
    plane: DeskPlane                        # AACS <-> board z. Usable on its own
    observations: Tuple[Observation, ...]
    lifted: Optional[np.ndarray] = None     # the same, for the plane at lift_mm
    lift_mm: float = 0.0                    # the height that second plane is at

    def pixel_to_aacs(self, pixel: Pixel, seen_at_mm: float = 0.0) -> Position:
        """Where the thing at this pixel is, given how high off the desk it is.

        `seen_at_mm` is the height of the *feature in the image*, not of the cup:
        0 for something lying on the desk, 4 for the printed face of a 4 mm tile.
        The z that comes back is that same height, because that is where the
        thing is.
        """
        x, y = self.pixel_to_ground(pixel, seen_at_mm)
        return (x, y, seen_at_mm)

    def pixel_to_board(self, pixel: Pixel, grip_height_mm: float,
                       seen_at_mm: float = 0.0) -> Position:
        """CPCS straight to what `MaxArm.move_to()` takes. The call the demo makes.

        Both halves in one: the pixel becomes AACS x and y, then the desk plane
        turns a height into a board z. The two are separable and
        `plane.aacs_to_board()` is the second half on its own, for an app that
        already knows where it wants to go.

        Two heights, because they are two different things. `seen_at_mm` is how
        high off the desk the feature in the pixel is -- 0 for a cube's base,
        4 for the top face of a 4 mm tile -- and it decides *where* the thing is.
        `grip_height_mm` is how high the cup has to be to seal on its top face,
        which is the piece's own height, and it decides where the cup goes.

        For a cube found by its base they are 0 and 40. For a tile found by its
        printed top they are both 4. Neither is guessed from the other.
        """
        x, y = self.pixel_to_ground(pixel, seen_at_mm)
        return self.plane.aacs_to_board((x, y, grip_height_mm))

    def aacs_to_board(self, position: Position) -> Position:
        """The desk plane's, for callers holding a mapping rather than a plane."""
        return self.plane.aacs_to_board(position)

    def board_to_aacs(self, position: Position) -> Position:
        return self.plane.board_to_aacs(position)

    def board_z(self, x: float, y: float, z_aacs: float = 0.0) -> float:
        return self.plane.board_z(x, y, z_aacs)

    def pixel_to_ground(self, pixel: Pixel, seen_at_mm: float = 0.0) -> Tuple[float, float]:
        """CPCS to AACS x and y, for a feature `seen_at_mm` above the desk.

        A pixel is a ray, and a ray is a straight line, so where it crosses
        `z = h` is linear in `h`. Two solved planes therefore give every height
        exactly -- this interpolates between the desk and the cube-top plane, and
        the same arithmetic extrapolates above the cubes, with the usual warning
        that extrapolation believes the fit further than it has been checked.
        """
        ground = _project(self.matrix, pixel)
        if not seen_at_mm:
            return ground
        if self.lifted is None:
            raise ValueError(
                f"this calibration only solved the desk plane, so it cannot place "
                f"something {seen_at_mm:g} mm above it. Re-run calibrate.py: the cube "
                f"tops are what carry the height, and they come out of the same frame")
        lifted = _project(self.lifted, pixel)
        fraction = seen_at_mm / self.lift_mm
        return (ground[0] + fraction * (lifted[0] - ground[0]),
                ground[1] + fraction * (lifted[1] - ground[1]))

    def parallax_mm(self, pixel: Pixel, height_mm: float) -> float:
        """How far a feature at this height sits from its own footprint, in AACS.

        The number that says whether height can be ignored. At this camera's
        elevation it is around 9 mm for a 4 mm tile, which is more than the arm's
        own accuracy -- so it cannot.
        """
        base = self.pixel_to_ground(pixel, 0.0)
        lifted = self.pixel_to_ground(pixel, height_mm)
        return float(np.hypot(lifted[0] - base[0], lifted[1] - base[1]))

    def camera(self) -> np.ndarray:
        """The 3x4 projection, AACS (x, y, height, 1) -> homogeneous pixel.

        Not stored, and it does not need to be: both planes are exact inverses of
        `_plane_of()` at 0 and `lift_mm`, which share their x and y columns, so the
        height column is their third columns' difference over `lift_mm`.
        """
        if self.lifted is None or not self.lift_mm:
            raise ValueError("this calibration solved only the desk plane; a camera needs the "
                             "cube tops as well. Re-run calibrate.py")
        desk, lifted = np.linalg.inv(self.matrix), np.linalg.inv(self.lifted)
        return np.column_stack((desk[:, 0], desk[:, 1],
                                (lifted[:, 2] - desk[:, 2]) / self.lift_mm, desk[:, 2]))

    def camera_position(self) -> np.ndarray:
        """Where the camera stands, AACS (x, y, height): the 3x4's null vector."""
        _, _, vectors = np.linalg.svd(self.camera())
        centre = vectors[-1]
        return centre[:3] / centre[3]

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """AACS (x, y, height) rows to pixel rows."""
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ self.camera().T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def plane_to_pixel(self, height_mm: float) -> np.ndarray:
        """The 3x3 homography taking (x, y, 1) on the plane `height_mm` up to a pixel."""
        if not height_mm:
            return np.linalg.inv(self.matrix)
        return _plane_of(self.camera(), height_mm)

    def ground_to_pixel(self, ground: Tuple[float, float], height_mm: float = 0.0) -> Pixel:
        """The way back, for drawing over a frame, at any height the camera allows."""
        point = self.plane_to_pixel(height_mm) @ np.array([ground[0], ground[1], 1.0])
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    def is_inside(self, pixel: Pixel) -> bool:
        """Whether this pixel is inside the patch of desk the fit was made from.

        Outside it the mapping is extrapolating, which is where an affine fit
        parts company with the camera's perspective fastest. Not an error --
        a cube a little outside the triangle is still worth reaching for -- but
        it is the first thing to suspect when one is missed.
        """
        # The desk points only. The cube tops are pixels well above the surface,
        # and letting them into the hull would claim a patch of the image the
        # calibration has nothing on.
        corners = np.array([obs.pixel for obs in self.observations if not obs.height_mm],
                           dtype=np.float32)
        hull = cv2.convexHull(corners.reshape(-1, 1, 2))
        # A pixel a whisker outside counts as inside: the calibration points are
        # themselves on the hull, and float32 rounding in the hull puts them a
        # fraction of a pixel out, which is not a reason to warn about one.
        return cv2.pointPolygonTest(hull, (float(pixel[0]), float(pixel[1])), True) >= -1.0

    def residuals_mm(self) -> List[float]:
        """How far the fit misses each point it was fitted through.

        Zero by construction whenever the fit has exactly as many points as it
        has freedoms, which is both of the cases that matter: three points fit an
        affine transform exactly, and a fourth point does not help because it
        switches the fit to a perspective one that then fits those four exactly
        too. The first informative residual arrives with a fifth point, and
        `held_out_mm()` is the honest number even then.
        """
        return [float(np.hypot(*(np.array(self.pixel_to_ground(obs.pixel, obs.height_mm))
                                 - np.array(obs.ground))))
                for obs in self.observations]

    def held_out_mm(self) -> Optional[List[float]]:
        """Refit without each point in turn and see how far off that point lands.

        This is the millimetre figure that decides whether a side camera can
        drive the demo, and it needs one point more than the fit consumes:
        five for a perspective fit, four for an affine one. Returns None when
        there are too few, rather than a number that only measures itself.

        A point whose refit collapses is left out of the answer rather than
        ending the run: the ones left behind happen to lie in a line, or there
        are only two of them left on that plane. That is a property of the
        subset, not a fault in the calibration, and it is also how this comes
        back None when there is nothing to hold out -- every refit collapses.
        """
        errors = []
        for index, held in enumerate(self.observations):
            rest = self.observations[:index] + self.observations[index + 1:]
            try:
                trial = fit(rest)
            except (ValueError, np.linalg.LinAlgError):
                continue
            errors.append(float(np.hypot(
                *(np.array(trial.pixel_to_ground(held.pixel, held.height_mm))
                  - np.array(held.ground)))))
        return errors or None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")

    def as_dict(self) -> dict:
        return {
            # One line, because this file is the handoff to the demo and to
            # whatever runs on the IMX95, and it will be read by someone who
            # does not have this module open.
            "about": "camera pixels (CPCS) to arm millimetres (AACS). "
                     "[x, y, w] = pixel_to_arm @ [px, py, 1]; x, y = x/w, y/w. "
                     "AACS z is height above the desk, 0 = cup snug on the bare "
                     "surface. board z = z_aacs + aacs_zero_plane(x, y), and "
                     "board z is what MaxArm.move_to takes.",
            "height": "a pixel is a ray, so a feature h mm above the desk lands "
                      "elsewhere: solve it on both planes and interpolate, "
                      "ground + (h / lift_mm) * (lifted - ground). Exact, because "
                      "a ray is straight.",
            "model": self.model,
            "pixel_to_arm": self.matrix.tolist(),
            "pixel_to_arm_lifted": None if self.lifted is None else self.lifted.tolist(),
            "lift_mm": self.lift_mm,
            "aacs_zero_plane": self.plane.as_dict(),
            "points": [{"colour": obs.colour, "pixel": list(obs.pixel), "arm": list(obs.arm),
                        "height_mm": obs.height_mm}
                       for obs in self.observations],
        }

    @classmethod
    def load(cls, path: Path) -> "DeskMapping":
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "DeskMapping":
        lifted = data.get("pixel_to_arm_lifted")
        return cls(
            model=data["model"],
            matrix=np.array(data["pixel_to_arm"], dtype=np.float64),
            plane=DeskPlane.from_dict(data["aacs_zero_plane"]),
            observations=tuple(
                Observation(point["colour"], tuple(point["pixel"]), tuple(point["arm"]),
                            point.get("height_mm", 0.0))
                for point in data["points"]
            ),
            lifted=None if lifted is None else np.array(lifted, dtype=np.float64),
            lift_mm=data.get("lift_mm", 0.0),
        )


def fit(observations: Sequence[Observation]) -> DeskMapping:
    """Solve the mapping from every point pair given. Three at the least.

    **With the cube tops, it is a camera.** Each cube gives two points at the
    same x and y and two known heights, so the points are not all on one plane,
    and a 3x4 projection -- a real camera, perspective and all -- can be solved
    through them: `_solve_camera()`. Both planes are then read off that one
    projection, so they agree with each other exactly and each is perspective,
    from three cubes and one frame. Measured on the owner's frame, the cubes'
    base edges -- which the fit never saw -- come out 38.6-43.1 mm through it,
    where the per-plane affine fit put them at 28-68.

    **Without the tops, four point pairs per plane is the line that matters.** The camera looks
    along the desk at about 23 degrees, and at that angle the scale across one
    frame swings by 2.7x -- 0.17 mm/px at the near cube against 0.75 mm/px
    receding at the far one (`work/STATUS-condor-prototype.md` §6). Only a
    perspective transform carries that, and it needs a fourth pair. With three
    the fit is affine: exact at the three cubes, and drifting away from them at a
    rate nothing in the fit can report. That is why `calibrate.py` prints the
    cube edge check -- it is the one number that catches an affine fit failing.

    A fourth pair costs one more cube placement at a coordinate the arm chose,
    photographed with the camera untouched; `calibrate.py --keep` adds it to the
    points already in the config.

    Observations above the desk -- the cube tops -- are solved as their own
    plane, which is what lets a pixel be turned into a position for something
    that is not lying flat on the surface. They are not mixed in with the desk
    points: the two planes are different transforms and averaging them would be
    a mapping of neither.
    """
    desk = [obs for obs in observations if not obs.height_mm]
    lifted = [obs for obs in observations if obs.height_mm]
    if len(desk) < 3:
        raise ValueError(f"need at least three point pairs on the desk, have {len(desk)}")

    lift_matrix, lift_mm = None, 0.0
    if lifted:
        heights = {obs.height_mm for obs in lifted}
        if len(heights) > 1:
            raise ValueError(f"the lifted points are at {len(heights)} different heights "
                             f"({sorted(heights)}); this solves one plane above the desk, "
                             f"not several")
        lift_mm = lifted[0].height_mm

    if lifted and len(lifted) >= 3:
        camera = _solve_camera(desk + lifted)
        matrix, lift_matrix = (np.linalg.inv(_plane_of(camera, height)) for height in (0.0, lift_mm))
        model = "camera"
    else:
        matrix, model = _solve(desk)
        if lifted:
            lift_matrix, _ = _solve(lifted)

    return DeskMapping(model=model, matrix=matrix,
                       plane=DeskPlane.through([obs.arm for obs in desk]),
                       observations=tuple(observations),
                       lifted=lift_matrix, lift_mm=lift_mm)


def _solve(observations: Sequence[Observation]) -> Tuple[np.ndarray, str]:
    """One plane: the transform through these pixels, and which kind it is."""
    if len(observations) < 3:
        raise ValueError(f"need at least three point pairs, have {len(observations)}")
    pixels = np.array([obs.pixel for obs in observations], dtype=np.float64)
    ground = np.array([obs.ground for obs in observations], dtype=np.float64)

    if len(observations) >= POINTS_FOR_PERSPECTIVE:
        matrix, _ = cv2.findHomography(pixels.reshape(-1, 1, 2), ground.reshape(-1, 1, 2), 0)
        if matrix is None:
            raise ValueError("the point pairs do not determine a perspective transform; "
                             "three collinear cubes will do this")
        model = "perspective"
    else:
        # x and y are solved together against [px, py, 1]: the same six numbers
        # OpenCV's getAffineTransform would give for exactly three points, but
        # least-squares, so a fourth affine point would still be usable.
        design = np.column_stack((pixels, np.ones(len(pixels))))
        coefficients, *_ = np.linalg.lstsq(design, ground, rcond=None)
        matrix = np.vstack((coefficients.T, (0.0, 0.0, 1.0)))
        model = "affine"

    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("the fit collapsed -- the cubes are probably collinear in the image")
    return np.asarray(matrix, dtype=np.float64), model


def _solve_camera(observations: Sequence[Observation]) -> np.ndarray:
    """The 3x4 projection taking (x, y, height above the desk) to a pixel.

    The direct linear transform, on coordinates shifted and scaled to the unit
    order first -- raw pixels in the thousands next to millimetres in the tens
    leave the linear system too badly conditioned to trust.
    """
    if len(observations) < POINTS_FOR_CAMERA:
        raise ValueError(f"a camera needs {POINTS_FOR_CAMERA} points, have {len(observations)}")
    pixels = np.array([obs.pixel for obs in observations], dtype=np.float64)
    world = np.array([(*obs.ground, obs.height_mm) for obs in observations], dtype=np.float64)
    pixel_norm, world_norm = _normaliser(pixels), _normaliser(world)
    pixels = (pixel_norm @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
    world = (world_norm @ np.column_stack((world, np.ones(len(world)))).T).T

    rows = []
    for point, (u, v, _) in zip(world, pixels):
        rows.append(np.concatenate((point, np.zeros(4), -u * point)))
        rows.append(np.concatenate((np.zeros(4), point, -v * point)))
    _, singular, vectors = np.linalg.svd(np.array(rows))
    # The answer is the null direction. A second one nearly as null means the
    # points leave the camera undetermined -- all on one plane, or in a line.
    if singular[-2] < 1e-6 * singular[0]:
        raise ValueError("the points do not determine a camera; the cubes may be in a line")
    camera = np.linalg.inv(pixel_norm) @ vectors[-1].reshape(3, 4) @ world_norm
    return camera / np.linalg.norm(camera)


def _normaliser(points: np.ndarray) -> np.ndarray:
    """Shift to the centroid and scale to unit mean distance, as a homogeneous matrix."""
    centre = points.mean(axis=0)
    scale = np.sqrt(points.shape[1]) / max(np.linalg.norm(points - centre, axis=1).mean(), 1e-12)
    matrix = np.eye(points.shape[1] + 1) * scale
    matrix[:-1, -1] = -scale * centre
    matrix[-1, -1] = 1.0
    return matrix


def _plane_of(camera: np.ndarray, height_mm: float) -> np.ndarray:
    """What the camera does to the plane `height_mm` above the desk: pixel <- x, y."""
    return np.column_stack((camera[:, 0], camera[:, 1], camera[:, 3] + height_mm * camera[:, 2]))


def _project(matrix: np.ndarray, pixel: Pixel) -> Tuple[float, float]:
    point = matrix @ np.array([pixel[0], pixel[1], 1.0])
    return (float(point[0] / point[2]), float(point[1] / point[2]))
