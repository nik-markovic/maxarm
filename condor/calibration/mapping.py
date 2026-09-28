"""The desk mapping: a pixel in the camera frame to a place on the arm's desk.

This is the thing the whole pilot exists to produce. `calibrate.py` solves one
and writes it to `config/calibration.json`; the demo loads it and asks it where
to send the arm for a cube it has just found in a frame.

    mapping = DeskMapping.load(Path("config/calibration.json"))
    x, y, z = mapping.to_grapple(detection.base_center, CUBE_SIZE_MM)

Two pieces, fitted from the same handful of point pairs:

**x, y** come from a plane-to-plane transform. Three point pairs is exactly
enough for an affine one and one short of a perspective one, which is the whole
accuracy story here -- see `fit()`.

**z** comes from a plane through the heights in `CUBE_POSITIONS`, and those are
the height of the *desk under each cube*, not a pose for the cup: the arm sets
the far cubes down with the cup at 78 and their 40 mm bodies stand on a desk that
reads 38 there. So the plane is where the desk is as the arm sees it, and the cup
goes one object-height above it -- `to_grapple()`. The plane is not flat because
the arm's reported z drifts upward as it extends (`work/STATUS-condor.md` §8.2);
three heights fix a plane exactly, so this interpolates that drift across the
work area rather than explaining it.

**The object's height is an argument, never a default.** A 40 mm cube and a 4 mm
letter tile sit on the same desk plane and are grappled 36 mm apart.

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


@dataclass(frozen=True)
class Observation:
    """One point pair: a cube's base in the image, and where the arm put it."""

    colour: str
    pixel: Pixel
    arm: Position

    @property
    def ground(self) -> Tuple[float, float]:
        return (self.arm[0], self.arm[1])


@dataclass(frozen=True)
class DeskMapping:
    model: str                              # "affine" or "perspective"
    matrix: np.ndarray                      # 3x3, homogeneous pixel -> arm x, y
    plane: Tuple[float, float, float]       # desk: z = z0 + dz_dx * x + dz_dy * y
    observations: Tuple[Observation, ...]

    def to_arm(self, pixel: Pixel) -> Position:
        """Where the base of whatever sits at this pixel is, in arm millimetres.

        The desk, in other words -- an object's footprint, not a pose the cup can
        be sent to. `to_grapple()` is the one to drive the arm with.
        """
        x, y = self.to_ground(pixel)
        return (x, y, self.desk_z(x, y))

    def to_grapple(self, pixel: Pixel, object_height_mm: float) -> Position:
        """Where to put the cup to pick up an object standing at this pixel.

        The cup grips a flat top face, so this is the desk plus how tall the
        object is. The height is not optional and not defaulted: the cubes are
        40 mm and the letter tiles that come later are 4 mm, and taking one for
        the other drives the cup 36 mm into the desk or leaves it 36 mm short.
        """
        x, y = self.to_ground(pixel)
        return (x, y, self.desk_z(x, y) + object_height_mm)

    def to_ground(self, pixel: Pixel) -> Tuple[float, float]:
        point = self.matrix @ np.array([pixel[0], pixel[1], 1.0])
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    def to_pixel(self, ground: Tuple[float, float]) -> Pixel:
        """The way back, for drawing the fit over a frame and looking at it."""
        point = np.linalg.inv(self.matrix) @ np.array([ground[0], ground[1], 1.0])
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    def desk_z(self, x: float, y: float) -> float:
        """Where the desk reads, in the arm's own z, at this point of the work area."""
        z0, dz_dx, dz_dy = self.plane
        return z0 + dz_dx * x + dz_dy * y

    def is_inside(self, pixel: Pixel) -> bool:
        """Whether this pixel is inside the patch of desk the fit was made from.

        Outside it the mapping is extrapolating, which is where an affine fit
        parts company with the camera's perspective fastest. Not an error --
        a cube a little outside the triangle is still worth reaching for -- but
        it is the first thing to suspect when one is missed.
        """
        corners = np.array([obs.pixel for obs in self.observations], dtype=np.float32)
        hull = cv2.convexHull(corners.reshape(-1, 1, 2))
        return cv2.pointPolygonTest(hull, (float(pixel[0]), float(pixel[1])), False) >= 0

    def residuals_mm(self) -> List[float]:
        """How far the fit misses each point it was fitted through.

        Zero by construction whenever the fit has exactly as many points as it
        has freedoms, which is both of the cases that matter: three points fit an
        affine transform exactly, and a fourth point does not help because it
        switches the fit to a perspective one that then fits those four exactly
        too. The first informative residual arrives with a fifth point, and
        `held_out_mm()` is the honest number even then.
        """
        return [float(np.hypot(*(np.array(self.to_ground(obs.pixel)) - np.array(obs.ground))))
                for obs in self.observations]

    def held_out_mm(self) -> Optional[List[float]]:
        """Refit without each point in turn and see how far off that point lands.

        This is the millimetre figure that decides whether a side camera can
        drive the demo, and it needs one point more than the fit consumes:
        five for a perspective fit, four for an affine one. Returns None when
        there are too few, rather than a number that only measures itself.

        A point whose refit collapses -- the ones left behind happening to lie
        in a line -- is left out of the answer rather than ending the run. It is
        a property of that subset, not a fault in the calibration.
        """
        if len(self.observations) <= _degrees_of_freedom(len(self.observations)):
            return None
        errors = []
        for index, held in enumerate(self.observations):
            rest = self.observations[:index] + self.observations[index + 1:]
            try:
                trial = fit(rest)
            except (ValueError, np.linalg.LinAlgError):
                continue
            errors.append(float(np.hypot(
                *(np.array(trial.to_ground(held.pixel)) - np.array(held.ground)))))
        return errors or None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")

    def as_dict(self) -> dict:
        z0, dz_dx, dz_dy = self.plane
        return {
            "model": self.model,
            "pixel_to_arm": self.matrix.tolist(),
            "desk_plane": {"z0": z0, "dz_dx": dz_dx, "dz_dy": dz_dy},
            "points": [{"colour": obs.colour, "pixel": list(obs.pixel), "arm": list(obs.arm)}
                       for obs in self.observations],
        }

    @classmethod
    def load(cls, path: Path) -> "DeskMapping":
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "DeskMapping":
        plane = data["desk_plane"]
        return cls(
            model=data["model"],
            matrix=np.array(data["pixel_to_arm"], dtype=np.float64),
            plane=(plane["z0"], plane["dz_dx"], plane["dz_dy"]),
            observations=tuple(
                Observation(point["colour"], tuple(point["pixel"]), tuple(point["arm"]))
                for point in data["points"]
            ),
        )


def fit(observations: Sequence[Observation]) -> DeskMapping:
    """Solve the mapping from every point pair given. Three at the least.

    **Four point pairs is the line that matters.** The camera looks along the
    desk at about 23 degrees, and at that angle the scale across one frame swings
    by 2.7x -- 0.17 mm/px at the near cube against 0.75 mm/px receding at the far
    one (`work/STATUS-condor-prototype.md` §6). Only a perspective transform
    carries that, and it needs a fourth pair. With three the fit is affine: exact
    at the three cubes, and drifting away from them at a rate nothing in the fit
    can report. That is why `calibrate.py` prints the cube edge check -- it is
    the one number that catches an affine fit failing.

    A fourth pair costs one more cube placement at a coordinate the arm chose,
    photographed with the camera untouched; `calibrate.py --keep` adds it to the
    points already in the config.
    """
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
    return DeskMapping(model=model, matrix=np.asarray(matrix, dtype=np.float64),
                       plane=fit_plane(observations), observations=tuple(observations))


def fit_plane(observations: Sequence[Observation]) -> Tuple[float, float, float]:
    """z = z0 + dz_dx * x + dz_dy * y through the known desk heights.

    Exact through three, least-squares through more. The tilt it finds is mostly
    the arm's own z drift with reach rather than the desk's: on the owner's arm
    the desk reads 36-38 under the two far cubes and 48 under the near one, which
    is 11 mm over 164 mm of reach on one flat desk. Three points cannot tell that
    apart from measurement noise, so treat the plane as an interpolation between
    heights that were each checked on the hardware -- not as a model of the arm.
    """
    design = np.array([(1.0, obs.arm[0], obs.arm[1]) for obs in observations])
    heights = np.array([obs.arm[2] for obs in observations])
    coefficients, *_ = np.linalg.lstsq(design, heights, rcond=None)
    return (float(coefficients[0]), float(coefficients[1]), float(coefficients[2]))


def _degrees_of_freedom(point_count: int) -> int:
    """Point pairs the fit consumes, in point-pair units: 4 perspective, 3 affine."""
    return POINTS_FOR_PERSPECTIVE if point_count >= POINTS_FOR_PERSPECTIVE else 3
