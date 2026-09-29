#!/usr/bin/env python3
"""Solve the CPCS-to-AACS mapping from the cubes the arm has just put down.

Run `arm.py` first: it takes three cubes off a stack and places them at
`CUBE_POSITIONS`, which is the one thing a photograph cannot tell you -- where
something on the desk is in *arm* millimetres. This takes a snapshot, finds the
three cubes in it, pairs each one's base with the coordinate the arm placed it
at, and solves the transform between the two. The result lands in
`config/calibration.json`, and the demo loads it to turn a piece it has spotted
into a pose for the cup.

Camera pixels are CPCS; arm millimetres with z measured up from the desk are
AACS, in which 0 is the cup snug on the surface and a 4 mm tile is gripped at 4.
`mapping.py` is where both are defined and is worth reading first.

    ./calibration/calibrate.py                     # snapshot, fit, write the config
    ./calibration/calibrate.py --image FRAME.jpg   # fit from a frame taken earlier
    ./calibration/calibrate.py --dry-run           # fit, print, write nothing

The camera and the arm must not move relative to each other after this, and the
cubes must be on the desk -- not on the stack they arrive in. A cube standing on
another cube has its base 40 mm up in the air and maps to a point 40 mm from
where it looks like it is.

**Three cubes are enough for a camera.** Each cube's base centre and top centre
are two points at known heights, and six such points solve a full 3x4
projection -- perspective, not affine; see `mapping.fit()`. The fit has only one
equation to spare, so its residuals say little; the cube edge check below, which
uses corners the fit never saw, is what judges it. A held-out error in
millimetres needs more points: place one cube somewhere else with the arm, leave
the camera alone, and add that frame to the points already in the config:

    ./calibration/calibrate.py --keep --at red=120,-180,44
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

import detect                                                         # noqa: E402
from arm import CUBE_POSITIONS, CUBE_SIZE_MM, Position                # noqa: E402
from mapping import DeskMapping, Observation, fit                     # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "config"
CALIBRATION = CONFIG / "calibration.json"
# Lossless: this is the frame a later run re-fits from, and JPEG moves the
# detected base centres by a pixel or so, which is a millimetre on the desk.
SNAPSHOT = CONFIG / "calibration-frame.png"
OVERLAY = CONFIG / "calibration-check.jpg"

DEFAULT_DEVICE = "/dev/video1"

# Both formats are tried and the bigger frame wins, uncompressed breaking a tie.
# Asking for an impossible size is how a V4L2 device is made to name its largest:
# it clamps the request to what it has. That beats a table of known cameras --
# the C920 on the bench does 2304x1536 in YUYV only while the one that replaced
# it does 2592x1944 in MJPG only, and the demo ends up on the IMX95 with neither.
CAPTURE_FORMATS = ("YUYV", "MJPG")
LARGER_THAN_ANY_SENSOR = 10000
SMALLEST_USEFUL_FRAME = 640 * 480

# The PILOT asks for the longest exposure available. It is the wrong lever here:
# the lit faces already saturate, and pushing more of the cube into saturation
# blurs the boundary this whole thing is measuring. What a static scene does want
# is the camera's own auto-exposure settled -- it arrives a few frames late after
# the device is opened -- and then several frames averaged, which costs nothing
# and takes sensor noise straight out of the mask edge.
SETTLE_FRAMES = 8
AVERAGED_FRAMES = 5

# Grid drawn over the check image, in arm millimetres: how the mapping thinks the
# desk lies under the camera. Wrong perspective shows up here long before it
# shows up in a residual.
# The work area around the three cubes, in AACS. Fixed rather than grown from
# the points, so the check image frames the same patch of desk every time.
GRID_X_MM = (-120.0, 120.0)
GRID_Y_MM = (-260.0, -80.0)
GRID_STEP_MM = 20.0
GRID_BGR = (255, 255, 0)        # cyan, against the detector's magenta wireframes
LIFTED_GRID_BGR = (0, 230, 255)  # yellow: the same millimetres one cube-height up
LIFTED_GRID_ALPHA = 0.5          # see-through, so the desk grid and cubes show under it


def main() -> int:
    args = parse_args()
    try:
        image, source = read_frame(args)
    except RuntimeError as error:
        print(f"no frame: {error}")
        return 1
    print(f"frame {image.shape[1]}x{image.shape[0]} from {source}")

    detections = detect.find_cubes(image, sorted(detect.COLOUR_WINDOWS))
    placements = cube_placements(args.at)
    pairs, missing = pair_up(detections, placements)
    for colour in missing:
        print(f"  no {colour} cube found -- it is not in the frame, it is turned face-on to "
              f"the camera, or the light has moved (see work/STATUS-condor-prototype.md §4)")
    observations = [observation for observation, _ in pairs]
    if args.keep and CALIBRATION.exists():
        observations = kept(observations)
    elif args.keep:
        print(f"  --keep, but there is no {CALIBRATION.name} yet -- this frame is all there is")
    if len(observations) < 3:
        print(f"\n{len(observations)} point pair(s); the fit needs three")
        return 1

    try:
        mapping = fit(observations)
    except ValueError as error:
        print(f"\nno mapping: {error}")
        return 1
    report(mapping, {id(observation): detection for observation, detection in pairs})

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    mapping.save(CALIBRATION)
    cv2.imwrite(str(SNAPSHOT), image)
    cv2.imwrite(str(OVERLAY), check_image(image, detections, mapping))
    print(f"\nwrote {CALIBRATION.name}, {SNAPSHOT.name} and {OVERLAY.name} to config/")
    print(f"look at config/{OVERLAY.name}: the grid is where the mapping thinks the arm's "
          f"millimetres are")
    return 0


def kept(observations: Sequence[Observation]) -> List[Observation]:
    """This frame's point pairs, added to the ones the existing config holds.

    A cube that has not moved since is dropped rather than added twice. Its
    second pixel carries almost no information -- the cube is in the same place
    -- but it would quietly double that point's weight in the fit, and worse,
    `held_out_mm()` would hold out one copy and have the other predict it. A
    held-out error that flatters itself is worse than none.
    """
    earlier = list(DeskMapping.load(CALIBRATION).observations)
    # Keyed by place *and* height: a cube contributes a point on the desk and one
    # on the cube-top plane, and those are not duplicates of each other.
    already = {(observation.ground, observation.height_mm) for observation in earlier}
    fresh = [observation for observation in observations
             if (observation.ground, observation.height_mm) not in already]
    print(f"  keeping {len(earlier)} point pair(s) from the existing calibration; "
          f"{len(fresh)} of this frame's {len(observations)} are new placements")
    return earlier + fresh


def report(mapping: DeskMapping, detections: Dict[int, detect.CubeDetection]) -> None:
    """Everything the fit can be judged by, and a straight word about what it cannot.

    `detections` is keyed by the identity of the observation it came from, not by
    colour: with `--keep` the same cube appears twice, once where it used to be
    and once where the arm has just moved it, and only one of those rows has a
    cube in this frame to measure.
    """
    desk_points = [obs for obs in mapping.observations if not obs.height_mm]
    print(f"\n{mapping.model} fit over {len(desk_points)} point pair(s) on the desk"
          f"{'' if mapping.lifted is None else f', and {len(mapping.observations) - len(desk_points)} on the cube-top plane'}\n")
    print("  cube    seen at   pixel                arm mm                 base edges by the fit")
    residuals = mapping.residuals_mm()
    for obs, residual in zip(mapping.observations, residuals):
        detection = detections.get(id(obs))
        edges = edge_lengths_mm(mapping, detection) if detection and not obs.height_mm else None
        edge_note = ("  ".join(f"{length:5.1f}" for length in edges) + f"  (want {CUBE_SIZE_MM:.0f})"
                     if edges else "" if detection else "from an earlier frame")
        seen = "desk   " if not obs.height_mm else f"+{obs.height_mm:<6.0f}"
        print(f"  {obs.colour:6}  {seen}  ({obs.pixel[0]:7.1f},{obs.pixel[1]:7.1f})  "
              f"({obs.arm[0]:6.1f},{obs.arm[1]:7.1f},{obs.arm[2]:5.1f})  {edge_note}")

    print(f"\n  residual at each point: {'  '.join(f'{value:.2f}' for value in residuals)} mm")
    if mapping.model == "camera" and len(mapping.observations) <= 6:
        print("  -- near zero because a camera through six points has one equation to "
              "spare, so this says little")
    elif max(residuals) < 0.01:
        print("  -- which is zero by construction: the fit has as many points as it has "
              "freedoms, so this measures nothing")
    held_out = mapping.held_out_mm()
    if held_out is None:
        print("  held-out error: needs one more point pair than this fit uses. "
              "`arm.py --move`, then `calibrate.py --keep --at <colour>=x,y,z`")
    else:
        short = ("" if len(held_out) == len(mapping.observations) else
                 f" ({len(held_out)} of {len(mapping.observations)} points; "
                 f"the rest left a fit with nothing to solve)")
        print(f"  held-out error: {'  '.join(f'{value:.1f}' for value in held_out)} mm "
              f"-- worst {max(held_out):.1f} mm{short}. This is the number that decides "
              f"whether a side camera can drive the demo")

    print(f"\n  The base edge check is the local one: a cube's base is {CUBE_SIZE_MM:.0f} mm "
          f"square wherever it stands,\n  so the fit should make it {CUBE_SIZE_MM:.0f} mm "
          f"square there.")
    if mapping.model == "camera":
        print("  The fit used the cubes' centres and never their corners, so this is an "
              "independent check.")
    if mapping.model == "affine":
        print("  It is also the only check a three-point fit has, since the residuals above "
              "are not\n  one. An affine fit cannot follow this camera's perspective, and the "
              "far cube is where\n  that shows first -- `arm.py --move` is the way to a "
              "fourth point pair.")

    if mapping.lifted is not None:
        middle = tuple(np.mean([obs.pixel for obs in desk_points], axis=0))
        full = mapping.parallax_mm(middle, mapping.lift_mm)
        print(f"\n  Height, in the middle of the calibrated patch: a feature "
              f"{mapping.lift_mm:.0f} mm above the\n  desk maps {full:.0f} mm from its own "
              f"footprint, and that scales straight with height --\n  {full / 10:.1f} mm for "
              f"a 4 mm piece. That is the camera looking along the desk rather than\n  down at "
              f"it, and it is why a pixel needs a height before it means a position.")

    desk = mapping.plane
    near, far = (0.0, -90.0), (0.0, -254.0)
    print(f"\n  AACS zero -- the cup snug on the bare desk -- in board z:"
          f"\n    {desk.z0:.1f} {desk.dz_dx:+.4f}*x {desk.dz_dy:+.4f}*y"
          f"   ({desk.board_z(*near):.1f} at the near cube, "
          f"{desk.board_z(*far):.1f} at the far ones)")
    print(f"  a piece of height h is gripped at AACS z = h, so a {CUBE_SIZE_MM:.0f} mm cube goes "
          f"to board z {desk.board_z(*near, CUBE_SIZE_MM):.0f} near and "
          f"{desk.board_z(*far, CUBE_SIZE_MM):.0f} far,\n  which is what the scene commands. "
          f"The surface reads lower at reach because the arm's\n  own z reads high there "
          f"(work/STATUS-condor.md §8.2).\n  That half is `DeskPlane` and needs no camera: "
          f"`DeskPlane.through(CUBE_POSITIONS.values())`\n  converts AACS to board on its own.")


def edge_lengths_mm(mapping: DeskMapping, detection: detect.CubeDetection) -> Tuple[float, float]:
    """The two visible base edges of this cube, measured through the mapping.

    The detector gives three of the four base corners: two diagonally opposite
    ones and the one nearest the camera between them. Both of those edges are one
    cube wide on the desk, so putting them through the mapping and getting
    something other than 40 mm is the mapping being wrong *there* -- which is the
    only local accuracy check available without a fourth point pair.
    """
    left, near, right = (np.array(mapping.pixel_to_ground(corner)) for corner in detection.base_corners)
    return (float(np.linalg.norm(near - left)), float(np.linalg.norm(right - near)))


def pair_up(detections: Sequence[detect.CubeDetection], placements: Dict[str, Position]
            ) -> Tuple[List[Tuple[Observation, detect.CubeDetection]], List[str]]:
    """Each detected cube against the coordinate the arm placed that colour at.

    The detection travels with the pair it produced: it is what the base edge
    check needs, and it is the only thing that can tell two rows of the same
    colour apart.
    """
    found = {det.colour: det for det in detections}
    pairs = []
    for colour, position in sorted(placements.items()):
        if colour not in found:
            continue
        detection = found[colour]
        # Two pixels per cube, same x and y, one cube-height apart. The second is
        # the only thing in the scene that says what height does to a pixel, and
        # without it the mapping can place a cube's base and nothing else.
        pairs.append((Observation(colour, detection.base_center, position, 0.0), detection))
        pairs.append((Observation(colour, detection.top_center, position, CUBE_SIZE_MM),
                      detection))
    return pairs, [colour for colour in sorted(placements) if colour not in found]


def cube_placements(overrides: Sequence[str]) -> Dict[str, Position]:
    """Where each cube is in this frame: `arm.py`'s coordinates unless told otherwise.

    An override is how a fourth point pair gets in -- the arm moves a cube
    somewhere else and that coordinate, not the one in `arm.py`, is what this
    frame shows.
    """
    placements: Dict[str, Position] = dict(CUBE_POSITIONS)
    for override in overrides:
        colour, _, coordinates = override.partition("=")
        numbers = [float(value) for value in coordinates.split(",")]
        if colour not in CUBE_POSITIONS or not 2 <= len(numbers) <= 3:
            raise SystemExit(f"--at wants <colour>=x,y[,z], got {override!r}")
        # Without a z, the height stays whatever `arm.py` says that cube grapples
        # at. That is only right if the cube has not moved much in reach.
        # The z is the desk under that cube, the same thing `CUBE_POSITIONS`
        # holds -- not a height for the cup, which is 40 mm higher.
        placements[colour] = (numbers[0], numbers[1],
                              numbers[2] if len(numbers) == 3 else CUBE_POSITIONS[colour][2])
    return placements


def read_frame(args: argparse.Namespace) -> Tuple[np.ndarray, str]:
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise RuntimeError(f"cannot read {args.image}")
        return image, str(args.image)
    return capture(args.device)


def capture(device: str) -> Tuple[np.ndarray, str]:
    """One snapshot at the largest size the camera will give, several frames averaged.

    The size is checked against what came back rather than against what the
    device accepted: this camera has been seen to agree to a format and then hand
    over 640x480 anyway.
    """
    camera = cv2.VideoCapture(_device_index(device), cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {device}; `v4l2-ctl --list-devices` says which is which")
    try:
        started = time.monotonic()
        fourcc, width, height = largest_frame(camera)
        if not width:
            raise RuntimeError(f"{device} opened but handed over no frame in "
                               f"{' or '.join(CAPTURE_FORMATS)}")
        if width * height < SMALLEST_USEFUL_FRAME:
            raise RuntimeError(f"{device} offers nothing bigger than {width}x{height}")
        _request(camera, fourcc, width, height)
        for _ in range(SETTLE_FRAMES):       # auto-exposure arrives a few frames late
            camera.read()
        image = average_frames(camera)
        elapsed = time.monotonic() - started
        return image, (f"{device} ({width}x{height} {fourcc}, {AVERAGED_FRAMES} frames "
                       f"averaged, {elapsed:.1f} s)")
    finally:
        camera.release()


def average_frames(camera: cv2.VideoCapture) -> np.ndarray:
    """Several frames of a scene that is not moving, averaged.

    Free accuracy, and aimed at the detector's actual limit: the mask boundary
    moves because the cube's edges are soft, and sensor noise is part of why.
    """
    total = None
    for index in range(AVERAGED_FRAMES):
        is_read, frame = camera.read()
        if not is_read:
            raise RuntimeError(f"the camera stopped after {index} frames")
        total = frame.astype(np.float64) if total is None else total + frame
    return (total / AVERAGED_FRAMES).round().astype(np.uint8)


def largest_frame(camera: cv2.VideoCapture) -> Tuple[str, int, int]:
    """The biggest frame this camera will actually hand over, and in which format.

    What the device *accepted* is not the question -- this camera has been seen
    to agree to a format and then hand over 640x480 anyway -- so each candidate
    is judged by the frame that comes back from it.
    """
    best = ("", 0, 0)
    for fourcc in CAPTURE_FORMATS:
        _request(camera, fourcc, LARGER_THAN_ANY_SENSOR, LARGER_THAN_ANY_SENSOR)
        is_read, frame = camera.read()
        if is_read and frame.shape[1] * frame.shape[0] > best[1] * best[2]:
            best = (fourcc, frame.shape[1], frame.shape[0])
    return best


def _request(camera: cv2.VideoCapture, fourcc: str, width: int, height: int) -> None:
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _device_index(device: str) -> int:
    """`/dev/video1` as the 1 that OpenCV wants.

    This build of OpenCV declines to open a camera by name -- "backend is
    generally available but can't be used to capture by name" -- and with the
    V4L2 backend an index is `/dev/video<index>` anyway, so the name is still
    what goes in the config and on the command line.
    """
    digits = "".join(character for character in device if character.isdigit())
    if not digits:
        raise RuntimeError(f"no camera number in {device!r}; give /dev/videoN or N")
    return int(digits)


def check_image(image: np.ndarray, detections: Sequence[detect.CubeDetection],
                mapping: DeskMapping) -> np.ndarray:
    """The detections, with the arm's coordinate grid drawn where the mapping puts it.

    Worth a look every time. The cube wireframes say the detector found the right
    things; the grid says the mapping agrees with the desk. Its lines should run
    with the desk's own -- a board edge, the wood grain -- and its squares shrink
    with distance; a grid of parallelograms is an affine fit, which cannot.
    """
    canvas = detect.annotate(image, list(detections))
    scale = max(1, round(canvas.shape[1] / 1280))
    xs, ys = _grid_lines()
    # The desk, and the plane the cube tops were solved at drawn heavier over it:
    # each cube's top centre should sit on its own coordinate there, the way its
    # base centre does on the desk.
    planes = [(0.0, GRID_BGR, scale, 1.0)]
    if mapping.lifted is not None:
        planes.append((mapping.lift_mm, LIFTED_GRID_BGR, 2 * scale, LIFTED_GRID_ALPHA))
    for height, colour, thickness, alpha in planes:
        layer = canvas.copy()
        # Forty points per line rather than two: a perspective mapping bends a
        # straight line in arm millimetres into a curve on the sensor.
        for x in xs:
            _polyline(layer, [mapping.ground_to_pixel((x, y), height)
                              for y in np.linspace(ys[0], ys[-1], 40)], colour, thickness)
        for y in ys:
            _polyline(layer, [mapping.ground_to_pixel((x, y), height)
                              for x in np.linspace(xs[0], xs[-1], 40)], colour, thickness)
        canvas = cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)
    if mapping.lifted is not None:
        _label(canvas, mapping.ground_to_pixel((xs[0], ys[0]), mapping.lift_mm),
               f"z={mapping.lift_mm:.0f}", scale, LIFTED_GRID_BGR)
        for detection in detections:
            top = tuple(int(value) for value in detection.top_center)
            cv2.drawMarker(canvas, top, (0, 0, 0), cv2.MARKER_CROSS, 20 * scale, 3 * scale)
            cv2.drawMarker(canvas, top, LIFTED_GRID_BGR, cv2.MARKER_CROSS, 20 * scale, scale)
    # Every other line, and y along the near edge: the far edge is where the
    # perspective packs the lines too close for a label each.
    for x in xs[::2]:
        _label(canvas, mapping.ground_to_pixel((x, ys[-1])), f"x={x:.0f}", scale)
    for y in ys[1:-1:2]:          # not the corner: x has that one
        _label(canvas, mapping.ground_to_pixel((xs[-1], y)), f"y={y:.0f}", scale)
    return canvas


def _grid_lines() -> Tuple[np.ndarray, np.ndarray]:
    return tuple(np.arange(low, high + GRID_STEP_MM / 2, GRID_STEP_MM)
                 for low, high in (GRID_X_MM, GRID_Y_MM))


def _polyline(canvas: np.ndarray, points: Sequence[Tuple[float, float]], colour: Tuple[int, int, int],
              thickness: int) -> None:
    # Clipped before the cast: a grid line that runs off toward the horizon can
    # come back as a number int32 cannot hold, and OpenCV clips the rest itself.
    line = np.clip(np.array(points), -1e6, 1e6).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [line], False, colour, thickness, cv2.LINE_AA)


def _label(canvas: np.ndarray, pixel: Tuple[float, float], text: str, scale: int,
           colour: Tuple[int, int, int] = GRID_BGR) -> None:
    position = (int(np.clip(pixel[0], -1e6, 1e6)) + 4, int(np.clip(pixel[1], -1e6, 1e6)) - 4)
    for ink, thickness in (((0, 0, 0), 3 * scale), (colour, scale)):
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, ink,
                    thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path,
                        help="fit from this frame instead of taking a snapshot")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"camera to snapshot (default {DEFAULT_DEVICE})")
    parser.add_argument("--at", action="append", default=[], metavar="COLOUR=X,Y[,Z]",
                        help="this frame has that cube at these arm coordinates, not the "
                             "ones in arm.py. Z is the desk under it, as CUBE_POSITIONS "
                             "means it -- not a height for the cup")
    parser.add_argument("--keep", action="store_true",
                        help="fit through the points in the existing calibration as well "
                             "as this frame's -- only valid if the camera has not moved")
    parser.add_argument("--dry-run", action="store_true", help="fit and print, write nothing")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
