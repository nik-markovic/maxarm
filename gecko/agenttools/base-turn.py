#!/usr/bin/env python3
"""The red cube's points fitted as an arm whose base turns off by a wave; and the ruler marks that would show it.

    ../.venv/bin/python agenttools/base-turn.py              # config/ as it is; no arm, no camera
    ../.venv/bin/python agenttools/base-turn.py ruler.json   # and the owner's marks and distances against it

Each Fine Tune point (and each of the three cubes) is where the servos said the cup was and where the camera
saw the cube. They are fitted as an arm whose reach is off by a share and a few mm, and whose base points off
by a wave that repeats every half turn (a sin 2b + c cos 2b, b the bearing the servos read); the camera's frame
turned and shifted onto the arm's, so that straight out is where the servo says. Prints the points by the way
the arm pointed and draws them beside this script: base-turn.png by direction, base-turn-desk.png from above.
Then what the ruler test should measure: the cup lowered onto tape on the desk at each of MARKS, a circle drawn
round it, the distances between the circles' centres measured.

ruler.json: {"marks": {"1": [x, y], ...}, "distances": {"1-2": 131.5, ...}}, x and y as the desk page's "arm
says". Marks it leaves out are taken as sent.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                      # noqa: E402
from camera.fit import HANDEDNESS                             # noqa: E402
from camera.model import least_squares                        # noqa: E402
from camera.refine import cube_sightings, seen_true           # noqa: E402

# 1-7 on a circle 200 mm out, x < 0 to x > 0; 8-10 on one 100 mm out; 11 straight out at 260. All on the desk
# page's 20 mm grid.
MARKS = {"1": (-200, 0), "2": (-160, -120), "3": (-120, -160), "4": (0, -200), "5": (120, -160),
         "6": (160, -120), "7": (200, 0), "8": (-60, -80), "9": (0, -100), "10": (60, -80), "11": (0, -260)}
DISTANCES = ["1-2", "3-4", "4-5", "6-7", "3-5", "8-10", "9-11"]
GROUPS = [(-105, -67.5), (-67.5, -22.5), (-22.5, 22.5), (22.5, 67.5), (67.5, 105)]
RESAMPLES = 300
CHART, DESK = Path(__file__).with_name("base-turn.png"), Path(__file__).with_name("base-turn-desk.png")
ARROW_TIMES = 8
CUBE_BGR = {"blue": (200, 80, 0), "green": (0, 150, 0), "red": (0, 0, 220)}
TEXT, GREY, LIGHT, LINE = (0, 0, 0), (60, 60, 60), (235, 235, 235), (40, 40, 200)


def polar(points) -> tuple:
    """Reach from the base's axis, and bearing: 0 straight out (-y), positive toward x > 0."""
    points = np.asarray(points, float).reshape(-1, 2)
    return np.hypot(points[:, 0], points[:, 1]), np.arctan2(points[:, 0], -points[:, 1])


def wave(arm: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    return arm[2] * np.sin(2 * bearing) + arm[3] * np.cos(2 * bearing)


def placed(arm: np.ndarray, reach: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    """Where the cup really is, in the arm's frame, for what the servos read: arm is (share, mm, a, c)."""
    out, really = arm[0] * reach + arm[1], bearing + wave(arm, bearing)
    return np.column_stack((out * np.sin(really), -out * np.cos(really)))


def turned(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def fit(reach, bearing, seen, is_wave: bool) -> np.ndarray:
    """(shift x, shift y, turn, share, mm, a, c): the camera frame onto the arm's, and the arm's own error."""
    evenly = placed(np.array([1.0, 0, 0, 0]), reach, bearing)
    across = (evenly - evenly.mean(0)).T @ (seen - seen.mean(0))
    start_turn = math.atan2(across[0, 1] - across[1, 0], across[0, 0] + across[1, 1])
    start = np.array([*(seen.mean(0) - turned(start_turn) @ evenly.mean(0)), start_turn, 1.0, 0.0, 0.0, 0.0])

    def misses(p):
        return (placed(p[3:], reach, bearing) @ turned(p[2]).T + p[:2] - seen).ravel()

    return least_squares(misses, start, np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3, 1e-6, 1e-6]), huber=1e9,
                         free=list(range(7 if is_wave else 5)), iterations=60)[0]


def in_arm_frame(p: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Where the camera saw each cube, from the arm's axis, turned so that straight out has no turning error."""
    return (seen - p[:2]) @ turned(p[2]) @ turned(p[6])


def outward(bearing: np.ndarray, error: np.ndarray) -> np.ndarray:
    """A turn error toward x > 0 as one away from straight out (negative: short of where the servo says)."""
    return np.where(bearing < 0, -error, error)


def distances(arm: np.ndarray, marks: dict) -> dict:
    where = {name: placed(arm, *polar(xy))[0] for name, xy in marks.items()}
    return {pair: float(np.linalg.norm(where[pair.split("-")[0]] - where[pair.split("-")[1]])) for pair in DISTANCES}


def put(picture, text, at, size=0.45, colour=GREY) -> None:
    cv2.putText(picture, text, (int(at[0]), int(at[1])), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)


def shade(reach: float) -> tuple:
    value = int(np.interp(reach, (110, 270), (190, 0)))
    return value, value, value


def draw_chart(bearing, reach, out_deg, reach_off, names, arm) -> None:
    width, top_h, low_h, left, right = 1200, 460, 230, 90, 90
    picture = np.full((40 + top_h + 50 + low_h + 90, width, 3), 255, np.uint8)
    span = (-105.0, 105.0)

    def x_of(degrees):
        return int(round(left + (degrees - span[0]) / (span[1] - span[0]) * (width - left - right)))

    def panel(y0, height, low, high, step, label, right_label=None):
        def y_of(value):
            return int(round(y0 + (high - value) / (high - low) * height))
        for degrees in range(-90, 91, 15):
            cv2.line(picture, (x_of(degrees), y0), (x_of(degrees), y0 + height), LIGHT, 1)
        for value in np.arange(math.ceil(low / step) * step, high + 1e-9, step):
            cv2.line(picture, (left, y_of(value)), (width - right, y_of(value)),
                     (150, 150, 150) if abs(value) < 1e-9 else LIGHT, 1)
            put(picture, f"{value:+.1f}" if abs(value) > 1e-9 else "0", (left - 52, y_of(value) + 5))
            if right_label:
                put(picture, right_label(value), (width - right + 8, y_of(value) + 5))
        cv2.rectangle(picture, (left, y0), (width - right, y0 + height), (120, 120, 120), 1)
        put(picture, label, (left, y0 - 10), 0.55, TEXT)
        return y_of

    def dot(centre, name, out, radius):
        if name in CUBE_BGR:
            cv2.rectangle(picture, (centre[0] - radius, centre[1] - radius), (centre[0] + radius, centre[1] + radius),
                          CUBE_BGR[name], -1)
        else:
            cv2.circle(picture, centre, radius, shade(out), -1, cv2.LINE_AA)
            cv2.circle(picture, centre, radius, TEXT, 1, cv2.LINE_AA)

    y_top = panel(40, top_h, -2.5, 2.0, 0.5, "How much further out the base really turned than its servo says: "
                  "degrees (left), mm at 200 mm out (right). Straight out is 0",
                  lambda degrees: f"{math.radians(degrees) * 200:+.1f}" if abs(degrees) > 1e-9 else "0")
    curve = np.radians(np.linspace(-100, 100, 401))
    line = np.degrees(outward(curve, wave(arm, curve) - arm[3]))
    cv2.polylines(picture, [np.array([(x_of(d), y_top(v)) for d, v in zip(np.degrees(curve), line)], np.int32)],
                  False, LINE, 2, cv2.LINE_AA)
    for name, degrees, value, out in zip(names, np.degrees(bearing), out_deg, reach):
        dot((x_of(degrees), y_top(value)), name, out, 6)
    for mark, xy in MARKS.items():
        if int(mark) <= 7:
            put(picture, mark, (x_of(math.degrees(math.atan2(xy[0], -xy[1]))) - 5, 40 + top_h - 8), 0.5, LINE)
    y0 = 40 + top_h + 50
    y_low = panel(y0, low_h, -2.0, 6.0, 2.0, "How much farther out the cup really was than the servos say, mm")
    for name, degrees, value, out in zip(names, np.degrees(bearing), reach_off, reach):
        dot((x_of(degrees), y_low(value)), name, out, 5)
    for degrees in range(-90, 91, 15):
        put(picture, f"{degrees}", (x_of(degrees) - 12, y0 + low_h + 20))
    put(picture, "x < 0 side     <--   which way the servo says the arm points, degrees from straight out   -->"
        "     x > 0 side", (left + 40, y0 + low_h + 45), 0.5, TEXT)
    put(picture, "dots: Fine Tune points, darker = farther out (110-270 mm); squares: the three cubes; line: the "
        "fitted wave; 1-7: the ruler test's marks", (left, y0 + low_h + 72))
    cv2.imwrite(str(CHART), picture)


def draw_desk(said: np.ndarray, really: np.ndarray, names) -> None:
    scale, width, height = 1.5, 900, 700
    origin = np.array([width / 2, 525.0])                      # the base's axis; straight out is up

    def at(xy):
        return tuple(np.round(origin + np.asarray(xy) * scale).astype(int))

    picture = np.full((height, width, 3), 255, np.uint8)
    for value in range(-280, 281, 20):
        colour = (200, 200, 200) if value == 0 else LIGHT
        cv2.line(picture, at((value, 100)), at((value, -300)), colour, 1)
        if -300 <= value <= 100:
            cv2.line(picture, at((-290, value)), at((290, value)), colour, 1)
    for radius in (100, 200, 270):
        cv2.circle(picture, at((0, 0)), int(radius * scale), (170, 170, 170), 1, cv2.LINE_AA)
        put(picture, f"{radius} mm", at((4, -radius - 4)))
    cv2.rectangle(picture, at((-60, -60)), at((60, 60)), (200, 200, 200), -1)
    put(picture, "base", at((-14, 4)), 0.5, TEXT)
    for name, start, end in zip(names, said, really):
        tip = start + (end - start) * ARROW_TIMES
        cv2.arrowedLine(picture, at(start), at(tip), LINE, 2, cv2.LINE_AA, tipLength=0.25)
        if name in CUBE_BGR:
            cv2.rectangle(picture, tuple(np.subtract(at(start), 5)), tuple(np.add(at(start), 5)), CUBE_BGR[name], -1)
        else:
            cv2.circle(picture, at(start), 4, TEXT, -1, cv2.LINE_AA)
    for mark, xy in MARKS.items():
        centre = at(xy)
        cv2.drawMarker(picture, centre, (0, 140, 255), cv2.MARKER_CROSS, 14, 2)
        put(picture, mark, (centre[0] + 7, centre[1] + 16), 0.5, (0, 110, 220))
    put(picture, "x < 0", at((-285, 90)), 0.55, TEXT)
    put(picture, "x > 0", at((250, 90)), 0.55, TEXT)
    put(picture, "straight out (y < 0)", (width / 2 + 8, 66), 0.55, TEXT)
    put(picture, "The desk from above, straight out up. Dots: where the servos said the cup was; arrows: to where "
        f"the camera", (12, 22), 0.5, TEXT)
    put(picture, f"saw the cube, drawn {ARROW_TIMES} times longer (1 mm = {ARROW_TIMES * scale:.0f} px). Squares: "
        "the three cubes. Orange crosses: the ruler test's marks.", (12, 44), 0.5, TEXT)
    cv2.imwrite(str(DESK), picture)


def main(argv) -> int:
    config = ROOT / "config"
    saved = store.load_camera(config)
    if saved is None:
        print("no camera.json")
        return 1
    alone = saved.cube_fit()
    points = store.load_refine_points(config)
    sightings = [point.sighting for point in points] + cube_sightings(alone)
    names = [f"#{point.id}" for point in points] + ["blue", "green", "red"]
    said = np.array([(sighting.x, sighting.y) for sighting in sightings])
    reach, bearing = polar(said)
    seen = np.array([seen_true(alone, sighting) for sighting in sightings]) @ HANDEDNESS

    uneven, even = fit(reach, bearing, seen, True), fit(reach, bearing, seen, False)
    for name, p in (("base turning evenly", even), ("base turning by the wave", uneven)):
        off = np.linalg.norm(placed(p[3:], reach, bearing) @ turned(p[2]).T + p[:2] - seen, axis=1)
        print(f"{name:25s}: the cube {np.median(off):.2f} mm from where the arm puts it typically, "
              f"{np.percentile(off, 90):.2f} for 90%, {off.max():.2f} worst")
    arm = uneven[3:]
    print(f"reach: {arm[0]:.4f} x what the servos say {arm[1]:+.2f} mm; wave {math.degrees(arm[2]):.2f} sin 2b "
          f"{math.degrees(arm[3]):+.2f} cos 2b degrees")

    really = in_arm_frame(uneven, seen)
    really_reach, really_bearing = polar(really)
    out_deg = np.degrees(outward(bearing, np.angle(np.exp(1j * (really_bearing - bearing)))))
    reach_off = really_reach - reach
    print("\npointing (servo)   points  further out, deg  mm at 200 out  mm where it was  farther out, mm")
    for low, high in GROUPS:
        group = (np.degrees(bearing) >= low) & (np.degrees(bearing) < high)
        print(f"{low:6.0f} to {high:5.0f}   {group.sum():6d}  {np.median(out_deg[group]):16.2f}  "
              f"{math.radians(np.median(out_deg[group])) * 200:13.1f}  "
              f"{np.median(np.radians(out_deg[group]) * reach[group]):15.1f}  {np.median(reach_off[group]):15.1f}")
    draw_chart(bearing, reach, out_deg, reach_off, names, arm)
    draw_desk(said, really, names)
    print(f"\npictures: {CHART.relative_to(ROOT)}, {DESK.relative_to(ROOT)}")

    ruler = json.loads(Path(argv[1]).read_text()) if len(argv) > 1 else {}
    marks = {name: tuple(ruler.get("marks", {}).get(name, xy)) for name, xy in MARKS.items()}
    measured = ruler.get("distances", {})
    says = distances(np.array([1.0, 0, 0, 0]), marks)
    evenly, by_wave = distances(even[3:], marks), distances(arm, marks)
    generator, resampled = np.random.default_rng(1), []
    for _ in range(RESAMPLES):
        pick = generator.integers(0, len(reach), len(reach))
        resampled.append(distances(fit(reach[pick], bearing[pick], seen[pick], True)[3:], marks))
    print("\nmarks " + ", ".join(f"{name} ({x:g}, {y:g})" for name, (x, y) in marks.items()))
    print("between  the arm says  if it turned evenly  with the wave (80% of resamples)  measured")
    for pair in DISTANCES:
        spread = np.percentile([one[pair] for one in resampled], [10, 90])
        print(f"{pair:7s}  {says[pair]:12.1f}  {evenly[pair]:19.1f}  {by_wave[pair]:14.1f} ({spread[0]:.1f}-"
              f"{spread[1]:.1f})  {measured[pair] if pair in measured else '':>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
