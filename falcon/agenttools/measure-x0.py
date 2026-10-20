#!/usr/bin/env python3
"""Reported against real, along the arm's centre line: the owner measures, this compares the explanations.

    ../.venv/bin/python agenttools/measure-x0.py              # on the arm; Ctrl-C holds it
    ../.venv/bin/python agenttools/measure-x0.py --dry-run    # the stops and the predictions, no arm
    ../.venv/bin/python agenttools/measure-x0.py --walk       # no owner: free-air stops, servos read

The arm stays on x = 0. Three times it holds the cup 5 mm over a sheet of paper (reach 100, 180,
260) for the owner to trace round the rim; then it holds the cup at six heights for the owner
to measure the cup tip above the desk with a ruler; then the reset pose, once more. At every stop
the servos are read (median of five, settled) and the line goes to agenttools/logs/ at once. At
the end it asks for the distances between the traced circles and prints, for every stop, what was
measured against what each explanation in work/KINEMATICS.md section 5 predicts from where the
servos really were.

`--walk` needs nobody: it visits the owner's card-touch positions of 2026-10-19 (work/KINEMATICS.md
1.1), coming down onto each from 20 mm above as the owner did, and the six heights, all in free
air at firmware z 90 or more -- 40 mm and more above the desk -- and logs where the servos land.
"""

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from maxarm import geometry                                  # noqa: E402
from maxarm.arm import MaxArm, MoveType                      # noqa: E402

CONFIG = HERE.parent / "config"
CIRCLE_REACH = (100.0, 180.0, 260.0)
CIRCLE_HOVER_MM = 5.0            # above where the owner's snug touches read: clear of the paper
CIRCLE_LIFT_MM = 40.0
CARD_TOUCHES = ((-260.0, 98.0), (-220.0, 98.0), (-200.0, 98.0), (-180.0, 100.0), (-160.0, 101.0),
                (-140.0, 103.0), (-120.0, 104.0), (-100.0, 104.0), (-80.0, 105.0), (-80.0, 135.0), (-260.0, 132.0))
WALK_FLOOR_Z = 90.0              # the walk never sends the cup lower: the desk reads 42-52
HEIGHTS = ((-260.0, 90.0), (-260.0, 150.0), (-200.0, 180.0), (-160.0, 120.0), (-120.0, 205.0), (-100.0, 120.0))
SQUASH_MM = 0.75                 # a snug desk touch presses the bellows about this much (owner: under 1)
SCALE = 0.9517                   # agenttools/fit-servo-scale.py
# On the owner's arm a precise move has settled when it returns: over the next 8 s the servos only
# flicker +-1 step between single reads. A second and the middle of nine reads smooth that out.
READ_WAIT_MS = 1000
READS = 9


def desk_row() -> List[Tuple[float, float]]:
    """(reach, firmware z) of the owner's desk touches straight out from the base, nearest first."""
    points = json.loads((CONFIG / "desk-points.json").read_text())["points"]
    return sorted((math.hypot(p["x"], p["y"]), p["z"]) for p in points if abs(p["x"]) < 1.0)


def desk_z(reach: float) -> float:
    reaches, heights = zip(*desk_row())
    return float(np.interp(reach, reaches, heights))


def stops() -> List[Dict]:
    plan = []
    for reach in CIRCLE_REACH:
        plan.append({"name": f"circle at reach {reach:.0f}", "kind": "circle",
                     "at": (0.0, -reach, desk_z(reach) + CIRCLE_HOVER_MM)})
    for y, z in HEIGHTS:
        plan.append({"name": f"height at reach {-y:.0f}, z {z:.0f}", "kind": "height", "at": (0.0, y, z)})
    plan.append({"name": "reset pose", "kind": "height", "at": geometry.HOME})
    return plan


# --- the explanations, each from where the servos really are -------------------------------

def _elevations(position) -> Tuple[float, float]:
    _, angle2, angle3 = geometry.inverse(position)
    return math.radians(180.0 - angle2), math.radians(-angle3)


def _wrist(position, scale: float) -> Tuple[float, float]:
    """(out from the base axis, up from the shoulder) of the wrist, servos turning `scale` of the assumed."""
    upper, fore = _elevations(position)
    upper = math.pi / 2 + scale * (upper - math.pi / 2)
    fore = scale * fore
    return (geometry.L1 + geometry.L2 * math.cos(upper) + geometry.L3 * math.cos(fore),
            geometry.L2 * math.sin(upper) + geometry.L3 * math.sin(fore))


def _desk_touches() -> List[Tuple[float, float, float]]:
    points = json.loads((CONFIG / "desk-points.json").read_text())["points"]
    return [(p["x"], p["y"], p["z"]) for p in points]


def predict_height(position, explanation: str, scale: float = SCALE) -> float:
    """Cup tip above the desk, hanging free, by one explanation."""
    if explanation == "reach only":       # the firmware's up-and-down is right; only the desk changes with reach
        return position[2] - desk_z(math.hypot(position[0], position[1])) - SQUASH_MM
    if explanation == "saved desk.json":
        saved = json.loads((CONFIG / "desk.json").read_text())
        true = geometry.true_position(position, tuple(saved["joint_offsets_deg"]))
        desk_to_arm = np.array(saved["desk_to_arm"])
        return float((np.linalg.inv(desk_to_arm) @ np.array([*true, 1.0]))[2]) - SQUASH_MM
    desk = float(np.mean([_wrist(touch, scale)[1] for touch in _desk_touches()]))
    return _wrist(position, scale)[1] - desk - SQUASH_MM


def predict_reach(position, explanation: str, scale: float = SCALE) -> float:
    if explanation == "servo scale":
        return _wrist(position, scale)[0]
    if explanation == "saved desk.json":
        offsets = tuple(json.loads((CONFIG / "desk.json").read_text())["joint_offsets_deg"])
        true = geometry.true_position(position, offsets)
        return math.hypot(true[0], true[1])
    return math.hypot(position[0], position[1])


EXPLANATIONS = ("servo scale", "reach only", "saved desk.json")


# --- running it ---------------------------------------------------------------------------

def ask(prompt: str, is_fake: bool, made_up: Optional[float]) -> Optional[str]:
    if is_fake:
        print(f"{prompt}{'' if made_up is None else f'{made_up:.1f}'}")
        return "" if made_up is None else f"{made_up:.1f}"
    return input(prompt).strip()


def where(arm: MaxArm):
    """Where the servos put the cup, firmware coordinates: the middle of nine reads, after a second."""
    pulses = arm._protocol.read_settled_pulses(READS, READ_WAIT_MS)
    return geometry.forward(geometry.pulse_to_deg(pulses)) if pulses else arm.get_position()


def measure(arm: MaxArm, is_fake: bool, log) -> List[Dict]:
    done = []
    for number, stop in enumerate(stops(), 1):
        x, y, z = stop["at"]
        print(f"\n[{number}/{len(stops())}] {stop['name']}: moving")
        if stop["kind"] == "circle":
            arm.move(x, y, z + CIRCLE_LIFT_MM, MoveType.FAST)
            result = arm.move(x, y, z, MoveType.PRECISE)
        elif stop["name"] == "reset pose":
            arm.home()
            result = arm.move(x, y, z, MoveType.PRECISE)
        else:
            arm.move(x, y, z, MoveType.FAST)
            result = arm.move(x, y, z, MoveType.PRECISE)
        if not result:
            print(f"   not reached: {result}; stopping here")
            break
        reported = where(arm)
        print("   the servos say ({:.1f}, {:.1f}, {:.1f})".format(*reported))
        entry = {"stop": stop["name"], "kind": stop["kind"], "sent": stop["at"], "reported": reported}
        if stop["kind"] == "circle":
            answer = ask("   trace round the cup's rim on the paper below it, then Enter (q to stop): ", is_fake, None)
            arm.move(x, y, z + CIRCLE_LIFT_MM, MoveType.FAST)
        else:
            made_up = predict_height(reported, "servo scale") + 0.3
            answer = ask("   cup tip above the desk, mm, to the half millimetre (Enter to skip, q to stop): ",
                         is_fake, made_up)
            entry["measured"] = float(answer) if answer and answer != "q" else None
        log.write(json.dumps(entry) + "\n")
        log.flush()
        done.append(entry)
        if answer == "q":
            break
    return done


def circle_distances(done: List[Dict], is_fake: bool) -> Dict[Tuple[int, int], float]:
    circles = [entry for entry in done if entry["kind"] == "circle"]
    if len(circles) < 2:
        return {}
    print("\nNow the circles: measure between their centres along the line, to the half millimetre.")
    measured = {}
    for a, b in ((0, 1), (1, 2), (0, 2)):
        if b >= len(circles):
            continue
        made_up = predict_reach(circles[b]["reported"], "servo scale") - \
            predict_reach(circles[a]["reported"], "servo scale")
        answer = ask(f"   reach {CIRCLE_REACH[a]:.0f} to {CIRCLE_REACH[b]:.0f}, mm (Enter to skip): ", is_fake, made_up)
        if answer:
            measured[(a, b)] = float(answer)
    return measured


def report(done: List[Dict], distances: Dict[Tuple[int, int], float]) -> None:
    print("\nHeights: the cup tip above the desk, measured and as each explanation predicts it (mm)")
    print(f"  {'stop':26s} {'measured':>9s}" + "".join(f"{name:>17s}" for name in EXPLANATIONS))
    misses = {name: [] for name in EXPLANATIONS}
    for entry in done:
        if entry["kind"] != "height":
            continue
        predicted = [predict_height(entry["reported"], name) for name in EXPLANATIONS]
        value = entry.get("measured")
        if value is not None:
            for name, guess in zip(EXPLANATIONS, predicted):
                misses[name].append(guess - value)
        shown = "-" if value is None else f"{value:.1f}"
        print(f"  {entry['stop']:26s} {shown:>9s}" + "".join(f"{guess:17.1f}" for guess in predicted))
    circles = [entry for entry in done if entry["kind"] == "circle"]
    if distances:
        print("\nCircles: distance between centres (mm)")
        for (a, b), value in distances.items():
            guesses = [predict_reach(circles[b]["reported"], name) - predict_reach(circles[a]["reported"], name)
                       for name in EXPLANATIONS]
            for name, guess in zip(EXPLANATIONS, guesses):
                misses[name].append(guess - value)
            print(f"  reach {CIRCLE_REACH[a]:.0f} to {CIRCLE_REACH[b]:.0f}: measured {value:.1f}; "
                  + ", ".join(f"{name} {guess:.1f}" for name, guess in zip(EXPLANATIONS, guesses)))
    print("\nHow far off each explanation is, over everything measured (mm):")
    for name, values in misses.items():
        if values:
            print(f"  {name:16s} typically {math.sqrt(np.mean(np.square(values))):.2f}, worst {max(map(abs, values)):.2f}")
    heights = [entry for entry in done if entry["kind"] == "height" and entry.get("measured") is not None]
    if len(heights) >= 3:
        grid = np.arange(0.90, 1.05, 0.0005)
        cost = [np.mean([(predict_height(e["reported"], "servo scale", s) - e["measured"]) ** 2 for e in heights])
                for s in grid]
        print(f"\nThe servo scale your heights alone point to: {grid[int(np.argmin(cost))]:.4f} "
              f"(the desk touches gave {SCALE})")


def walk(arm: MaxArm, log) -> None:
    """Free air only: each card-touch position from 20 mm above, then the six heights; servos read at each."""
    plan = [("card", (0.0, y, z), 20.0) for y, z in CARD_TOUCHES] + \
           [("height", (0.0, y, z), 0.0) for y, z in HEIGHTS] + [("reset", geometry.HOME, 0.0)]
    for number, (kind, (x, y, z), above) in enumerate(plan, 1):
        if z < WALK_FLOOR_Z or z + above < WALK_FLOOR_Z:
            raise ValueError(f"refusing ({x}, {y}, {z}): below the walk's floor")
        if above:
            arm.move(x, y, z + above, MoveType.FAST)
        result = arm.move(x, y, z, MoveType.FAST if not above else MoveType.SLOW)
        if result:
            result = arm.move(x, y, z, MoveType.PRECISE)
        if not result:
            print(f"[{number}/{len(plan)}] ({x:.0f}, {y:.0f}, {z:.0f}) not reached: {result}; stopping")
            return
        reported = where(arm)
        _, angle2, angle3 = geometry.inverse(reported)
        line = {"kind": kind, "sent": (x, y, z), "reported": reported,
                "upper_arm_deg": 180.0 - angle2, "forearm_deg": -angle3,
                "pulses": [round(v, 1) for v in geometry.deg_to_pulse(geometry.inverse(reported))]}
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(f"[{number}/{len(plan)}] {kind:6s} sent ({x:.0f}, {y:.0f}, {z:.1f})  servos say "
              "({:.2f}, {:.2f}, {:.2f})".format(*reported) + f"  z off {reported[2] - z:+.2f}", flush=True)


def dry_run() -> int:
    arm = MaxArm(config_dir=None)
    is_fine = True
    print("Stops, the firmware position sent, and what each explanation predicts there:")
    for stop in stops():
        x, y, z = stop["at"]
        reachable = arm.can_reach(x, y, z) and (stop["kind"] != "circle" or arm.can_reach(x, y, z + CIRCLE_LIFT_MM))
        is_fine = is_fine and reachable
        guess = (f"tip " + ", ".join(f"{name} {predict_height(stop['at'], name):.1f}" for name in EXPLANATIONS)
                 if stop["kind"] == "height" else
                 "reach " + ", ".join(f"{name} {predict_reach(stop['at'], name):.1f}" for name in EXPLANATIONS))
        print(f"  {stop['name']:26s} ({x:.0f}, {y:.0f}, {z:.1f}) {'' if reachable else 'NOT REACHABLE '}| {guess}")
    return 0 if is_fine else 1


@contextlib.contextmanager
def board(is_fake: bool):
    if not is_fake:
        print("connecting: the board resets and the arm goes to its reset pose; stand clear")
        with MaxArm(config_dir=None) as arm:
            yield arm
        return
    from fake_arm import fake_arm
    with fake_arm(config_dir=None) as (arm, _, _):
        yield arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the stops and predictions; no arm")
    parser.add_argument("--fake", action="store_true", help="run on the fake arm with made-up answers")
    parser.add_argument("--walk", action="store_true", help="no owner: free-air stops only, servos read")
    args = parser.parse_args()
    if args.dry_run:
        return dry_run()
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    kind = "walk-x0" if args.walk else "measure-x0"
    log_path = logs / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}{'-fake' if args.fake else ''}.jsonl"
    with board(args.fake) as arm, log_path.open("w") as log:
        arm.stop_on_ctrl_c()
        print(f"log: {log_path}", flush=True)
        if args.walk:
            walk(arm, log)
            arm.home()
            return 0
        if not args.fake:
            input("Lay a sheet of paper on the desk along the arm's centre line, from near the base out to\n"
                  "about 280 mm, held flat. Have the ruler ready. Enter to start: ")
        try:
            done = measure(arm, args.fake, log)
        except KeyboardInterrupt:
            print("\nstopped; the arm holds where it is")
            return 1
        distances = circle_distances(done, args.fake)
        if distances:
            log.write(json.dumps({"circle_distances": {f"{a}-{b}": v for (a, b), v in distances.items()}}) + "\n")
        report(done, distances)
    return 0


if __name__ == "__main__":
    sys.exit(main())
