#!/usr/bin/env python3
"""Try the new motion on the real arm, one small stage at a time. No camera needed.

    ../.venv/bin/python agenttools/try-arm.py --dry-run    # check every target, no board
    ../.venv/bin/python agenttools/try-arm.py              # the arm, prompted stage by stage
    ../.venv/bin/python agenttools/try-arm.py --fake       # the same, against the fake board

Desk coordinates throughout (`config/desk.json`): z is height above the desk.
Before each stage it says what it is about to do: Enter runs it, `s` skips it,
`q` ends the run. After each move it prints where the arm was sent and where
its servos say it is -- the open-loop accuracy, which is worth seeing.

Ctrl-C holds the arm where it is; a second one quits. Everything printed also
goes to agenttools/logs/, a line at a time, so an interrupted run keeps it.
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from maxarm import MaxArm, MoveType, Space                    # noqa: E402

SAFE = (150.0, -150.0, 80.0)            # high over the desk on the camera's side
FAR = (-50.0, -220.0, 80.0)
TILE_AT = (120.0, -200.0)               # where the owner puts a tile for the grab stage
DROP_AT = (-40.0, -200.0)
TILE_HEIGHT_MM = 4.0
CLEARANCE_MM = TILE_HEIGHT_MM + 10.0


def stages(arm: MaxArm, say):
    def move(x, y, z, move_type=MoveType.FAST, angle=None):
        report(say, arm, f"move to ({x:.0f}, {y:.0f}, {z:.0f}) {move_type.value}"
               + (f", cup to {angle:+.0f}" if angle is not None else ""),
               arm.move(x, y, z, move_type, angle=angle))

    def grab(x, y, height, angle):
        report(say, arm, f"grab at ({x:.0f}, {y:.0f}, {height:.0f}), facing {angle:+.0f}",
               arm.grab(x, y, height, angle=angle))

    def drop(x, y, height, angle):
        report(say, arm, f"drop at ({x:.0f}, {y:.0f}, {height:.0f}), facing {angle:+.0f}",
               arm.drop(x, y, height, angle=angle))

    return [
        ("fast", "three fast moves high over the desk, at 200 mm/s",
         lambda: (move(*SAFE), move(*FAR), move(*SAFE))),
        ("slow", "30 mm down at 10 mm/s -- three seconds -- and back up fast",
         lambda: (move(SAFE[0], SAFE[1], 50.0, MoveType.SLOW), move(*SAFE))),
        ("turn", "the cup square to the arm's axes (0), then +120, then -30: swings of 120 and "
                 "150 degrees, about a second each, and each move waits for it",
         lambda: (move(*SAFE, angle=0.0), move(*SAFE, angle=120.0), move(*SAFE, angle=-30.0),
                  move(*SAFE, angle=0.0))),
        ("clearance", f"clearance {CLEARANCE_MM:.0f}: down to 10 mm, then to another low point -- "
                      f"it should climb to {CLEARANCE_MM + 5:.0f}, cross, and come down",
         lambda: (arm.set_clearance(CLEARANCE_MM), move(SAFE[0], SAFE[1], 10.0),
                  move(FAR[0], FAR[1], 10.0), move(*SAFE))),
        ("base", "left of the base to right of it, behind its front edge: it should go by both "
                 "front corners. Needs free desk left of centre",
         lambda: (move(-150.0, -60.0, 80.0), move(150.0, -60.0, 80.0), move(*SAFE))),
        ("grab", f"put a {TILE_HEIGHT_MM:.0f} mm tile at desk ({TILE_AT[0]:.0f}, {TILE_AT[1]:.0f}) "
                 f"first. Grab it, set it down at ({DROP_AT[0]:.0f}, {DROP_AT[1]:.0f}) turned 45, "
                 f"then take it back",
         lambda: (arm.set_clearance(CLEARANCE_MM), grab(*TILE_AT, TILE_HEIGHT_MM, 0.0),
                  drop(*DROP_AT, TILE_HEIGHT_MM + 1.0, 45.0),
                  grab(*DROP_AT, TILE_HEIGHT_MM, 45.0),
                  drop(*TILE_AT, TILE_HEIGHT_MM + 1.0, 0.0))),
        ("home", "back to the reset pose", lambda: report(say, arm, "home", arm.home())),
    ]


def report(say, arm: MaxArm, what: str, result) -> None:
    measured = arm.get_position() if result.status.value != "refused" else None
    sent = "({:.0f}, {:.0f}, {:.0f})".format(*result.position)
    seen = " servos say ({:.0f}, {:.0f}, {:.0f})".format(*measured) if measured else ""
    say(f"   {what}: {result.status.value}, sent to {sent}{seen}"
        + (f" -- {result.detail}" if result.detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="check the targets only")
    parser.add_argument("--fake", action="store_true", help="the fake board, waits skipped")
    args = parser.parse_args()
    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / time.strftime("try-arm-%Y%m%d-%H%M%S.txt")
    with log_path.open("w") as log:
        def say(line: str) -> None:
            print(line)
            log.write(line + "\n")
            log.flush()

        if args.dry_run:
            return dry_run(say)
        with board(args.fake) as arm:
            arm.use_coordinates(Space.DESK)
            arm.stop_on_ctrl_c()
            say(f"connected; log in {log_path}")
            for name, description, run in stages(arm, say):
                answer = "" if args.fake else input(f"\n-- {name}: {description}\n   Enter, s, q: ")
                if answer.strip() == "q":
                    break
                if answer.strip() != "s":
                    say(f"\n-- {name}")
                    run()
    return 0


def dry_run(say) -> int:
    arm = MaxArm()
    arm.use_coordinates(Space.DESK)
    targets = [SAFE, FAR, (SAFE[0], SAFE[1], 50.0), (SAFE[0], SAFE[1], 10.0), (FAR[0], FAR[1], 10.0),
               (-150.0, -60.0, 80.0), (150.0, -60.0, 80.0), (*TILE_AT, TILE_HEIGHT_MM),
               (*DROP_AT, TILE_HEIGHT_MM + 1.0)]
    for target in targets:
        say(f"   ({target[0]:5.0f}, {target[1]:5.0f}, {target[2]:4.0f})  "
            f"{'reachable' if arm.can_reach(*target) else 'NOT REACHABLE'}")
    return 0 if all(arm.can_reach(*target) for target in targets) else 1


@contextlib.contextmanager
def board(is_fake: bool):
    if not is_fake:
        print("connecting -- the board resets and homes the arm; stand clear")
        with MaxArm() as arm:
            yield arm
        return
    from fake_arm import fake_arm
    with fake_arm() as (arm, _, _):
        yield arm


if __name__ == "__main__":
    sys.exit(main())
