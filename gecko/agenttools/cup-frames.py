#!/usr/bin/env python3
"""The cup photographed at desk points, to see its size against the calibration's. MOVES THE ARM.

    ../.venv/bin/python agenttools/cup-frames.py --out /tmp/cup-frames

For each point: over it at 20 mm over the desk, slowly down, corrected (PRECISE), as the calibration pages move;
the servos read; a full-size frame. Then home. Desk space through config/; the desk must be clear but for flat
things. Each frame and reading is written as it is taken (cup-frames.json beside the frames), so Ctrl-C keeps
what was taken; Ctrl-C holds the arm where it is.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from maxarm import MaxArm, Space                         # noqa: E402
from maxarm.spaces import CONFIG_DIR                     # noqa: E402
from camera.capture import capture                      # noqa: E402
from common.motion import Motion                        # noqa: E402

# Desk x, y and the cup's height over the desk: at 2 the cup's lip just touches it (desk 0 is the cup squeezed).
POINTS = [((100.0, -200.0), 2.0), ((160.0, 0.0), 2.0), ((200.0, -150.0), 2.0), ((-150.0, -150.0), 2.0),
          ((0.0, -230.0), 2.0), ((160.0, 0.0), 20.0)]
HOVER_MM = 20.0
REACH_LIMIT_MM = 270.0           # the owner's, for this run
SETTLE_S = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/tmp/cup-frames"))
    args = parser.parse_args()
    far = [point for point, _ in POINTS if math.hypot(*point) > REACH_LIMIT_MM]
    if far:
        parser.error(f"past {REACH_LIMIT_MM:.0f} mm: {far}")
    args.out.mkdir(parents=True, exist_ok=True)
    motion = Motion(MaxArm(config_dir=None), CONFIG_DIR)
    motion.arm.stop_on_ctrl_c()
    print("connecting: the board resets and homes the arm", flush=True)
    motion.connect()
    taken = []
    try:
        for index, ((x, y), z) in enumerate(POINTS, 1):
            result = motion.approach(Space.DESK, (x, y, z), precise=True, hover=HOVER_MM)
            print(f"{index}: desk ({x:.0f}, {y:.0f}, {z:.0f}): {result.status.value} {result.detail}", flush=True)
            if not result:
                break
            time.sleep(SETTLE_S)
            motion.read()
            says = motion.spaces.convert(motion.measured, Space.ARM, Space.DESK) if motion.measured else None
            frame, how = capture()
            name = f"cup-{index}.png"
            cv2.imwrite(str(args.out / name), frame)
            taken.append({"frame": name, "sent": [x, y, z], "says": says, "taken": how})
            (args.out / "cup-frames.json").write_text(json.dumps(taken, indent=2) + "\n")
            print(f"   arm says {tuple(round(v, 2) for v in says) if says else None}; {how}", flush=True)
    finally:
        print("home", flush=True)
        motion.reset()
        motion.arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
