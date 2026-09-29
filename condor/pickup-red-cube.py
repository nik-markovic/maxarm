#!/usr/bin/env python3
"""Find the red cube wherever it is and pick it up.

    ./pickup-red-cube.py

Snapshot, find the red cube's base, put that pixel through the calibration to get
an arm coordinate, hover over it, come down onto its top face, grip, lift.

Needs `config/calibration.json` (run `calibration/calibrate.py` first) and the
camera not to have moved since. Writes the overlay it worked from to
`/tmp/pickup-red-cube.jpg` -- look at it if the arm goes somewhere surprising.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))

import cv2                                                            # noqa: E402
from maxarm import MaxArm                                             # noqa: E402
from arm import CUBE_SIZE_MM, Stop                                    # noqa: E402
from mapping import DeskMapping                                       # noqa: E402
import calibrate                                                      # noqa: E402
import detect                                                         # noqa: E402

HOVER_Z = 80.0        # AACS: clear above the cube
LIFT_Z = 90.0         # AACS: where it ends up holding it
OVERLAY =  Path(ROOT / "../work/pickup-red-cube.jpg")


def main() -> int:
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")

    arm = MaxArm()
    arm.connect()
    Stop(arm)
    arm.home()

    image, source = calibrate.capture(calibrate.DEFAULT_DEVICE)
    found = detect.find_cubes(image, ["red"])
    cv2.imwrite(str(OVERLAY), detect.annotate(image, found))
    if not found:
        print(f"no red cube found -- not in the frame, or turned face-on to the camera. "
              f"see {OVERLAY}")
        return 1
    pixel = found[0].base_center

    # The cube's base is on the desk, so the pixel is a feature at AACS 0; the
    # cup grips its top face, which is a cube-height up.
    x, y, _ = mapping.pixel_to_aacs(pixel)
    print(f"red cube at pixel ({pixel[0]:.0f}, {pixel[1]:.0f}) from {source}")
    print(f"           -> AACS ({x:.0f}, {y:.0f}), gripping at AACS z {(CUBE_SIZE_MM-2.0):.0f}")
    print(f"           -> board {mapping.pixel_to_board(pixel, CUBE_SIZE_MM)}")
    if not mapping.is_inside(pixel):
        print("   note: outside the patch the calibration was fitted on")
    input(f"   look at {OVERLAY}, then press Enter to pick it up: ")

    arm = MaxArm()
    arm.connect()
    Stop(arm)
    try:        
        arm.move_to(*mapping.pixel_to_board(pixel, HOVER_Z))
        arm.move_to(*mapping.pixel_to_board(pixel, CUBE_SIZE_MM-2.0))
        arm.grip()
        time.sleep(1.0)
        print(arm.move_to(*mapping.pixel_to_board(pixel, LIFT_Z)))
        arm.move_to(*mapping.pixel_to_board(pixel, CUBE_SIZE_MM))        
        arm.release()
        time.sleep(1.0)
        arm.home()
        print("Done.")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
