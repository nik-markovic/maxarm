#!/usr/bin/env python3
"""Run the cube detector over one image and draw what it found.

The detector itself is `calibration/detect.py` -- this is the command line that
looks at it, and the overlay is the only way to judge a detection by eye.

    ./agenttools/find-cube.py files/cubes1.jpeg -o /tmp/overlay.jpg
    ./agenttools/find-cube.py files/red-cube.jpeg -c red

`blob_err` is the column to keep an eye on: it is how far the cube's coloured
blob centre sits from its base centre, i.e. how wrong this would be if someone
went back to using the blob. It reads 25-50 mm on the sample frames.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))

import cv2                                                          # noqa: E402
from detect import COLOUR_WINDOWS, annotate, find_cubes             # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-c", "--colours", nargs="+", choices=sorted(COLOUR_WINDOWS),
                        default=sorted(COLOUR_WINDOWS))
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read {args.image}")

    detections = find_cubes(image, args.colours)
    missing = set(args.colours) - {d.colour for d in detections}
    if missing:
        print(f"not found: {', '.join(sorted(missing))}")
    if not detections:
        return 1

    for det in detections:
        across, receding = sorted(det.base_edge_px, reverse=True)
        print(
            f"{det.colour:6} base=({det.base_center[0]:7.1f},{det.base_center[1]:7.1f}) "
            f"blob=({det.silhouette_center[0]:7.1f},{det.silhouette_center[1]:7.1f}) "
            f"edges={across:5.1f}/{receding:5.1f}px "
            f"scale={det.mm_per_px:.3f}/{det.mm_per_px_receding:.3f}mm/px "
            f"elev{'=' if det.is_square_on else '<'}{det.elevation_deg:4.1f}deg "
            f"blob_err={det.ground_offset_mm(det.silhouette_center):3.0f}mm"
        )
    print("\nbase centres are measured; elev/scale assume a square-on cube and read "
          "high when one is turned ('<')")

    if args.out:
        cv2.imwrite(str(args.out), annotate(image, detections))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
