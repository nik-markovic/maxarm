#!/usr/bin/env python3
"""The calibration app on the fake board: try the pages with no arm plugged in.

    ../.venv/bin/python agenttools/serve-fake.py                 # then open http://localhost:8080
    ../.venv/bin/python agenttools/serve-fake.py --port 8081
    ../.venv/bin/python agenttools/serve-fake.py --real-camera   # the camera that is plugged in, the arm fake
    ../.venv/bin/python agenttools/serve-fake.py --frame shot.png

The library still waits out every move, so the pages run at the real arm's
pace; the fake arm goes exactly where it is sent. The camera is a still frame
(`fake_camera.py`) unless --real-camera. config/ is copied to a scratch
directory first, so nothing written touches the real one.
"""

import argparse
import contextlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

import fake_arm                                               # noqa: E402
from fake_camera import fake_camera                           # noqa: E402
from maxarm.spaces import CONFIG_DIR                          # noqa: E402

FRAME = ROOT / "training" / "tiles" / "baseline" / "tiles-1.png"


def load_app():
    spec = importlib.util.spec_from_file_location("calibration_app", ROOT / "calibration" / "calibration.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--real-camera", action="store_true", help="the camera that is plugged in, not a still frame")
    parser.add_argument("--frame", type=Path, default=FRAME, help="the still frame the fake camera sees")
    args = parser.parse_args()
    fake_arm.TRACKING_GAIN = 1.0
    frame = None if args.real_camera else cv2.imread(str(args.frame))
    if not args.real_camera and frame is None:
        parser.error(f"cannot read {args.frame}")
    camera = contextlib.nullcontext() if args.real_camera else fake_camera(frame)
    with tempfile.TemporaryDirectory(prefix="gecko-config-") as scratch, fake_arm.fake_pyserial(), camera:
        for file in CONFIG_DIR.glob("*.json"):
            shutil.copy2(file, scratch)
        print(f"config copied to {scratch}; it goes when this stops")
        load_app().run("127.0.0.1", args.port, Path(scratch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
