#!/usr/bin/env python3
"""The calibration app on the fake board: try the pages with no arm plugged in.

    ../.venv/bin/python agenttools/serve-fake.py                 # then open http://localhost:8080
    ../.venv/bin/python agenttools/serve-fake.py --port 8081
    ../.venv/bin/python agenttools/serve-fake.py --real-camera   # the camera that is plugged in, the arm fake
    ../.venv/bin/python agenttools/serve-fake.py --frame config/camera-frame.png   # the owner's cubes, as taken
    ../.venv/bin/python agenttools/serve-fake.py --frame agenttools/tile-shots/still-1.png \
        --camera-json agenttools/tile-shots/still-camera.json                  # the Tiles page on real tiles

The library still waits out every move, so the pages run at the real arm's
pace; the fake arm goes exactly where it is sent. The camera is a still frame
(`fake_camera.py`) unless --real-camera: by default three cubes drawn at the
places the camera page suggests (`cube_scene.py`), so CALIBRATE has something
to fit. config/ is copied to a scratch directory first, so nothing written
touches the real one; --camera-json puts another camera.json there, one that
fits the frame given.
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

import cube_scene                                             # noqa: E402
import fake_arm                                               # noqa: E402
from fake_camera import fake_camera                           # noqa: E402
from maxarm.spaces import CONFIG_DIR                          # noqa: E402


def load_app():
    spec = importlib.util.spec_from_file_location("calibration_app", ROOT / "calibration" / "calibration.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--real-camera", action="store_true", help="the camera that is plugged in, not a still frame")
    parser.add_argument("--frame", type=Path, help="the still frame the fake camera sees, not the drawn cubes")
    parser.add_argument("--camera-json", type=Path, help="the camera.json the app uses, one that fits the frame")
    args = parser.parse_args()
    fake_arm.TRACKING_GAIN = 1.0
    frame = cv2.imread(str(args.frame)) if args.frame else cube_scene.render().frame
    if not args.real_camera and frame is None:
        parser.error(f"cannot read {args.frame}")
    camera = contextlib.nullcontext() if args.real_camera else fake_camera(frame)
    with tempfile.TemporaryDirectory(prefix="impala-config-") as scratch, fake_arm.fake_pyserial(), camera:
        for file in CONFIG_DIR.glob("*.json"):
            shutil.copy2(file, scratch)
        if args.camera_json:
            shutil.copy2(args.camera_json, Path(scratch) / "camera.json")
        print(f"config copied to {scratch}; it goes when this stops")
        load_app().run("127.0.0.1", args.port, Path(scratch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
