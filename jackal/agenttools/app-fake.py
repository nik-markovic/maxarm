#!/usr/bin/env python3
"""The /IOTCONNECT app on the fake board, without the cloud, its camera a still frame: to see that it runs.

    ../.venv/bin/python agenttools/app-fake.py                          # detect, place TEAM, find-cube, reader
    ../.venv/bin/python agenttools/app-fake.py "detect" "place MAT"     # these commands, typed for it

`src/main.py --no-iotc`, its console's lines given here. The fake arm goes
exactly where it is sent, at the real arm's pace, and says nothing about the real
one; the camera is the owner's still-1 (`tile-shots/`), with the camera.json it
was taken with. config/ is copied to a scratch directory first, so nothing written
touches the real one.
"""

import importlib.util
import io
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path[:0] = [str(SRC), str(HERE)]

import fake_arm                                               # noqa: E402
from fake_camera import fake_camera                           # noqa: E402
from maxarm.spaces import CONFIG_DIR                          # noqa: E402

STILL = HERE / "tile-shots" / "still-1.png"
STILL_CAMERA = HERE / "tile-shots" / "still-camera.json"
COMMANDS = ["detect", "place TEAM", "find-cube", "reader-location left", "reader-location", "place"]


class TypedLines(io.StringIO):
    """The console, typed for: these lines, then the end."""

    def isatty(self) -> bool:
        return True


def main() -> int:
    lines = sys.argv[1:] or COMMANDS
    spec = importlib.util.spec_from_file_location("jackal_main", SRC / "main.py")
    app_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_main)
    fake_arm.TRACKING_GAIN = 1.0
    with tempfile.TemporaryDirectory(prefix="jackal-app-") as scratch, fake_arm.fake_pyserial(), \
            fake_camera(cv2.imread(str(STILL))):
        for file in CONFIG_DIR.glob("*.json"):
            shutil.copy2(file, scratch)
        shutil.copy2(STILL_CAMERA, Path(scratch) / "camera.json")
        (Path(scratch) / "camera-view.json").unlink(missing_ok=True)
        app_main.CONFIG_DIR = Path(scratch)
        sys.argv = ["main.py", "--no-iotc"]
        sys.stdin = TypedLines("".join(f"{line}\n" for line in lines))
        print(f"config copied to {scratch}; typing: {' / '.join(lines)}", flush=True)
        app_main.main()
        written = Path(scratch) / "appcfg.json"
        print(f"appcfg.json: {written.read_text().strip() if written.exists() else 'not written'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
