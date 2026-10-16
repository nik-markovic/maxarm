#!/usr/bin/env python3
"""The calibration app on the fake board: try the page with no arm plugged in.

    ../.venv/bin/python agenttools/serve-fake.py              # then open http://localhost:8080
    ../.venv/bin/python agenttools/serve-fake.py --port 8081

The library still waits out every move, so the page runs at the real arm's
pace; the fake arm goes exactly where it is sent. config/ is copied to a
scratch directory first, so SAVE never touches the real desk.json.
"""

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

import fake_arm                                               # noqa: E402
from maxarm.spaces import CONFIG_DIR                          # noqa: E402


def load_app():
    spec = importlib.util.spec_from_file_location("calibration_app", ROOT / "calibration" / "calibration.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    fake_arm.TRACKING_GAIN = 1.0
    with tempfile.TemporaryDirectory(prefix="falcon-config-") as scratch, fake_arm.fake_pyserial():
        for file in CONFIG_DIR.glob("*.json"):
            shutil.copy2(file, scratch)
        print(f"config copied to {scratch}; it goes when this stops")
        load_app().run("127.0.0.1", args.port, Path(scratch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
