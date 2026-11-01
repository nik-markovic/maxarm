#!/usr/bin/env python3
"""CALIBRATE, RECORD POINT and SAVE CALIBRATION through the app's requests, on the fake arm and a still frame: that
the new code runs, and what it writes. Nothing about the arm or a camera follows from it.

    .venv/bin/python impala/agenttools/height/drive-save.py [port]

The frame is the set-aside CALIBRATE frame (config/camera-frame-previous.png) and the cubes' points its own; the
red cube is recorded a few times with the fake arm a little off it each time. A scratch config, removed after. The
limits and the states before and after SAVE go to /tmp/height-replies.json, for drive-page.mjs and the Fine Tune
page's script.
"""

import json
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(ROOT / "agenttools")]

import fake_arm                                      # noqa: E402
from fake_camera import fake_camera                 # noqa: E402

CONFIG = ROOT / "config"
OFFSETS = [(0.0, 0.0), (1.5, -1.0), (-1.0, 2.0), (2.0, 1.5)]


def call(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=300) as reply:
        return json.loads(reply.read())


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    fake_arm.TRACKING_GAIN = 1.0
    frame = cv2.imread(str(CONFIG / "camera-frame-previous.png"))
    with tempfile.TemporaryDirectory(prefix="impala-height-") as scratch, fake_arm.fake_pyserial(), \
            fake_camera(frame):
        scratch = Path(scratch)
        for name in ("desk.json", "desk-points.json", "camera-view.json"):
            shutil.copy2(CONFIG / name, scratch / name)
        shutil.copy2(CONFIG / "camera-points-previous.json", scratch / "camera-points.json")
        import importlib.util
        spec = importlib.util.spec_from_file_location("calibration_app", ROOT / "calibration" / "calibration.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        threading.Thread(target=app.run, args=("127.0.0.1", port, scratch), daemon=True).start()
        for _ in range(100):
            try:
                if call(port, "/api/state")["connection"]["status"] == "ready":
                    break
            except Exception:
                pass
            time.sleep(0.3)
        print("CALIBRATE:", call(port, "/api/camera/calibrate", {}))
        show(call(port, "/api/state")["tests"]["refine"])
        red = json.loads((scratch / "camera-points.json").read_text())["cubes"]["red"]
        for dx, dy in OFFSETS:
            print("move:", call(port, "/api/camera/move", {"x": red["x"] + dx, "y": red["y"] + dy, "z": 42.0})["status"])
            print("RECORD POINT:", call(port, "/api/tests/record", {}))
        recorded = call(port, "/api/state")
        show(recorded["tests"]["refine"])
        print("SAVE:", call(port, "/api/tests/save", {}))
        state = call(port, "/api/state")
        show(state["tests"]["refine"])
        replies = {"limits": call(port, "/api/limits"), "states": [recorded, state], "clicks": ["save"],
                   "watch": ["points", "refine-fit", "refine-saved", "refine-count"]}
        Path("/tmp/height-replies.json").write_text(json.dumps(replies))
        print("camera page:", state["camera"]["calibration"]["summary"])
        text = (scratch / "camera-refine-points.json").read_text()
        print(f"camera-refine-points.json: {len(text)} bytes, {text.count(chr(10))} lines; point 4 begins:")
        start = text.index('"id": 4')
        print(text[start - 6:start + 700])


def show(refine):
    print("  fit:", refine["fit"], "| problem:", refine["problem"])
    for row in refine["points"]:
        print(f"  {row['id']:2d} {row['cube'] or '':5s} off {row['off']} height {row['height']} seen {row['seen']}")


if __name__ == "__main__":
    main()
