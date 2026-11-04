#!/usr/bin/env python3
"""The cup over the owner's grid paper at desk z 1, point by point: at each, the arm's reading 3 s after it landed
(raw, true and desk) and a live frame; then the paper alone, the arm at home and at the side, for the grid.
Through the running calibration app (:8080). Writes points.json and frames/ here.

    run.py              (the owner's go, 2026-11-03: the arm over the paper at desk z about 1)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from maxarm import raw  # noqa: E402

APP = "http://localhost:8080"
Z = 1.0
XS = (-80, -40, 0, 40, 80)
YS = (-100, -130, -160, -190, -220)
SIDE = (-190.0, -20.0, 80.0)        # desk: off the paper, for its picture


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(APP + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=120))


def state() -> dict:
    return json.load(urllib.request.urlopen(APP + "/api/state", timeout=10))


def frame(name: str) -> str:
    """One whole frame of the live view, as JPEG, into frames/."""
    with urllib.request.urlopen(APP + "/api/camera/feed", timeout=10) as feed:
        data = b""
        while data.count(b"\xff\xd9") < 2:
            data += feed.read(65536)
    end = data.rfind(b"\xff\xd9")
    start = data.rfind(b"\xff\xd8", 0, end)
    path = HERE / "frames" / f"{name}.jpg"
    path.write_bytes(data[start:end + 2])
    return path.name


def land(x: float, y: float, z: float) -> dict:
    target = {"x": x, "y": y, "z": z}
    outcome = post("/api/camera/move", target)
    time.sleep(3)
    post("/api/camera/move", target)        # the same again: nothing moves, the servos read afresh
    now = state()
    true = now["measured"]
    return {"status": outcome["status"], "sent_desk": [x, y, z], "desk": now["camera"]["measured"],
            "true": true, "raw": list(raw.raw_position(tuple(true)))}


def main() -> None:
    (HERE / "frames").mkdir(exist_ok=True)
    rows = []
    for row, y in enumerate(YS):
        for x in (XS if row % 2 == 0 else XS[::-1]):
            point = land(float(x), float(y), Z)
            point["frame"] = frame(f"p{x:+04d}{y:+04d}")
            point["taken"] = time.strftime("%H:%M:%S")
            rows.append(point)
            print(f"({x:+4d}, {y:+4d}): {point['status']}, raw {[round(v, 2) for v in point['raw']]}", flush=True)
            (HERE / "points.json").write_text(json.dumps({"z": Z, "points": rows}, indent=1))
    post("/api/reset", {})
    time.sleep(2)
    home = frame("paper-home")
    post("/api/camera/move", dict(zip("xyz", SIDE)))
    time.sleep(2)
    side = frame("paper-side")
    (HERE / "points.json").write_text(json.dumps({"z": Z, "points": rows, "paper": [home, side]}, indent=1))
    print("paper frames:", home, side)


if __name__ == "__main__":
    main()
