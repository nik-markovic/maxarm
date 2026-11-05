#!/usr/bin/env python3
"""The owner's seal test, through the running calibration app: the cup down to z 42 true (lands near 44, on the desk)
and up to 50, the supply read with the pump off and on at each. The pump goes off before each lift, so nothing
pulls on the desk. Every read lands in readouts.jsonl, tagged.

    ../.venv/bin/python agenttools/board-readouts/seal-test.py [cycles]
"""

import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:8080/api/"
LOW, HIGH = 42.0, 50.0
OFF_READS, ON_READS = 3, 4
PUMP_SETTLE_S = 1.5
VENT_S = 2.0


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.load(reply)


def move(x: float, y: float, z: float) -> None:
    reply = post("move", {"x": x, "y": y, "z": z})
    if reply["status"] != "done":
        raise SystemExit(f"move to z {z} {reply['status']}: {reply['detail']}")
    time.sleep(1.0)


def reads(tag: str, count: int) -> None:
    for _ in range(count):
        found = post("desk/readout", {"tag": tag, "all": False})["readout"]
        supply = found["supply"]
        print(f"  {tag:14s} pump {'on ' if found['suction'] else 'off'} supply {supply['mean']:7.1f}", flush=True)


def pump(is_on: bool) -> None:
    post("suction", {"on": is_on})
    time.sleep(PUMP_SETTLE_S if is_on else VENT_S)


def main() -> None:
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with urllib.request.urlopen(API + "state", timeout=10) as reply:
        state = json.load(reply)
    x, y = state["sent"][0], state["sent"][1]
    if state["suction"]:
        pump(False)
    for cycle in range(1, cycles + 1):
        for height, name in ((LOW, "low"), (HIGH, "high")):
            move(x, y, height)
            reads(f"seal{cycle} {name} off", OFF_READS)
            pump(True)
            reads(f"seal{cycle} {name} on", ON_READS)
            pump(False)
    print("done: the arm at z", HIGH, "pump off")


if __name__ == "__main__":
    main()
