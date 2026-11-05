#!/usr/bin/env python3
"""The seal's timing, through the running calibration app (started with MAXARM_TRACE=1 and a long grip wait, see
trace-seal.py), straight out where the owner saw it go wrong: the pump on and off sent down onto the desk, up in
the air, and sent to 45 from below, as the owner had it. Prints the library's verdict for each; the trace has
every read.

    ../.venv/bin/python agenttools/board-readouts/seal-timing.py
"""

import json
import urllib.request

API = "http://127.0.0.1:8080/api/"
X, Y = 0.0, -164.42
DOWN, AIR, OWNERS = 42.0, 50.0, 45.0


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.load(reply)


def move(x: float, y: float, z: float) -> None:
    reply = post("move", {"x": x, "y": y, "z": z})
    if reply["status"] != "done":
        raise SystemExit(f"move to z {z} {reply['status']}: {reply['detail']}")


def cycles(label: str, count: int) -> None:
    for _ in range(count):
        on = post("suction", {"on": True})
        state = on["state"]
        z = state["measured"][2] if state["measured"] else float("nan")
        print(f"{label:28s} read z {z:5.2f}  {on['detail']}", flush=True)
        post("suction", {"on": False})


def main() -> None:
    with urllib.request.urlopen(API + "state", timeout=10) as reply:
        if json.load(reply)["suction"]:
            post("suction", {"on": False})
    move(X, Y, DOWN)
    cycles(f"sent {DOWN:g} from above", 2)
    move(X, Y, AIR)
    cycles(f"sent {AIR:g}, in the air", 1)
    move(X, Y, DOWN)
    move(X, Y, OWNERS)
    cycles(f"sent {OWNERS:g} from below", 2)
    print(f"done: the arm sent to {OWNERS:g}, the pump off")


if __name__ == "__main__":
    main()
