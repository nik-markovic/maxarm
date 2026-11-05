#!/usr/bin/env python3
"""The seal at several heights, through the running calibration app, the pump on briefly at each: from 52 down to
each height, 3 reads with the pump off, the pump on, 5 reads, the pump off. Every read to seal-run.jsonl, labelled.

    ../.venv/bin/python agenttools/board-readouts/seal-run.py [rounds]
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8080/api/"
OUT = Path(__file__).with_name("seal-run.jsonl")
ABOVE = 52.0
HEIGHTS = (52.0, 42.0, 43.0, 44.0, 45.0, 46.0)
OFF_READS, ON_READS = 3, 5


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.load(reply)


def move(x: float, y: float, z: float) -> list:
    reply = post("move", {"x": x, "y": y, "z": z})
    if reply["status"] != "done":
        raise SystemExit(f"move to z {z} {reply['status']}: {reply['detail']}")
    time.sleep(0.8)
    return reply["state"]["measured"]


def reads(label: dict, count: int) -> tuple:
    found = []
    with OUT.open("a") as out:
        for index in range(count):
            reply = post("desk/readout", {})
            read = reply["readout"]
            row = {**label, "index": index, "time": read["time"], "suction": read["suction"], "mean": read["mean"],
                   "measured": reply["state"]["measured"]}
            out.write(json.dumps(row) + "\n")
            found.append(read["mean"])
    return found, row["measured"]


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with urllib.request.urlopen(API + "state", timeout=10) as reply:
        state = json.load(reply)
    x, y = state["sent"][0], state["sent"][1]
    if state["suction"]:
        post("suction", {"on": False})
    for round_ in range(1, rounds + 1):
        for height in HEIGHTS:
            move(x, y, ABOVE)
            if height != ABOVE:
                move(x, y, height)
            label = {"round": round_, "sent": height}
            off, landed = reads({**label, "phase": "off"}, OFF_READS)
            post("suction", {"on": True})
            on, _ = reads({**label, "phase": "on"}, ON_READS)
            post("suction", {"on": False})
            reference = sum(off) / len(off)
            dips = " ".join(f"{reference - value:5.1f}" for value in on)
            print(f"round {round_} sent {height:4.0f} read z {landed[2] if landed else float('nan'):5.1f}"
                  f"  off {reference:7.1f}  dips {dips}", flush=True)
    move(x, y, ABOVE)
    print("done: the arm at z", ABOVE, "pump off")


if __name__ == "__main__":
    main()
