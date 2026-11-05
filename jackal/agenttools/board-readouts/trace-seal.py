#!/usr/bin/env python3
"""The pump's dip over time, from a MAXARM_TRACE log of the calibration app: for every time the pump went on, the
supply against the last second of reads before it, and the cup's height, every read for the grip's wait.

    cd jackal
    MAXARM_TRACE=1 MAXARM_SUCTION_GRAB_MS=3500 ../.venv/bin/python src/calibration/calibration.py \\
        2> agenttools/board-readouts/trace.log
    ../.venv/bin/python agenttools/board-readouts/trace-seal.py agenttools/board-readouts/trace.log [samples]

`samples`: the supply samples per read the log was taken with, `protocol.SUPPLY_SAMPLES` by default.
"""

import ast
import re
import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maxarm import geometry, raw              # noqa: E402
from maxarm.protocol import SUPPLY_SAMPLES   # noqa: E402

LINE = re.compile(r"^\s+([\d.]+)s (.) (.*?)(?: \[\d+ ms\])?$")
BEFORE_S = 1.0
BIN_S = 0.25


def events(path: Path, samples: int = SUPPLY_SAMPLES):
    """(seconds, 'on' | 'off' | 'note', text) and (seconds, 'read', (z, supply)), in order."""
    for line in path.read_text(errors="replace").splitlines():
        match = LINE.match(line)
        if not match:
            continue
        when, marker, body = float(match.group(1)), match.group(2), match.group(3)
        if marker == ">" and body.startswith("nozzle.on()"):
            yield when, "on", body
        elif marker == ">" and body.startswith("nozzle.off()"):
            yield when, "off", body
        elif marker == "#" and ("seal" in body or body.startswith("to ")):
            yield when, "note", body
        elif marker == "<" and body.startswith("#R# (["):
            try:
                pulses, total = ast.literal_eval(body[len("#R# "):])
            except (SyntaxError, ValueError):
                continue
            # A servo that did not answer reads False, which is an int.
            if isinstance(total, int) and all(type(value) is int for value in pulses):
                z = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(pulses))))[2]
                yield when, "read", (z, total / samples)


def main() -> None:
    found = list(events(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else SUPPLY_SAMPLES))
    reads = [(when, value) for when, kind, value in found if kind == "read"]
    for index, (when, kind, text) in enumerate(found):
        if kind == "note":
            print(f"{when:9.3f}  {text}")
        if kind != "on":
            continue
        ended = next((later for later, kind_later, _ in found[index + 1:] if kind_later == "off"), when + 10.0)
        before = [supply for at, (_, supply) in reads if when - BEFORE_S <= at < when]
        after = [(at - when, z, supply) for at, (z, supply) in reads if when < at < ended]
        if not before or not after:
            print(f"{when:9.3f}  pump on: no reads {'before' if not before else 'after'}")
            continue
        level = fmean(before)
        print(f"{when:9.3f}  pump on, z {after[0][1]:.2f}, pump off level {level:.1f} over {len(before)} reads;"
              f" dip and z by {BIN_S:g} s:")
        start = 0.0
        while start < after[-1][0]:
            chunk = [(z, supply) for at, z, supply in after if start <= at < start + BIN_S]
            if chunk:
                print(f"           {start:4.2f}-{start + BIN_S:4.2f} s  dip {level - fmean(s for _, s in chunk):5.1f}"
                      f"  z {fmean(z for z, _ in chunk):6.2f}  ({len(chunk)} reads)")
            start += BIN_S


if __name__ == "__main__":
    main()
