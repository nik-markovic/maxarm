#!/usr/bin/env python3
"""What drive.py's runs say: landings against the whole pulses commanded, overshoot, settling, small steps, base turns.

    ../.venv/bin/python agenttools/mechanics/analyze.py [runs/*.jsonl ...]     (default: every run)

Pulses read minus pulses commanded: + past the command in the direction of the leg, - short of it.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from maxarm import geometry, raw                                   # noqa: E402
from maxarm.spaces import Spaces                                   # noqa: E402

HERE = Path(__file__).resolve().parent
SETTLED_S = 1.0          # the last this many seconds of a leg's reads are where it settled
SPACES = Spaces()


def desk_z(pulses) -> float:
    arm = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(float(p) for p in pulses))))
    return SPACES.arm.to_desk(arm)[2]


def settled(reads):
    last = [r for r in reads if r[0] >= reads[-1][0] - SETTLED_S]
    return [st.mean(r[i] for r in last) for i in (1, 2, 3)]


def despiked(reads):
    """Each read's pulses replaced by the median of it and its two neighbours: a single read far off both
    neighbours -- a bad read, the arm cannot go there and back in a read's time -- is dropped."""
    out = [reads[0]]
    for before, read, after in zip(reads, reads[1:], reads[2:]):
        out.append([read[0]] + [sorted((before[i], read[i], after[i]))[1] for i in (1, 2, 3)])
    return out + [reads[-1]] if len(reads) > 1 else out


def spikes(reads, by=2):
    """How many single reads were more than BY pulses off both neighbours, on any servo."""
    return sum(1 for before, read, after in zip(reads, reads[1:], reads[2:])
               if any(abs(read[i] - before[i]) > by and abs(read[i] - after[i]) > by for i in (1, 2, 3)))


def settle_time(reads, window_s=0.5, within=0.5):
    """When the cup stopped moving: the last time a rolling WINDOW_S mean of any servo was further than WITHIN
    pulses from where it settled. A rolling mean, so that one pulse of flicker is not taken for movement."""
    reads = despiked(reads)
    final = settled(reads)
    last = 0.0
    for read in reads:
        window = [r for r in reads if read[0] - window_s < r[0] <= read[0]]
        if len(window) >= 3 and any(abs(st.mean(r[i] for r in window) - final[i - 1]) > within for i in (1, 2, 3)):
            last = read[0]
    return last


def q(values, share):
    values = sorted(values)
    return values[min(len(values) - 1, int(share * (len(values) - 1) + 0.5))]


def spread(values, digits=1):
    if not values:
        return "-"
    f = f"{{:+.{digits}f}}"
    return f"{f.format(st.median(values))} [{f.format(q(values, 0.1))}..{f.format(q(values, 0.9))}]"


def direction(before, commanded):
    return [0 if before is None else (1 if c > b + 0.5 else -1 if c < b - 0.5 else 0)
            for b, c in zip(before or [None] * 3, commanded)]


def load(paths):
    entries = []
    for path in paths:
        for line in open(path):
            entry = json.loads(line)
            if entry.get("reads"):
                entries.append(entry)
    return entries


def descents(entries) -> None:
    print("== descent and ascent legs: settled read minus command, pulses; heights over the desk, mm ==")
    groups = defaultdict(list)
    for e in entries:
        if e["kind"] not in ("descent40", "descent2", "ascent40"):
            continue
        reach = (e["arm"][0] ** 2 + e["arm"][1] ** 2) ** 0.5
        band = "inner <140" if reach < 140 else "outer >230" if reach > 230 else "middle"
        groups[(e["kind"], band)].append(e)
        groups[(e["kind"], "all")].append(e)
    for (kind, band), legs in sorted(groups.items()):
        miss = [[s - c for s, c in zip(settled(e["reads"]), e["commanded"])] for e in legs]
        target = [e["desk"][2] for e in legs]
        landed = [desk_z(settled(e["reads"])) - t for e, t in zip(legs, target)]
        # Overshoot: the furthest the cup went past the target in the leg's own direction (down for a descent).
        pick = max if kind.startswith("ascent") else min
        deepest = [pick(desk_z(r[1:]) for r in despiked(e["reads"])) - t for e, t in zip(legs, target)]
        aim = [desk_z(e["commanded"]) - t for e, t in zip(legs, target)]
        moved_late = []
        for e in legs:
            end = e["move_ms"] / 1000.0
            moved_late.append(settle_time(e["reads"]) - end)
        print(f"{kind:9s} {band:10s} n={len(legs):3d}  base {spread([m[0] for m in miss])}  s2 {spread([m[1] for m in miss])}"
              f"  s3 {spread([m[2] for m in miss])}")
        print(f"{'':20s}        height landed-target {spread(landed, 2)} mm (whole-pulse aim alone {spread(aim, 2)});"
              f" furthest during the leg {spread(deepest, 2)}; last change after the move ends {spread(moved_late, 2)} s")
    again = defaultdict(list)
    for e in entries:
        if e["kind"] == "descent2":
            again[e["label"]].append(desk_z(settled(e["reads"])))
    pairs = [abs(v[1] - v[0]) for v in again.values() if len(v) >= 2]
    if pairs:
        print(f"descent to 2 mm twice at the same place: heights apart median {st.median(pairs):.2f}, "
              f"90% {q(pairs, 0.9):.2f}, worst {max(pairs):.2f} mm (n={len(pairs)})")


def glitches(entries) -> None:
    reads = sum(len(e["reads"]) for e in entries)
    bad = sum(spikes(e["reads"]) for e in entries)
    worst = max((max(abs(r[i] - b[i]) for b, r, a in zip(e["reads"], e["reads"][1:], e["reads"][2:]) for i in (1, 2, 3)
                     if abs(r[i] - b[i]) > 2 and abs(r[i] - a[i]) > 2) for e in entries
                 if spikes(e["reads"])), default=0)
    print(f"== single bad reads (one read >2 pulses off both neighbours): {bad} in {reads} reads, "
          f"1 in {reads // max(bad, 1)}; the largest {worst} pulses ==")


def steps(entries) -> None:
    print("== single-servo steps: did it move, and where it ended against the command ==")
    rows = defaultdict(list)
    for e in entries:
        if e["kind"] != "step" or e.get("before") is None:
            continue
        servo, step = e["servo"], e["step"]
        final = settled(e["reads"])
        moved = (final[servo] - e["before"][servo]) * (1 if step > 0 else -1)
        miss = (final[servo] - e["commanded"][servo]) * (1 if step > 0 else -1)
        level = "z5" if e["desk"][2] < 20 else "z40"
        rows[(servo + 1, abs(step), "up" if step > 0 else "down", level)].append((moved, miss))
    for key in sorted(rows):
        v = rows[key]
        print(f"servo {key[0]} step {key[1]} pulses {key[2]:4s} {key[3]:3s} n={len(v):2d}: moved {spread([m for m, _ in v])}"
              f"  ended vs command {spread([x for _, x in v])}")


def turns(entries) -> None:
    print("== base turned onto a place: base read minus command, along the turn (+ past, - short) ==")
    rows = defaultdict(list)
    for e in entries:
        if e["kind"] != "turn":
            continue
        final = settled(e["reads"])
        along = (final[0] - e["commanded"][0]) * (-e["side"])     # side -1: came from below, turning +
        level = "z5" if e["desk"][2] < 20 else "z40"
        rows[(e["turn"], "turning +" if e["side"] < 0 else "turning -", level)].append(along)
    for key in sorted(rows):
        print(f"turn {key[0]:3d} pulses {key[1]} {key[2]:3s} n={len(rows[key]):2d}: {spread(rows[key])}")


def between(entries) -> None:
    print("== targets between base pulses: base landed minus the exact target; sideways mm at the cup ==")
    import math
    rows = defaultdict(list)
    downs = defaultdict(list)
    for e in entries:
        if e["kind"] not in ("between", "between_down"):
            continue
        reach = math.hypot(e["arm"][0], e["arm"][1])
        mm = reach * math.radians(0.24)
        final = settled(e["reads"])
        off = final[0] - e["exact_base"]
        turning = "turning +" if e["side"] < 0 else "turning -"
        key = (round(reach), turning)
        if e["kind"] == "between":
            rows[key].append((e["quarter"], off, off * mm, final[0] - e["commanded"][0]))
        else:
            downs[key].append(final[0] - e["before"][0] if e.get("before") else 0.0)
    for key in sorted(rows):
        v = rows[key]
        print(f"reach {key[0]} {key[1]} n={len(v)}: base vs exact {spread([o for _, o, _, _ in v], 2)} pulses ="
              f" {spread([m for _, _, m, _ in v], 2)} mm; vs whole pulse commanded {spread([c for *_, c in v], 2)}")
        print("    by quarter (exact target = commanded whole pulse + quarter/4 - rounding):",
              " ".join(f"{q}:{o:+.2f}" for q, o, _, _ in sorted(v)))
        if downs[key]:
            print(f"    base moved by the descent to 2 mm: {spread(downs[key], 2)} pulses")


def pickups(entries) -> None:
    import math
    downs = [e for e in entries if e["kind"] == "pickup_down" and "d3" not in e]
    if not downs:
        return
    print("== pickups: height landed over the 2 mm target (+ too shallow), reach error (+ too far out), by extras ==")

    def landed(e):
        final = settled(e["reads"])
        arm = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(final))))
        want = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(e["exact_low"]))))
        return SPACES.arm.to_desk(arm)[2] - 2.0, math.hypot(arm[0], arm[1]) - math.hypot(want[0], want[1]), final

    groups = defaultdict(list)
    for e in downs:
        h, r, final = landed(e)
        for key in ((e["extra2"], e["extra3"], "all", "all"), (e["extra2"], e["extra3"], e["reach"], "all"),
                    (e["extra2"], e["extra3"], "all", e["drop"])):
            groups[key].append((h, r, final[1] - e["exact_low"][1], final[2] - e["exact_low"][2]))
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), str(k[3]))):
        v = groups[key]
        hs = [h for h, *_ in v]
        what = "all" if key[2] == "all" and key[3] == "all" else (f"reach {key[2]}" if key[3] == "all" else f"drop {key[3]:.0f}")
        print(f"extras s2 {key[0]:+d} s3 {key[1]:+d} {what:9s} n={len(v):2d}: height {spread(hs, 2)} worst shallow {max(hs):+.2f}"
              f" (>1 mm: {sum(h > 1.0 for h in hs)}); reach {spread([r for _, r, _, _ in v], 2)};"
              f" s2 vs exact {spread([a for *_, a, _ in v])} s3 {spread([b for *_, b in v])}")
    print("== pickups: the base against its exact target, by approach side and aim past ==")
    base = defaultdict(list)
    for e in [e for e in entries if e["kind"] == "pickup_down"]:
        final = settled(e["reads"])[0]
        base[(e["side"], e["past"])].append((final - e["exact_base"]) * e["side"])
    for key in sorted(base):
        v = base[key]
        name = "turning -" if key[0] > 0 else "turning +"
        print(f"{name} aimed {key[1]} past n={len(v):2d}: short of exact {spread(v, 2)} pulses;"
              f" within 1 pulse {sum(abs(x) <= 1.0 for x in v)}/{len(v)}, within 0.5 {sum(abs(x) <= 0.5 for x in v)}/{len(v)}")


def tuning(entries) -> None:
    import math
    downs = [e for e in entries if e["kind"] == "pickup_down" and "d3" in e]
    if not downs:
        return

    def landed(e):
        final = settled(e["reads"])
        arm = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(final))))
        want = raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(e["exact_low"]))))
        return (SPACES.arm.to_desk(arm)[2] - 2.0, math.hypot(arm[0], arm[1]) - math.hypot(want[0], want[1]),
                final[1] - e["exact_low"][1], final[2] - e["exact_low"][2])

    print("== tuning: by reach and variant (servo 2, servo 3 offsets from the guess): height over target, reach error,"
          " servos against exact ==")
    by = defaultdict(list)
    for e in downs:
        by[(e["reach"], e["d2"], e["d3"])].append((e["extra2"], e["extra3"], *landed(e)))
    for key in sorted(by):
        v = by[key]
        print(f"reach {key[0]:3.0f} d2 {key[1]:+.1f} d3 {key[2]:+.1f} (extras {v[0][0]:+.2f} {v[0][1]:+.2f}) n={len(v)}:"
              f" height {spread([x[2] for x in v], 2)} reach {spread([x[3] for x in v], 2)}"
              f" s2 {spread([x[4] for x in v], 2)} s3 {spread([x[5] for x in v], 2)}")
    print("== tuning: the extras that would zero each servo's miss, by reach (straight line through the variants) ==")
    for reach in sorted({k[0] for k in by}):
        fits = []
        for servo, index in ((2, 4), (3, 5)):
            pts = [(x[0] if servo == 2 else x[1], x[index]) for k, v in by.items() if k[0] == reach
                   and (k[2] == 0.0 if servo == 2 else k[1] == 0.0) for x in v]
            xs, ys = [a for a, _ in pts], [b for _, b in pts]
            mx, my = st.mean(xs), st.mean(ys)
            var = sum((a - mx) ** 2 for a in xs)
            slope = sum((a - mx) * (b - my) for a, b in pts) / var if var else float("nan")
            zero = mx - my / slope if slope and slope == slope else float("nan")
            fits.append(f"servo {servo}: {zero:+.2f} (gain {slope:.2f})")
        print(f"reach {reach:3.0f}: " + "; ".join(fits))
    guess = [landed(e) for e in downs if e["d2"] == 0.0 and e["d3"] == 0.0]
    hs = sorted(h for h, *_ in guess)
    rs = sorted(r for _, r, *_ in guess)
    print(f"== tuning: the guess itself, every reach, n={len(hs)}: height median {st.median(hs):+.2f}, 10% {q(hs, .1):+.2f},"
          f" 90% {q(hs, .9):+.2f}, 95% {q(hs, .95):+.2f}, most shallow {hs[-1]:+.2f}, deepest {hs[0]:+.2f};"
          f" reach median {st.median(rs):+.2f} [{q(rs, .1):+.2f}..{q(rs, .9):+.2f}] ==")
    allh = sorted(landed(e)[0] for e in downs)
    print(f"   every variant, n={len(allh)}: height 90% {q(allh, .9):+.2f}, most shallow {allh[-1]:+.2f}")


def main() -> None:
    paths = sys.argv[1:] or sorted((HERE / "runs").glob("*.jsonl"))
    entries = load(paths)
    glitches(entries)
    descents(entries)
    steps(entries)
    turns(entries)
    between(entries)
    pickups(entries)
    tuning(entries)


if __name__ == "__main__":
    main()
