#!/usr/bin/env python3
"""Check `kinematics_reference` against the board, then report the envelope.

Two passes, both motionless:

  1. Agreement. Over a grid spanning the whole workspace, compare the host
     port's verdict and pulses with the board's own `position_to_pulses`.
     A port that disagrees anywhere is not a formula we can put in a library.
  2. Consequences. With agreement established, walk the limits the formula
     implies and print them: which constraint binds where, and r_min/r_max
     per Z plane.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datacollection"))

import kinematics_reference as kin  # noqa: E402
from maxarm_link import MaxArmLink, ReplError  # noqa: E402



def board_pulses(link: MaxArmLink, points):
    """Board verdict per point: a pulse triple, or the exception text."""
    out = []
    for index, (x, y, z) in enumerate(points, 1):
        if index % 50 == 0:
            print(f"  ...{index}/{len(points)}")
        try:
            value = link.evaluate(
                f"[round(v, 2) for v in arm.position_to_pulses(({-x}, {y}, {z}))]"
            )
            out.append(tuple(value))
        except ReplError as exc:
            text = str(exc)
            # The board distinguishes the two failure modes for us, and which
            # one fires is exactly what the host port has to reproduce.
            if "Invalid angle" in text:
                out.append("invalid_angle")
            elif "Unreachable" in text:
                out.append("unreachable")
            else:
                raise
    return out


def host_pulses(points):
    out = []
    for point in points:
        try:
            out.append(tuple(round(v, 2) for v in kin.position_to_pulses(point)))
        except kin.Unreachable as exc:
            out.append("invalid_angle" if "Invalid angle" in str(exc) else "unreachable")
    return out


def build_grid():
    """Spread points across the workspace, deliberately including the edges."""
    points = []
    for z in (50.0, 84.4, 120.0, 160.0, 204.0, 224.0):
        for bearing in (-150.0, -120.0, -90.0, -45.0, 0.0, 45.0, 120.0, 150.0):
            for radius in (55.0, 100.0, 160.0, 220.0, 260.0, 285.0, 300.0):
                angle = math.radians(bearing)
                points.append((round(radius * math.sin(angle), 3),
                               round(-radius * math.cos(angle), 3),
                               z))
    return points


def compare(link):
    points = build_grid()
    print(f"comparing {len(points)} points against the board...")
    board = board_pulses(link, points)
    host = host_pulses(points)

    # A disagreement within a whisker of a joint limit is not a modelling
    # error: the board works in single precision, so right on the boundary the
    # two arithmetics legitimately differ. Those are counted separately.
    EPSILON_DEGREES = 1e-3

    mismatches, boundary = [], []
    for point, b, h in zip(points, board, host):
        differs = (
            b != h if isinstance(b, str) or isinstance(h, str)
            else max(abs(bi - hi) for bi, hi in zip(b, h)) > 0.05
        )
        if not differs:
            continue
        margin = kin.angle_margin(point)
        if abs(margin) < EPSILON_DEGREES:
            boundary.append((point, margin))
        else:
            mismatches.append((point, b, h, margin))

    agreed = len(points) - len(mismatches) - len(boundary)
    print(f"  agree: {agreed}/{len(points)}")
    print(f"  on the boundary (sub-1e-3 deg, single- vs double-precision): "
          f"{len(boundary)}")
    for point, margin in boundary[:4]:
        print(f"    {point}: {margin:+.2e} deg from the limit")
    for point, b, h, margin in mismatches[:15]:
        print(f"  MISMATCH {point}: board={b} host={h} margin={margin:.4f} deg")
    return not mismatches


def report_envelope():
    """What the formula implies, per Z plane."""
    print("\nenvelope implied by the formula (no sag, no desk, no no-fly):")
    print(f"  base fan: +/-120 deg about straight-front, "
          f"i.e. atan2(x, -y) in [-120, +120]")
    print("\n   z     r_min  r_max   binds at r_min      binds at r_max")
    for z in (48, 60, 84, 104, 140, 170, 190, 204, 215, 224):
        radii = [r / 2 for r in range(2 * 25, 2 * 320)]
        ok = [r for r in radii if kin.limit_reason((0.0, -r, float(z))) is None]
        if not ok:
            print(f"  {z:4d}    -- nothing reachable --")
            continue
        lo, hi = min(ok), max(ok)
        why_lo = kin.limit_reason((0.0, -(lo - 0.5), float(z)))
        why_hi = kin.limit_reason((0.0, -(hi + 0.5), float(z)))
        print(f"  {z:4d}   {lo:6.1f} {hi:6.1f}   {why_lo:<18} {why_hi}")

    print("\n  which constraint ever binds, over the whole workspace:")
    seen = {}
    for z in range(44, 232, 2):
        for r in range(26, 320, 2):
            reason = kin.limit_reason((0.0, float(-r), float(z)))
            if reason:
                seen[reason] = seen.get(reason, 0) + 1
    for reason, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<18} {count} grid points")


def main() -> int:
    with MaxArmLink() as link:
        ok = compare(link)
    report_envelope()
    print("\nRESULT:", "formula matches the board exactly"
          if ok else "formula DISAGREES with the board -- do not use")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
