#!/usr/bin/env python3
"""True z above the desk: how far the arm over-reads heights, from touches on things of known height.

    ../.venv/bin/python agenttools/fit-true-z.py                  # config/desk-points.json
    ../.venv/bin/python agenttools/fit-true-z.py path/to/config

True z by reach alone, raw z + 0.05 x (reach - 207), makes the flat desk read level, 44. A touch
on top of something of known height above the desk then shows how far that is off up there, and
which height term the library's formula (`maxarm/raw.py`) needs. Points come from the desk page's
list -- those within 20 mm of the desk are desk touches, the rest stacks of 40 mm cubes, as many
cubes as the height says -- and from the owner's 120 mm stack touches below, which are kept here
as they are off the list. The owner's card touches and the reset pose (work/KINEMATICS.md 1.1 and
3) are never fitted, only checked. No arm needed; nothing moves.
"""

import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "calibration")]

from maxarm import raw                                       # noqa: E402
from desk import store                                       # noqa: E402

DESK_Z = 44.0
CUBE_MM = 40.0
# The owner's 2026-10-19 touches, x = 0, firmware z in whole millimetres. A card was touched
# 0.5-1 mm more lightly than the desk (KINEMATICS 1.1), so it reads that much taller.
CARD_LIGHTER_MM = (0.5, 0.75, 1.0)
CARDS = [(53.98, y, z) for y, z in ((-260, 98), (-220, 98), (-200, 98), (-180, 100), (-160, 101),
                                     (-140, 103), (-120, 104), (-100, 104), (-80, 105))]
CARDS += [(85.60, -80, 135.0), (85.60, -260, 132.5)]
# Reset pose: servos at 500, 500, 500, which the firmware takes for (0, -162.9, 212.8); the cup tip
# 160 mm over the desk in free air, about 1.5 mm more than a snug desk touch squeezes it.
HOME = (162.9, 212.8, 161.5)
# The owner's touches on a 120 mm stack of three cubes, 2026-10-20, x = 0: (number, y, raw z as read).
# Taken off the touch list once fitted, as the desk plane goes through every point there (KINEMATICS 4.7).
STACK_120 = [(77, -239.52, 166.184), (78, -230.09, 165.805), (79, -219.73, 166.173), (80, -210.34, 167.593),
             (81, -199.77, 167.431), (82, -190.19, 169.160), (83, -180.6, 168.980), (84, -160.66, 168.967),
             (85, -150.27, 170.017), (86, -140.27, 169.697), (87, -130.52, 170.664), (88, -119.56, 169.342),
             (89, -110.4, 171.100)]

Form = Callable[[np.ndarray, np.ndarray], np.ndarray]
FORMS: Dict[str, Form] = {
    "flat share of the height": lambda h, dr: np.column_stack([h]),
    "flat share, by reach": lambda h, dr: np.column_stack([h, h * dr]),
    "height squared": lambda h, dr: np.column_stack([h ** 2]),
    "height squared, by reach": lambda h, dr: np.column_stack([h ** 2, h ** 2 * dr]),
}


def base_true(raw_z: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """True z by reach alone, the library's formula without its height term."""
    return raw_z + raw.TRUE_Z_PER_MM_OF_REACH * (reach - raw.LEVEL_REACH_MM)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "config"
    points, _ = store.load_points(config)
    listed = {point.id for point in points}
    xyz = np.array([raw.raw_position(point.target) for point in points]
                   + [(0.0, y, z) for number, y, z in STACK_120 if number not in listed])
    true_now = np.array([raw.true_position(tuple(row)) for row in xyz])
    reach = np.hypot(xyz[:, 0], xyz[:, 1])
    over_desk = base_true(xyz[:, 2], reach) - DESK_Z
    is_desk = np.abs(over_desk) < 20.0
    height = np.where(is_desk, 0.0, CUBE_MM * np.round(over_desk / CUBE_MM))
    over = over_desk - height                   # how much higher than it really is the touch reads, by reach alone
    left_now = true_now[:, 2] - DESK_Z - height

    slope, intercept = np.polyfit(reach[is_desk], xyz[is_desk, 2], 1)
    print(f"{is_desk.sum()} desk touches from {config / store.POINTS_FILE}, {(~is_desk).sum()} on cubes")
    print(f"  desk by reach alone: {DESK_Z + over[is_desk].mean():.2f}, {rms(over[is_desk]):.2f} mm RMS, "
          f"worst {np.abs(over[is_desk]).max():.2f}")
    print(f"  the reach line refitted on the desk: {-slope:.4f} per mm, level at "
          f"{(intercept - DESK_Z) / -slope:.1f} mm (in use {raw.TRUE_Z_PER_MM_OF_REACH}, {raw.LEVEL_REACH_MM:.0f})")
    for level in sorted(set(height)):
        mine = height == level
        line = f"  {level:.0f} mm up ({mine.sum()} touches): "
        if level:
            line += (f"by reach alone {over[mine].mean():+.2f} too high, {over[mine].min():+.2f} to "
                     f"{over[mine].max():+.2f}; ")
        print(line + f"with the library's formula {left_now[mine].mean():+.2f}, {rms(left_now[mine]):.2f} mm RMS, "
              f"worst {np.abs(left_now[mine]).max():.2f}")

    fit_on = ~is_desk
    card_reach = np.array([abs(y) for _, y, _ in CARDS], dtype=float)
    card_over = base_true(np.array([z for _, _, z in CARDS]), card_reach) - DESK_Z
    card_height = np.array([size for size, _, _ in CARDS])
    home_over = base_true(np.array([HOME[1]]), np.array([HOME[0]]))[0] - DESK_Z - HOME[2]
    print("\nfitted on the cube touches; the cards and the reset pose only checked:")
    for name, form in FORMS.items():
        dr = reach - raw.LEVEL_REACH_MM
        coef, *_ = np.linalg.lstsq(form(height[fit_on], dr[fit_on]), over[fit_on], rcond=None)
        left = over[fit_on] - form(height[fit_on], dr[fit_on]) @ coef
        out = []
        for index in np.where(fit_on)[0]:
            keep = fit_on.copy()
            keep[index] = False
            one, *_ = np.linalg.lstsq(form(height[keep], dr[keep]), over[keep], rcond=None)
            out.append(over[index] - (form(height[[index]], dr[[index]]) @ one)[0])
        predict = lambda h, r: form(np.atleast_1d(h), np.atleast_1d(r) - raw.LEVEL_REACH_MM) @ coef
        cards = {lighter: card_over - (card_height + lighter) - predict(card_height + lighter, card_reach)
                 for lighter in CARD_LIGHTER_MM}
        small = card_height < 60
        print(f"  {name}: {' '.join(f'{c:.3g}' for c in coef)}")
        print(f"    cubes {rms(left):.2f} mm RMS (each left out in turn {rms(np.array(out)):.2f}), worst "
              f"{np.abs(left).max():.2f}; 54 mm card off by "
              + " / ".join(f"{cards[lighter][small].mean():+.2f}" for lighter in CARD_LIGHTER_MM)
              + f" for a {' / '.join(str(lighter) for lighter in CARD_LIGHTER_MM)} mm lighter touch; "
              f"reset pose {home_over - predict(HOME[2], HOME[0])[0]:+.1f}")


if __name__ == "__main__":
    main()
