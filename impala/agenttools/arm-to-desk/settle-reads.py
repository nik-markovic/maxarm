#!/usr/bin/env python3
"""settle.py's runs: when the cup stopped, by the camera; and ways of reading the arm, tried on the reads it logged,
against the mean of its last 3 s of reads (the cup still by then, the camera says).

    .venv/bin/python impala/agenttools/arm-to-desk/settle-reads.py settle.json settle-more.json

- app: what the app read (400 ms after the move, the median of 5 reads);
- wait 2.5 s: the mean of the reads from 1.5 to 2.5 s after the move;
- still 1 s: reads until two 1 s spans' means agree within half a pulse each servo, from 1 s after the move, at most
  5 s; their last span's mean.
The reads start 0.7 s after the move here (the app's read comes first), so the spans start there too.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import Space, Spaces, protocol                                 # noqa: E402
from camera import store                                                   # noqa: E402

SPACES = Spaces(IMPALA / "config")
PATCH, SEARCH = 70, 30


def desk(pulses):
    return np.array(SPACES.convert(protocol._true_position(tuple(float(p) for p in pulses)), Space.ARM, Space.DESK))


def cup_track(fit, run, frames, folder):
    """The cup's place in each frame from its place in the last, desk mm along the desk, by template matching."""
    x0, y0 = run["window"][:2]
    centre = fit.desk_to_pixel @ (*run["settled_desk"], 1.0)
    grey = lambda name: cv2.cvtColor(cv2.imread(str(folder / name)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    last = grey(frames[-1][1])
    cx, cy = np.clip(np.round(centre[:2] / centre[2] - (x0, y0)).astype(int), PATCH + SEARCH,
                     np.array(last.shape[::-1]) - PATCH - SEARCH)
    template = last[cy - PATCH:cy + PATCH, cx - PATCH:cx + PATCH]
    x, y, z = run["settled_desk"]
    seen = lambda dx, dy: (lambda p: p[:2] / p[2])(fit.desk_to_pixel @ (x + dx, y + dy, z, 1.0))
    to_mm = np.linalg.inv(np.column_stack((seen(1, 0) - seen(0, 0), seen(0, 1) - seen(0, 0))))
    track = []
    for when, name in frames:
        area = grey(name)[cy - PATCH - SEARCH:cy + PATCH + SEARCH, cx - PATCH - SEARCH:cx + PATCH + SEARCH]
        scores = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, (bx, by) = cv2.minMaxLoc(scores)
        track.append((when - run["done"], np.linalg.norm(to_mm @ (bx - SEARCH, by - SEARCH))))
    return track


def still_read(t, pulses):
    """Reads until two 1 s spans' means agree within half a pulse, from 1 s after the move; at most 5 s."""
    end = 2.0
    while end <= 5.0:
        last = pulses[(t > end - 1.0) & (t <= end)]
        before = pulses[(t > end - 2.0) & (t <= end - 1.0)]
        if len(last) and len(before) and np.all(np.abs(last.mean(axis=0) - before.mean(axis=0)) <= 0.5):
            return last.mean(axis=0), end
        end += 0.25
    return pulses[(t > 4.0) & (t <= 5.0)].mean(axis=0), 5.0


def main():
    fit = store.load_camera(IMPALA / "config")
    rows = []
    for file in sys.argv[1:]:
        data = json.loads((HERE / file).read_text())
        folder = HERE / data.get("frames_dir", "settle-frames")
        for index, run in enumerate(data["runs"]):
            frames = [(when, name) for when, point, name in data["frames"] if point == index and when <= run["final"]]
            track = cup_track(fit, run, frames, folder)
            moving = [after for after, off in track if off > 0.15]
            stopped = max(moving) if moving else None
            after_read = max((off for after, off in track if after >= run["read"] - run["done"] + 0.6), default=0)
            polls = np.array([p for p in run["polls"] if p[1] is not None], float)
            t, pulses = polls[:, 0] - run["done"], polls[:, 1:]
            truth = desk(pulses[t >= t[-1] - 3.0].mean(axis=0))
            app = np.array(run["app_desk"])
            fixed = desk(pulses[(t > 1.5) & (t <= 2.5)].mean(axis=0))
            careful, took = still_read(t, pulses)
            careful = desk(careful)
            rows.append((np.linalg.norm((app - truth)[:2]), np.linalg.norm((fixed - truth)[:2]),
                         np.linalg.norm((careful - truth)[:2]), took))
            print(f"{file:18s} {str(run['target'][:2]):16s} move {run['done'] - run['started']:4.1f} s, read "
                  f"{run['read'] - run['done']:4.2f} s after; the cup last seen moving "
                  f"{'-' if stopped is None else f'{stopped:4.1f} s':>6s} after the move, "
                  f"{after_read:4.2f} mm after the app's read; across, from the last 3 s: app {rows[-1][0]:4.2f}, "
                  f"wait 2.5 s {rows[-1][1]:4.2f}, still 1 s {rows[-1][2]:4.2f} mm ({took:.2f} s); "
                  f"z app {app[2] - truth[2]:+.2f}")
    rows = np.array(rows)
    for name, column in (("app", 0), ("wait 2.5 s", 1), ("still 1 s", 2)):
        print(f"{name:10s}: across from the last 3 s' mean, median {np.median(rows[:, column]):.2f} mm, "
              f"worst {rows[:, column].max():.2f}")
    print(f"still 1 s took {np.median(rows[:, 3]):.2f} s typically, {rows[:, 3].max():.2f} at most (from the move)")


if __name__ == "__main__":
    main()
