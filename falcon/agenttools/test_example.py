#!/usr/bin/env python3
"""`main.py`, run as written, against the fake board, on a real scene.

Only pyserial is replaced (`fake_arm.fake_pyserial`), so `MaxArm()` in the
example opens a "port" like any other and everything under it runs for real.
The frame is `tiles-1.png`, and the calibration the one it was taken with
(`harness.BASELINE`): `config/` is per rig and may hold none.
"""

import contextlib
import functools
import importlib.util
import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maxarm import MaxArm, Space, Spaces                       # noqa: E402
from maxarm.geometry import HOME                               # noqa: E402
from fake_arm import fake_pyserial, instant_waits              # noqa: E402
from harness import BASELINE, check                            # noqa: E402

FRAME = ROOT / "training" / "tiles" / "baseline" / "tiles-1.png"


def test_example_runs() -> bool:
    print("main.py on tiles-1.png, against the fake board:")
    example = _load(ROOT / "main.py")
    example.CAMERA = BASELINE / "camera.json"
    example.MaxArm = functools.partial(MaxArm, config_dir=BASELINE)
    output = io.StringIO()
    with fake_pyserial() as boards, instant_waits(), contextlib.redirect_stdout(output):
        code = example.main(["main.py", str(FRAME)])
    lines = output.getvalue().splitlines()
    board = boards[0]
    grips = [entry for entry in board.log if entry == ("suction", True)]
    releases = [index for index, entry in enumerate(board.log) if entry == ("suction", False)]
    # Where each release happened: the move just before it is the slow descent.
    set_down = [board.log[index - 1][1] for index in releases]
    spaces = Spaces(BASELINE)
    slots = [spaces.convert((px, py, example.DROP_HEIGHT_MM), Space.CAMERA, Space.ARM)
             for px, py in example.SLOTS]
    at_slots = [min(range(len(slots)), key=lambda i: math.dist(point, slots[i])) for point in set_down]
    is_each_at_its_slot = all(math.dist(point, slots[i]) < 0.02 and i == n
                              for n, (point, i) in enumerate(zip(set_down, at_slots)))
    upright = sum("upright" in line for line in lines)
    return all([
        check("it runs to the end", code == 0, f"exit {code}"),
        check("the tiles out of reach are left alone, the rest fill the row",
              len(lines) == len(example.SLOTS) and all("done" in line for line in lines),
              "\n        ".join([""] + lines)),
        check("a grip and a release per slot", len(grips) == len(releases) == len(example.SLOTS),
              f"{len(grips)} / {len(releases)}"),
        check("each tile set down at its own slot, in order", is_each_at_its_slot),
        check("upright where the cup could turn it, as found where not",
              all(("upright" in line) != ("as found" in line) for line in lines),
              f"{upright} upright, {len(lines) - upright} as found"),
        check("and the arm goes home at the end", board.arm.position == HOME),
    ])


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("example_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


TESTS = [test_example_runs]
