#!/usr/bin/env python3
"""config/desk.json from the seal runs: the plane through every point's highest seal, MARGIN_MM deeper. No arm.

    cd jackal
    ../.venv/bin/python agenttools/true-z/desk-from-seal.py [run.jsonl ...]   # by default fit-desk.py's runs

The heights are the library's true z as raw.py has it now, so run this after raw.py's DESK_WAVE_MM is set from the
same runs. What is left after raw.py's wave is the arm's mounting tilt, which is the desk plane's to take. The
desk.json it replaces is kept as desk-previous.json, as the desk page's SAVE does.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parents[1] / "src" / "calibration")]

from maxarm import raw                     # noqa: E402
from maxarm.spaces import CONFIG_DIR       # noqa: E402
from desk import plane, store              # noqa: E402

from seal_edge import rebuild              # noqa: E402

MARGIN_MM = 0.5
# A point this far off the plane through the others is not the desk's: it is left out, and named.
OFF_MM = 1.0


def main() -> None:
    paths = ([Path(arg) for arg in sys.argv[1:]]
             or [path for path in HERE.glob("desk-seal*.jsonl") if not path.stem.endswith("-1")])
    found = []
    for path in paths:
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        found += [entry for entry in rebuild(entries) if entry["found"] and not entry["point"].startswith("ref")]
    touches = [raw.true_position(tuple(entry["top_raw"])) for entry in found]
    touches = [(x, y, z - MARGIN_MM) for x, y, z in touches]
    fitted = plane.fit(touches)
    kept = [touch for touch in touches if abs(fitted.height_of(touch)) <= OFF_MM]
    left_out = [entry["point"] for entry, touch in zip(found, touches) if abs(fitted.height_of(touch)) > OFF_MM]
    problem = plane.problem_with(kept)
    if problem:
        raise SystemExit(problem)
    fitted = plane.fit(kept)
    deviations = plane.deviations(kept, fitted)
    store.save_desk(CONFIG_DIR, fitted, [store.Point(index + 1, *touch, sent=None) for index, touch in enumerate(kept)])
    print(f"{CONFIG_DIR / store.DESK_FILE}: a plane through {len(kept)} points, {MARGIN_MM} mm under the highest seal"
          f" at each; tilted {fitted.tilt_deg:.3f} deg, at the base axis {fitted.offset:.2f}; the points off it "
          f"{plane.rms(deviations):.2f} typically, {max(abs(value) for value in deviations):.2f} at worst"
          + (f"; left out, over {OFF_MM} mm off: {', '.join(left_out)}" if left_out else ""))


if __name__ == "__main__":
    main()
