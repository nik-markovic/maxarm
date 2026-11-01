#!/usr/bin/env python3
"""After a precise move, does the arm still move once the app has read where it is?

    MAXARM_TRACE=1 .venv/bin/python impala/agenttools/arm-to-desk/settle.py [seconds] [more|all] [tag] 2> trace.txt

MOVES THE ARM: run only with the owner's say-so, the arm and the camera free (the app stopped). The arm is sent to
calibration points at z 42 as a grid click on Fine Tune sends it (`Motion.approach`, precise, over the cubes at 60);
then the app's read of it, as the worker reads after every move (`Motion.read`: 400 ms, then the median of 5 reads);
then the servos read again and again for `seconds` (6), and a settled read at the end. The camera's frames all the
while, in the mode the app uses, each timed as it comes, cut round the cup. The trace says every command and reply,
the precise rounds among them, timed from the trace's start (`trace_start` in the JSON).

Writes settle.json and settle-frames/ beside this file (settle-more.json, settle-more-frames/ with `more`; both
lists with `all`; `tag` added to the names); settle-check.py and settle-reads.py read them.
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, Space                                           # noqa: E402
from maxarm import protocol                                                # noqa: E402
from camera import store                                                   # noqa: E402
from camera.feed import Feed                                               # noqa: E402
from common.motion import OVER_CUBES_MM, Motion                            # noqa: E402

CONFIG = IMPALA / "config"
NOTCH_MM = 42.0
# The cubes' points as CALIBRATE's arm was sent over them, and two of Fine Tune's, desk mm; `more`: ten more of
# Fine Tune's.
POINTS = [(-140.0, -220.0), (110.0, -220.0), (127.0, -60.0), (-10.0, -150.0), (0.0, -230.0)]
MORE = [(-90.0, -130.0), (70.0, -130.0), (-110.0, -144.0), (40.0, -200.0), (-104.0, -110.0), (90.0, -180.0),
        (39.0, -250.0), (-79.0, -190.0), (121.0, -110.0), (141.0, -149.0)]
HALF_CROP = 260                                       # frame pixels round the cup, each way


class Recorder:
    """The camera's frames as they come, each with the time it came, cut round `window`."""

    def __init__(self, feed: Feed) -> None:
        self.feed = feed
        self.window = None
        self.frames = []                              # (time, point index, crop)
        self.full = {}                                # point index: the newest whole frame
        self.point = -1
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for _ in self.feed.frames():
            now = time.monotonic()
            with self.feed.condition:
                image = self.feed.image
            if image is not None and self.window is not None:
                x0, y0, x1, y1 = self.window
                self.frames.append((now, self.point, image[y0:y1, x0:x1].copy()))
                self.full[self.point] = image.copy()
            if self.stopping.is_set():
                return


def pixel(fit, point):
    seen = fit.desk_to_pixel @ (*point, 1.0)
    return seen[:2] / seen[2]


def to_desk(motion, pulses):
    arm = protocol._true_position(pulses)
    return list(motion.spaces.convert(arm, Space.ARM, Space.DESK)) if arm else None


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    which = sys.argv[2] if len(sys.argv) > 2 else ""
    points, name = {"more": (MORE, "settle-more"), "all": (POINTS + MORE, "settle-all")}.get(which, (POINTS, "settle"))
    name += f"-{sys.argv[3]}" if len(sys.argv) > 3 else ""
    frames_dir = HERE / f"{name}-frames"
    fit = store.load_camera(CONFIG)
    width, height = fit.frame_size
    frames_dir.mkdir(exist_ok=True)
    feed = Feed(config_dir=CONFIG)
    feed.start()
    recorder = Recorder(feed)
    recorder.thread.start()
    feed.still()                                      # the camera running before the arm moves
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, CONFIG)
    runs = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        for index, (x, y) in enumerate(points):
            target = (x, y, NOTCH_MM)
            u, v = pixel(fit, target)
            x0, y0 = int(np.clip(u - HALF_CROP, 0, width - 2 * HALF_CROP)), int(np.clip(v - HALF_CROP, 0, height - 2 * HALF_CROP))
            recorder.window, recorder.point = (x0, y0, x0 + 2 * HALF_CROP, y0 + 2 * HALF_CROP), index
            started = time.monotonic()
            result = motion.approach(Space.DESK, target, precise=True, hover=OVER_CUBES_MM)
            done = time.monotonic()
            motion.read()                                                  # as the worker reads after a move
            read = time.monotonic()
            app = list(motion.spaces.convert(motion.measured, Space.ARM, Space.DESK)) if motion.measured else None
            polls = []
            while time.monotonic() - read < seconds:
                pulses = arm._protocol.read_pulses()
                polls.append([time.monotonic(), *(pulses or (None, None, None))])
            settled = arm._protocol.read_settled_pulses()
            final = time.monotonic()
            runs.append({"target": target, "status": result.status.value, "detail": result.detail,
                         "started": started, "done": done, "read": read, "app_desk": app,
                         "polls": polls, "settled_pulses": list(settled) if settled else None,
                         "settled_desk": to_desk(motion, settled), "final": final,
                         "window": list(recorder.window), "pixel": [float(u), float(v)]})
            print(f"{target}: {result.status.value}; the app's read {np.round(app, 2).tolist() if app else None}, "
                  f"settled {np.round(to_desk(motion, settled), 2).tolist() if settled else None}", flush=True)
    finally:
        try:
            motion.reset()
        finally:
            arm.disconnect()
            recorder.stopping.set()
            time.sleep(1.0)
            feed.close()
    frames = []
    for number, (when, point, crop) in enumerate(recorder.frames):
        file = f"{point}-{number:04d}.png"
        cv2.imwrite(str(frames_dir / file), crop)
        frames.append([when, point, file])
    for point, image in recorder.full.items():
        cv2.imwrite(str(frames_dir / f"{point}-full.png"), image)
    (HERE / f"{name}.json").write_text(json.dumps({"trace_start": protocol._TRACE_START, "runs": runs,
                                                    "frames": frames, "frames_dir": frames_dir.name}, indent=1))
    print(f"{len(frames)} frames; wrote {HERE / (name + '.json')}")


if __name__ == "__main__":
    main()
