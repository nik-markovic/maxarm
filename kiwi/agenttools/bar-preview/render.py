"""The stream's bar drawn in a few states over config/camera-frame.png, for a look: no arm, no camera."""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from applib.camera.store import CameraFile  # noqa: E402
from applib.overlay import stream as overlay_stream  # noqa: E402
from applib.overlay.stream import StreamOverlay  # noqa: E402
from applib.stream import stream_size  # noqa: E402

overlay_stream.draw_tiles = lambda *args, **kwargs: None     # fake tiles: only the bar is looked at
camera_file = CameraFile(ROOT / "config")
fit, problem = camera_file.get()
size = stream_size(fit.frame_size)
frame = cv2.resize(cv2.imread(str(ROOT / "config" / "camera-frame.png")), size)
found = {"stamp": time.time_ns() - 190e9, "tiles": [{}] * 14}
ready = {"state": "ready", "detail": "", "since": None, "typical_s": 345}
loading = {"state": "loading", "detail": "", "since": time.time() - 100, "typical_s": 345}
word = {"status": "placing", "steps": [{"status": "placed", "letter": "G", "n": 3},
                                       {"status": "carrying", "letter": "O", "n": 7},
                                       {"status": "waiting", "letter": "A", "n": 1},
                                       {"status": "waiting", "letter": "T", "n": 9}]}
states = {
    "1-idle-no-agent": dict(scene={"found": found}),
    "2-idle-agent-ready-answer": dict(scene={"found": found}, agent={"model": ready, "thinking": None},
                                      answer=("A goat is a farm animal kept for its milk and meat; the arm has "
                                              "started setting GOAT down along the dropoff arc for the reader.", True)),
    "3-agent-thinking-arm-placing": dict(scene={"found": found, "word": word}, doing="Placing GOAT",
                                         agent={"model": ready, "thinking": {"text": "now a farm animal, please",
                                                "doing": "Thinking · pass 2", "seconds": 12}}),
    "4-arm-lost-cloud-off-loading": dict(scene={"found": None}, problem="it stopped answering: read timed out",
                                         cloud="cloud: reconnecting", agent={"model": loading, "thinking": None},
                                         answer=("The arm is not connected; it is tried again every 30 s.", False)),
}
for name, state in states.items():
    overlay = StreamOverlay(camera_file, size, lambda: state["scene"], lambda: None)
    if "agent" in state:
        overlay.watch_agent(lambda: state["agent"])
    overlay.set_doing(state.get("doing", ""))
    overlay.watch_arm(lambda: (state.get("doing", ""), state.get("problem", "")))
    overlay.set_cloud_status(state.get("cloud", "cloud: online"))
    if "answer" in state:
        overlay.show_answer(*state["answer"])
    picture = frame.copy()
    overlay.burn(picture)
    cv2.imwrite(str(Path(__file__).with_name(f"{name}.png")), picture)
    print(name)
