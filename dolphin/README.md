# dolphin — finding and reading Scrabble tiles

Spec: [`work/PILOT-dolphin.md`](../work/PILOT-dolphin.md).
State of play: [`work/STATUS-dolphin.md`](../work/STATUS-dolphin.md).

One call turns a camera frame into tiles: the letter, where the tile is in arm millimetres,
and which way it faces.

```python
tiles = scene.find_tiles(frame, DeskMapping.load(Path("config/calibration.json")), Reader())
for tile in tiles:
    print(tile.letter, tile.centre_mm, tile.baseline_deg)
```

```
./agenttools/find-tiles.py --live tiles-2    # snapshot, find, draw -> files/tiles-2-found.jpg
./agenttools/find-tiles.py files/tiles-1.png # the same on a saved frame
./agenttools/test-tiles.py                   # every saved scene whose letters are known
./agenttools/stress-tiles.py files/tiles-1.png  # the scene on synthetic desks, at lower resolutions
./agenttools/dump-crops.py files/tiles-1.png # what the reader sees, per tile and turn
../.venv/bin/python training/train.py        # re-train the reader (host only, ~15 min)
```

## How it works

| step | module             | what it does                                                        |
| ---- | ------------------ | ------------------------------------------------------------------- |
| 1    | `tiles/topdown.py` | The frame resampled onto the plane 4 mm up: tile tops from above    |
| 2    | `tiles/glyphs.py`  | Printed letters: ink on a quiet face. No brightness assumed         |
| 3    | `tiles/pose.py`    | The 17.5 x 20 mm box, fitted to the edges facing away from the camera |
| 4    | `tiles/crop.py`    | The top face square-on from the original frame, four quarter turns  |
| 5    | `tiles/read.py`    | A 289 KB int8 network: which letter, and which turn is upright      |
| 6    | `tiles/scene.py`   | The call; refits along the letter, drops non-letters and overlaps   |

The calibration is condor's, unchanged: `config/` is a copy. The full 3x4 camera is recovered from
the two planes it stores (`calibration/mapping.py`, `DeskMapping.camera()`), so nothing had to be
re-run.

| directory      | holds                                                             |
| -------------- | ----------------------------------------------------------------- |
| `tiles/`       | The tile finder                                                    |
| `calibration/` | condor's mapping, extended to any height; the snapshot code        |
| `training/`    | Synthetic tile renderer and the reader's training. Host only       |
| `config/`      | condor's calibration output, copied                                |
| `files/`       | Scenes; `reader.tflite`, and `reader.keras`, its float original     |
| `agenttools/`  | Tools for looking at what the finder does                          |

## Running it on the IMX95

Everything in `tiles/` needs OpenCV, NumPy and a TFLite interpreter, and uses nothing past
OpenCV 4. `read.py` takes `tflite_runtime` first, which is what the BSP ships. The model is int8
with uint8 input and output, which is the form `neutron-converter` takes; it has not been
through the converter yet.
