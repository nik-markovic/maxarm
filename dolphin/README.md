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

Tiles may touch at their corners but not along their edges. If a letter can be read that no tile
could be placed on, `find_tiles` raises `scene.UnplacedLetters`, carrying the tiles it did place
and where the others are: ask the user to spread the tiles, and look again.

```
./agenttools/find-tiles.py --live tiles-2    # snapshot, find, draw -> files/tiles-2-found.jpg
./agenttools/find-tiles.py training/tiles/baseline/tiles-1.png  # the same on a saved frame
./agenttools/stress-tiles.py training/tiles/baseline/tiles-1.png  # the scene on synthetic desks, at lower resolutions
./agenttools/dump-crops.py training/tiles/baseline/tiles-1.png  # what the reader sees, per tile and turn
./agenttools/measure-skew.py training/tiles/baseline/tiles-1.png  # how square the tiles come out: checks a calibration
./calibration/refine.py files/grid-1.png     # recalibrate from grid paper -> config/calibration.json
./training/tiles/make-reader.sh              # rebuild the reader from scratch (host only, ~15 min)
../.venv/bin/python training/tiles/check-reader.py  # the reader on the baseline scenes
```

**Rebuilding the reader.** `files/reader.tflite` is trained on synthetic tiles, and
`training/tiles/make-reader.sh` makes it again from nothing but that directory and the network: it
checks out the fonts (`fonts.txt`: 40 families of google/fonts at one pinned commit) into
`training/tiles/fonts/`, trains with every random choice seeded, exports the best epoch as int8,
and checks it on `training/tiles/baseline/` -- four real scenes with known letters, and the
calibration they were taken with, so no camera or calibration of your own is needed.
`requirements.txt` there pins the packages. On the same machine and versions the model comes
out byte-identical; `files/reader.json` records its hash, the fonts' hash, the versions and the
accuracy, to compare a rebuild against. Another CPU may round differently and give a model that
differs in its bytes -- then the test is what says whether it is as good.

`files/work-area.jpg` shows where the finder looks; tiles outside that outline are not searched.

## How it works

| step | module             | what it does                                                        |
| ---- | ------------------ | ------------------------------------------------------------------- |
| 1    | `tiles/topdown.py` | The frame resampled onto the plane 4 mm up: tile tops from above    |
| 2    | `tiles/glyphs.py`  | Printed letters: ink on a quiet face. No brightness assumed         |
| 3    | `tiles/pose.py`    | The 17.5 x 20 mm box, fitted to the edges facing away from the camera |
| 4    | `tiles/crop.py`    | The top face square-on from the original frame, four quarter turns  |
| 5    | `tiles/read.py`    | A 289 KB int8 network: which letter, and which turn is upright      |
| 6    | `tiles/scene.py`   | The call; refits along the letter, drops non-letters and overlaps, refuses a scene with letters it could not place |

The calibration starts from condor's and is refined with a sheet of 5 mm grid paper laid flat in
the work area (`calibration/refine.py`). condor's camera had non-square pixels, which skewed every
tile; the grid gives a square-pixel camera in true millimetres. What the arm itself gets wrong --
its x travels about 7% short -- is kept apart as `desk_to_arm`, fitted through condor's three
cubes, and only the tile positions handed to the arm go through it.

| directory      | holds                                                             |
| -------------- | ----------------------------------------------------------------- |
| `tiles/`       | The tile finder                                                    |
| `calibration/` | condor's mapping, extended; grid-paper calibration; snapshot code  |
| `training/`    | One directory per model; `tiles/` is the reader, with its baseline. Host only |
| `config/`      | The grid-refined calibration, and condor's original                |
| `files/`       | Scenes; `reader.tflite`, and `reader.keras`, its float original     |
| `agenttools/`  | Tools for looking at what the finder does. Transient: removed at release |

## Running it on the IMX95

Everything in `tiles/` needs OpenCV, NumPy and a TFLite interpreter, and uses nothing past
OpenCV 4. `read.py` takes `tflite_runtime` first, which is what the BSP ships. The model is int8
with uint8 input and output, which is the form `neutron-converter` takes; it has not been
through the converter yet.
