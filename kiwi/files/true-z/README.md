# The desk found by the cup's seal

The measurements behind `src/maxarm/raw.py`'s `DESK_WAVE_MM`: at points all over the desk, the cup was lowered in
small steps until the suction sealed, and the highest reading that sealed is where the desk is.

| file                              | what                                                                     |
| --------------------------------- | ------------------------------------------------------------------------ |
| `2026-11-05-run1-first-13.jsonl`  | the first 13 points, an earlier search in millimetres; not in the fit    |
| `2026-11-05-run2-letter.jsonl`    | 48 points in front of the arm, rings 115 to 265 mm out                   |
| `2026-11-05-run3-affirm.jsonl`    | the same rings out to 60 degrees, and the 175 mm ring from -105 to +105  |
| `2026-11-05-points.csv`           | one row per point of the three runs                                      |

Each run is one JSON object per line, written as the arm went:

- `"kind": "try"`: one landing. `x`, `y`, `sent`: where the cup was sent, true by the formula then in use;
  `sent_raw` the same as the firmware was sent it; `read`, `read_raw`: where the servos said the cup came down,
  before the pump; `library` the library's seal verdict, `state` the run's own (a seal kept only if it pulled the
  arm down 0.1 mm or more); `dip` the pump's dip in the supply pin's counts; `pulled` how far the seal pulled the
  arm down, mm; `aim` the servo pulses aimed at, if any; `time` Unix seconds.
- `"kind": "point"`: one point's result. `top` the highest reading that sealed and `top_raw` its firmware
  position, `bottom` the lowest that did not; or `why` there was none.

Run 1's tries carry the library's verdict in `state` and have no `library`. The raw positions do not depend on any
formula, so they can be worked out again by a newer one.

In the CSV, `seal_raw_*` is the highest seal as the firmware read it, `seal_true_z_then` and `miss_true_z_then` by
the formula then in use, `seal_true_z_now` by `raw.py` now, and `seal_over_desk_now` its height over the desk now
(0.5 mm is where it should be).
