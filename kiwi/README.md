# The tiles demo: how the code fits together

One program, `src/main.py`, started by `run.sh`, drives one arm, one camera and, if the board has an Ara240, one
language model. It has two parts:

- **The calibration pages** (`src/calibration/`, their files in `web/`, port 8080) set the camera up and try the
  result: Desk, to get to know the arm, then Camera, Fine Tune, Tiles and Chat.
- **The /IOTCONNECT app** (`src/app/`) takes commands from the dashboard or the console, sends what it found as
  telemetry, and streams the camera's picture with the tiles drawn over it over KVS WebRTC.

`--no-webapp` leaves the pages out, `--no-iotc` the cloud. `src/main.py`'s docstring has the layout, the one-way
imports between the parts, and every thread.

Things that are not obvious from the code layout:

- Both parts drive the arm through one thread (`applib/worker.py`), one command at a time, whichever asks; RESET on
  a page and the `stop` command cut short either's.
- Both see the camera through one GStreamer pipeline (`applib/camera/feed.py`): the pages' live view and the stream
  are its branches, so neither takes the camera from the other. GStreamer and its Python bindings come from the
  system, not pip; the venv is made with `--system-site-packages` to see them.
- They share the code too. The Tiles page's LOCATE TILES and PLACE are the app's `detect` and `place`
  (`applib/tilework.py`), Fine Tune's FIND RED CUBE is `find-cube` (`applib/redcube.py`), and the Chat page and the
  `agent` command are one service (`applib/agent/`). What both draw over the camera's picture is drawn in Python
  (`applib/overlay/`): the stream's overlay, laid over the pages' live view as a layer.
- They share `config/`: the calibration, the camera's mode and focus, and the settings in `appcfg.json`
  (`config/README.md`). The app starts its /IOTCONNECT part only once the camera is calibrated and the device's
  files are there; until then the rest runs and says what is missing.
- The stream is encoded once for every viewer, only while someone watches, by the board's hardware H.264 encoder
  where GStreamer has one, else x264 in PyAV (`applib/stream.py`).

## The language model

Optional: without an Ara240 the demo runs as before, and `agent` and the Chat page say where the model is. The model,
Qwen2.5-7B-Instruct, runs on the Ara240 behind NXP's eIQ AAF Connector, an OpenAI-style endpoint on the board only;
`connector/` installs, patches and runs it. The model only reads what a request wants. Code finds the words the tiles
spell, in `files/words-en.tsv`: the longest, or for a clue those WordNet ties to it (`files/words-wordnet.tsv`) and
those a small embedding model on the CPU finds nearest in meaning, and the model picks one. Both word files are made
by `training/words/make-words.py`.

## Commands and telemetry

| command                     | does                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `detect`                    | park the arm, take a snapshot, find the tiles and read their letters                |
| `place` *LETTERS*           | detect, set those letters down along the dropoff arc for the reader, detect again   |
| `find-cube`                 | park the arm, find the red cube in the live view, bring the cup over it             |
| `reader-location` [*where*] | where the reader stands: `back`, `front`, `left` or `right`; alone, where they are  |
| `agent` *SENTENCE*          | the sentence to the language model, which finds a word of the tiles and sets it down |
| `stop`                      | cut short the model and the arm, drop what is waiting; the cup lets go and goes home |

The console takes the same commands as text: `place TEAM`, `find cube`, `reader left`, `agent go`.

| telemetry                | is                                                                              |
| ------------------------ | ------------------------------------------------------------------------------- |
| `tiles`                  | the letters detected, left to right from behind the arm: `?` not read, `-` blank |
| `err_no_reach`           | the letters out of the arm's reach                                              |
| `err_too_close`          | the letters too close to another tile to pick up                                |
| `answer`                 | the model's answer to `agent`, then what became of the word                     |
| `agent_status`           | `off`, `not installed`, `stopped`, `waiting`, `loading`, `ready` or `failed`    |
| `version`, `sdk_version` | the app's and the SDK's                                                         |

## Where things are

| directory          | holds                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------- |
| `src/main.py`      | the program: the only place the parts are wired together                                |
| `src/app/`         | what each command does (`app.py`)                                                       |
| `src/applib/`      | how, for both parts: the tiles, the letters, the camera, the arm's thread and moves,    |
|                    | the overlay, the settings, the language model's tools, /IOTCONNECT, the stream          |
| `src/maxarm/`      | the arm's library: [`src/maxarm/README.md`](src/maxarm/README.md)                       |
| `src/calibration/` | the pages' requests: [`src/calibration/README.md`](src/calibration/README.md)           |
| `web/`             | the pages: HTML, scripts, styles, and each page's `help.md`                             |
| `config/`          | per rig, not in git: [`config/README.md`](config/README.md)                             |
| `files/`           | the letter reader (`letters.tflite`), the word lists, the device template, and          |
|                    | `ch341/<uname -r>/ch341.ko`, the arm's USB serial driver for that board image           |
| `training/`        | how the letter reader and the word lists are made                                       |
| `scripts/`         | install, the boot service, the USB driver; `package.sh` makes `tiles.tgz`, without      |
|                    | `config/`, so a newer package unpacked over the old keeps the calibration and identity  |
| `connector/`       | the Ara240's model: NXP's connector, installed and run by its scripts                   |
| `agenttools/`      | development checks and the fake board; not shipped                                      |
