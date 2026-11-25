# kiwi — the tiles demo, calibration pages and /IOTCONNECT in one program, on the i.MX95

Spec: [`work/PILOT-kiwi.md`](../work/PILOT-kiwi.md).
State of play: [`work/STATUS-kiwi.md`](../work/STATUS-kiwi.md).
Started as a copy of `jackal/`, whose [`work/STATUS-jackal.md`](../work/STATUS-jackal.md) the code's section
references still point to.

One program, `src/main.py`, started by `run.sh`, in two parts on one arm, one camera and one language model.
**The calibration pages** (`src/calibration/`, their files in `web/`, port 8080) calibrate the desk and the camera and
try them: desk, camera, Fine Tune and Tiles pages, and the Chat page, the language model on the Ara240 finding words
of the tiles. **/IOTCONNECT** (`src/app/`) takes commands from the dashboard or the console, sends what it found as
telemetry, and streams the camera's picture with the tiles drawn over it over KVS WebRTC. `--no-webapp` leaves the
pages out, `--no-iotc` the cloud.

/IOTCONNECT starts only with the camera calibrated and the device's files in `config/` (below); until then the rest
runs and says what is missing at the start: calibrate, or put the files there, and start `run.sh` again. Both parts
drive the arm one command at a time, whichever asks, and RESET or `stop` cuts short either's. The pages' live view and
the stream are two branches of the camera's one pipeline: neither takes the camera from the other.

Both read the camera through GStreamer, which comes from the system, not pip: its Python bindings (`gi`) and their
`Gst` and `GstVideo` typelibs, and the base and good plugins. The i.MX95's image has them; on Ubuntu,
`python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-good`. The venv sees them
(`--system-site-packages`).

On the board, as root:

```
mkdir -p ~/kiwi && cd ~/kiwi
# TODO(owner): the package's download location
wget -O tiles.tgz https://downloads.iotconnect.io/partners/nxp/packages/tiles.tgz
tar zxf tiles.tgz
./install.sh            # once, and again after unpacking a newer package: the arm's USB driver, venv/, packages;
                        # with an Ara240 in the board, its model too, started at every boot (11 GB, many minutes)
./run.sh                # the pages on http://<board>:8080, /IOTCONNECT once it can, the console
./run.sh --no-iotc      # without the cloud
./run.sh --no-webapp    # without the pages
scripts/install-service.sh   # run.sh at every boot, as the service `tiles`; --remove takes it out
```

An Ara240 put in after `install.sh`: `connector/install.sh`, then `connector/install-service.sh`.

The package is made on a PC by `scripts/package.sh`. `config/` is not in it, so a newer one unpacked over the old
keeps the calibration and the device's identity.

On a PC, in a venv of your own if you like: `scripts/install.sh`, then `python src/main.py`.

## /IOTCONNECT

1. Import [`files/device-template.json`](files/device-template.json) as a device template.
2. Create a device with it. Either let /IOTCONNECT make the certificate -- copy the credentials zip's certificate and
   key to `config/device-cert.pem` and `config/device-pkey.pem` -- or paste in `config/device-cert.pem`, which
   `scripts/install.sh` made.
3. Download the device's `iotcDeviceConfig.json` into `config/`.

| command                     | does                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `detect`                    | park the arm, take the snapshot, find the tiles and read their letters              |
| `place` *LETTERS*           | detect, then set those letters down along the dropoff arc for the reader; detect    |
|                             | again                                                                               |
| `find-cube` [*red*]         | park the arm, find the red cube in the live view, bring the cup over it             |
| `reader-location` [*where*] | where the reader stands -- `back`, `front`, `left` or `right` -- or, alone, where   |
|                             | they stand now                                                                      |
| `agent` *SENTENCE*          | the sentence to the language model on the Ara240 -- `go`, `a 7-letter word`, `a     |
|                             | farm animal`, `elaborate` -- which finds a word of the tiles and has it set down    |
| `stop`                      | the model's request and whatever moves the arm cut short, the commands waiting      |
|                             | dropped; the cup lets go, lifts clear and goes home, as RESET on the pages          |

Each is answered by its acknowledgement, the answer the console prints -- `agent`'s cut short, the whole of it the
`answer` telemetry. The console takes the same commands as text: `place TEAM`, `find cube`, `reader left`,
`agent go`.

## The Ara240

Optional: without it the demo runs as before, and `agent` and the Chat page say where the model is -- not
installed, not running, waiting for the Ara240, loading, about 6 minutes after the connector starts, or ready. The
model, Qwen2.5-7B-Instruct, reads what a request wants; code finds the word among those the tiles spell, in the
word list `files/words-en.tsv` -- the longest, or for a clue the words WordNet ties to it (`files/words-wordnet.tsv`),
those nearest it in meaning by a small embedding model on the CPU that `scripts/install.sh` fetches, and the model's
own ideas, the model choosing one -- and the arm sets it down while the model explains it (`src/applib/agent/`,
one service for the Chat page and the `agent` command, one request at a time). `training/words/make-words.py` makes both files. The image
brings the Ara240's runtime and the wheels of NXP's eIQ AAF Connector, which holds the model on the Ara240 behind
an OpenAI-style endpoint, port 3000, on the board only. `connector/install.sh` installs the connector into its own
`connector/venv/`, patched (`apply-patch.sh`), and fetches the model into `/usr/share/llm/`, where the image's
`ara2-fetch-models` puts it. `connector/install-service.sh` starts it at every boot, as the systemd service
`iotc-aaf-connector` (`--remove` takes it out); without the service, `connector/run.sh` starts it in the
background, its log `connector/server.log`. With or without, `connector/run.sh` restarts it and `connector/run.sh --stop` stops it.
`connector/config/server_config.json` is its configuration.

| telemetry       | is                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------- |
| `tiles`         | the letters on the tiles detected, left to right from behind the arm, lower case: `?` not   |
|                 | read, `-` blank                                                                             |
| `err_no_reach`  | the letters of those out of the arm's reach; empty when none is                             |
| `err_too_close` | the letters of those too close to another to pick up; empty when none is                    |
| `answer`        | the language model's answer to `agent`, and then what became of the word it set down; sent  |
|                 | once each                                                                                   |
| `agent_status`  | where the language model is: `off`, `not installed`, `stopped`, `waiting`, `loading`,       |
|                 | `ready` or `failed`                                                                         |
| `version`, `sdk_version` | the app's and the SDK's                                                            |

The stream is the camera's live view at the calibration's frame size, made as large as H.264 level 3.1 holds, its
height in eights for the board's encoder (1248 x 720 from the C920's 2560 x 1472), with the Tiles page's overlay drawn
in -- the areas, the tiles, their letters, a word being set down -- the red cube after `find-cube`, and a bar: what
the arm is doing and for how long, or why it cannot be used, or the tiles last seen; the cloud when it is not online;
the language model on the Ara240, if there is one, and while it answers, the request and what it is doing. Each
command's answer shows over the bar, yellow, red when refused. It is a branch of the camera's GStreamer pipeline, run only while someone watches, at the
live view's 5 frames a second at most: H.264 from the board's hardware encoder where GStreamer has one, else x264 in
PyAV at its fastest, encoded once for every viewer (`src/applib/stream.py`).

## What the pages and /IOTCONNECT share

The calibration's choices carry over: the camera's mode and focus (`config/camera-view.json`, `camera.json`), and the
app's settings, the tiles' height, the grip and the reader (`config/appcfg.json`, only what differs from the
defaults, which are in `src/applib/settings.py`). So does the code: the Tiles page's LOCATE TILES and PLACE are the app's `detect` and
`place` (`src/applib/tilework.py`), Fine Tune's FIND RED CUBE its `find-cube` (`src/applib/redcube.py`), and what both
pages draw over the camera's picture is drawn in Python (`src/applib/overlay/`), the stream's overlay, laid over the
pages' live view as a layer.

| directory           | holds                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `src/main.py`       | the program: the composition root of the pages and /IOTCONNECT                           |
| `src/app/`          | what the commands do, command by command (`app.py`)                                      |
| `src/applib/`       | how, for both: the tiles, the letters, the camera, the arm's one thread (`worker.py`)    |
|                     | and moves, the overlay, the settings, the language model and its tools (`agent/`); and   |
|                     | the commands, telemetry, /IOTCONNECT, the stream and WebRTC                              |
| `src/maxarm/`       | the arm's library                                                                        |
| `src/calibration/`  | the calibration pages' requests; guide in its `README.md`                                |
| `web/`              | the calibration pages: HTML, scripts, styles                                             |
| `config/`           | the calibration, the settings and the device's identity; per rig, not in git             |
| `files/`            | the letter reader, `letters.tflite`, what it was built from, and the device template;    |
|                     | `ch341/<uname -r>/ch341.ko`, the arm's USB serial driver for that i.MX95 image           |
| `training/`         | `letters/`: the reader's recipe                                                          |
| `scripts/`          | `install.sh`, `install-ch341.sh`; `package.sh` makes `tiles.tgz`, `build-ch341.sh` the driver |
|                     | `install-service.sh` starts the demo at boot (`systemd/`)                                |
| `connector/`        | the Ara240's language model: NXP's connector, installed and run by its scripts           |
| `agenttools/`       | the fake board and camera, the overlay drawn on the owner's stills, checks; not shipped  |

The arm's library, its calls and limits: [`src/maxarm/README.md`](src/maxarm/README.md).
