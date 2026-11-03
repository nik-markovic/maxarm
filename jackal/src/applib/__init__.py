"""The demo's own libraries, the arm's (`maxarm`) aside: what the /IOTCONNECT app and the calibration app are made
of, and what they share. Nothing here knows of either app's pages, nor of the cloud but `iotc.py`.

| module      | what it owns                                                                         |
| ----------- | ------------------------------------------------------------------------------------ |
| `tiles`     | the 25 mm letter tiles in a camera frame: where each one is, and its turn            |
| `letters`   | the letter on each tile, and which way up it lies                                    |
| `areas`     | where tiles are picked up and set down, which the arm can take, where the reader is  |
| `word`      | a word set down: which tile for each letter, where it goes, how it turns             |
| `tilework`  | the tiles found, read and placed, and a word of them set down: the Tiles page's and  |
|             | the app's `detect` and `place`                                                       |
| `redcube`   | the red cube found and the cup brought over it: Fine Tune's and the app's            |
| `camera`    | which camera, the live view, the camera calibration and its files, the cubes found   |
| `motion`    | how the arm moves near the desk: carefully, in arm or desk space                     |
| `overlay`   | what is drawn over the camera's picture: on the pages, and on the stream             |
| `settings`  | the app's settings, `config/appcfg.json`: the grip height, the reader                |
| `stream`    | the camera's picture with the overlay in it, frame after frame, for WebRTC           |
| `commands`  | commands from any source, run off the other threads, each with its answer            |
| `console`   | the console as a source of commands                                                  |
| `telemetry` | what the device tells the cloud, kept until it is sent                               |
| `iotc`      | /IOTCONNECT: telemetry out, commands in, the stream's channel                        |
| `webrtc`    | KVS WebRTC: the stream to every viewer                                               |
"""
