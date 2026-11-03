"""The camera: which one, its live view, the calibration that ties its pixels to the desk, and the cubes in a frame.
The camera page fits the calibration (`calibration/camera/`); both apps use it.

| module     | what it owns                                                             |
| ---------- | ------------------------------------------------------------------------ |
| `devices`  | which camera: the owner's if it is plugged in, never a virtual one       |
| `feed`     | the live view at full size, and the stills and snapshots taken of it     |
| `store`    | the camera calibration's files in `config/`, `camera.json` among them    |
| `fit`      | the camera that sees the cubes where the arm says they stand             |
| `model`    | the pinhole camera, the 40 mm cube, and the least-squares solver         |
| `cubes`    | the cubes in a frame: coloured, cube-shaped blobs, and their names       |
| `edges`    | where a cube drawn through the camera meets the frame's edges            |
| `sighting` | a cube seen in a frame, measured through a camera: where, and how tall   |
| `locate`   | the red cube, found by the calibrated camera                             |
"""
