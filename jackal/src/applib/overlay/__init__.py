"""The overlay: what is drawn over the camera's picture, in Python, for the calibration app's pages and the
/IOTCONNECT app's stream alike.

| module   | what it owns                                                                        |
| -------- | ----------------------------------------------------------------------------------- |
| `canvas` | the transparent layer marks are drawn on (`Canvas`), and desk space onto it (`View`) |
| `inks`   | the colours, which the pages are given too                                          |
| `tiles`  | the tiles found, their letters, the work areas, a word set down                     |
| `cube`   | the red cube found, and where the arm came down over it                             |
| `stream` | the app's own: the tiles, the cube and a status bar over every frame of the stream  |

    layer = picture_layer(fit, 1200, 1.5, lambda canvas, view: draw_tiles(canvas, view, work.scene()))
    layer.png()                   # a page lays it over its live view
    layer.burn(frame)             # or into a frame of its size, for a picture; the stream has GStreamer lay it over

A page asks for its layer at the size it shows the picture, in its own pixels, and
its pixel ratio as `scale`: lines and letters come out the size they always were.
"""

from typing import Callable, Tuple

from applib.camera.fit import CameraFit

from .canvas import Canvas, Layer, View
from .cube import draw_cube
from .tiles import draw_tiles

__all__ = ["Canvas", "Layer", "View", "draw_cube", "draw_tiles", "picture_layer", "picture_size"]

# The widest a page's layer is drawn: the camera's own frame is no wider.
MAX_WIDTH_PX = 4096


def picture_size(fit: CameraFit, width: int) -> Tuple[int, int]:
    """The calibration's picture `width` pixels wide, as many high as its shape says."""
    frame_width, frame_height = fit.frame_size
    width = max(16, min(int(width), MAX_WIDTH_PX))
    return width, max(1, round(width * frame_height / frame_width))


def picture_layer(fit: CameraFit, width: int, scale: float, draw: Callable[[Canvas, View], None]) -> Layer:
    """`draw(canvas, view)` on a layer the shape of the calibration's picture, `width` pixels wide."""
    size = picture_size(fit, width)
    canvas = Canvas(size, scale)
    draw(canvas, View(fit.desk_to_pixel, fit.frame_size, size, fit.desk_to_true))
    return canvas.layer()
