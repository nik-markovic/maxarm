"""The red cube as the camera found it, drawn over the picture: on Fine Tune and on the stream.

    draw_cube(canvas, view, found)            # found: FoundCube.to_json(), and where the arm was sent and landed

- its edges, thin, in pink, fainter where its own faces hide them, and an X
  across its bottom under its centre: the cube's own edge shows beside them;
- where the arm was sent over it: a small dot on the cube's top;
- where the servos then said the cup was: the cup's rim, yellow, on the same
  plane -- the gap between the two is the arm's own miss;
- its desk x, y over it, clear of the cup's end, and the miss, X and Y.
"""

from __future__ import annotations

from applib.camera.model import BASE_DIAGONALS, CUBE_MM

from . import inks
from .canvas import Canvas, View

# The cup's skirt, its outside edge: 21 mm across (the owner's).
CUP_RADIUS_MM = 10.5
# The cube's x, y are written this far over the cup's end, which comes down to `down_mm`, and at least so many
# design px: the cup's end stays in sight.
LABEL_OVER_CUP_MM = 16.0
LABEL_OVER_CUP_PX = 18.0
SENT_DOT_PX = 2.5


def draw_cube(canvas: Canvas, view: View, found: dict, down_mm: float) -> None:
    """`found` as the red cube's finder has it (`FoundCube.to_json()`), with `sent` and `landed`, desk mm, once the
    arm has come down over it to `down_mm`."""
    corners = view.path(found["corners"])
    if any(corner is None for corner in corners):
        return
    pink = inks.BGR["cube"]
    for start, end in [*found["hidden"], *BASE_DIAGONALS]:
        canvas.lines([corners[start], corners[end]], pink, 1.0, 0.45)
    for start, end, is_outline in found["seen"]:
        canvas.lines([corners[start], corners[end]], pink, 1.2 if is_outline else 1.0, 0.9)
    landed = found.get("landed")
    if landed:
        canvas.lines(view.ring(landed[0], landed[1], CUP_RADIUS_MM, CUBE_MM), inks.BGR["yellow"], 1.5, 0.95)
    sent = found.get("sent") or [found["x"], found["y"]]
    aim = view.at(sent[0], sent[1], CUBE_MM)
    canvas.disc(aim, SENT_DOT_PX + 1.5, inks.BGR["shadow"], 1.0)
    canvas.disc(aim, SENT_DOT_PX, inks.BGR["white"], 1.0)
    missed = (f"  arm off X {landed[0] - sent[0]:+.1f} Y {landed[1] - sent[1]:+.1f}" if landed else "")
    end = view.at(found["x"], found["y"], down_mm)
    over = view.at(found["x"], found["y"], down_mm + LABEL_OVER_CUP_MM)
    if end and over:
        bottom = min(over[1], end[1] - LABEL_OVER_CUP_PX * canvas.scale)
        canvas.label(f"{found['x']:.1f}, {found['y']:.1f}{missed}", (end[0], bottom - 11 * canvas.scale), pink)
