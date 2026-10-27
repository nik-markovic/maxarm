"""The 25 mm letter tiles in a camera frame: where each one is on the desk, and how it is turned. See `find.py`.

| module    | what it owns                                                          |
| --------- | --------------------------------------------------------------------- |
| `find`    | `find_tiles()`: the steps, the checks, and `Tile`                     |
| `topdown` | the frame resampled onto the plane of the tiles' top faces, in mm     |
| `edges`   | where the view's colour changes, how sharply, and which way           |
| `votes`   | the places most voted for as a tile's centre, by the edges round them |
| `pose`    | a tile's square fitted to the edges, and how well it stands out       |
"""

from .find import TILE_HEIGHT_MM, TILE_MM, Tile, find_tiles

__all__ = ["TILE_HEIGHT_MM", "TILE_MM", "Tile", "find_tiles"]
