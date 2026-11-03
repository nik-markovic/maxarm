"""The letters on the tiles: which capital each shows, and which way up it lies. See `read.py`.

| module   | what it owns                                                               |
| -------- | -------------------------------------------------------------------------- |
| `read`   | `read_letters()`: the steps, and `Letter`                                  |
| `face`   | a tile's face, square-on, from the frame, at any quarter turn               |
| `ink`    | whether a face has ink, and the dot that marks a letter's bottom right      |
| `reader` | the int8 network that says which capital a face shows, if that turn is up   |
"""

from .read import MIN_CONFIDENCE, Letter, read_letters

__all__ = ["MIN_CONFIDENCE", "Letter", "read_letters"]
