"""The Tiles page: where the 25 mm letter tiles are on the desk, which the arm can pick up, and a word of them.

| module      | what it owns                                                                   |
| ----------- | ------------------------------------------------------------------------------ |
| `api`       | the page's requests: LOCATE TILES, a visit to a tile, PLACE, the snapshot      |
| `index.html`, `tiles.css`, `tiles.js` | the page                                             |
| `grid.js`   | the view from above: the work areas, and the tiles in them                     |
| `view.js`   | the same drawn over the camera's picture, live or the snapshot                 |

The tiles are found by `applib/tiles`, placed in the work areas by
`applib/areas.py`, and a word of them planned by `applib/word.py`: the final
app's own libraries, which know nothing of this page.
"""
