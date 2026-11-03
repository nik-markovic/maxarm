"""The Tiles page: where the 25 mm letter tiles are on the desk, which the arm can pick up, and a word of them.

| module      | what it owns                                                                   |
| ----------- | ------------------------------------------------------------------------------ |
| `api`       | the page's requests: LOCATE TILES, a visit to a tile, PLACE, the settings, the |
|             | tiles drawn over the picture                                                   |

What it does with the tiles is `applib/tilework.py`, the /IOTCONNECT app's too: the
tiles found by `applib/tiles`, read by `applib/letters`, placed in the work areas by
`applib/areas.py`, a word of them planned by `applib/word.py`, and drawn by
`applib/overlay`. The page is in `web/tiles/`.
"""
