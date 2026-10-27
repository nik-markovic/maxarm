"""The demo's own libraries: what the final application is made of, besides the arm's library (`maxarm`).

| package   | what it owns                                                                 |
| --------- | ---------------------------------------------------------------------------- |
| `tiles`   | the 25 mm letter tiles in a camera frame: where each one is, and its turn    |
| `letters` | the letter on each tile, and which way up it lies                            |
| `areas`   | where tiles are picked up and set down, and which tiles the arm can take now |
| `word`    | a word set down: which tile for each letter, where it goes, how it turns     |
"""
