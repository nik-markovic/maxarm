"""What every calibration page shares: the web server, the arm's one thread, how it moves, its controls.

| module                | what it owns                                                               |
| --------------------- | -------------------------------------------------------------------------- |
| `server`              | the web server: the pages' files and the JSON API                          |
| `worker`              | the one thread that talks to the arm; commands, RESET                      |
| `motion`              | how the pages move the arm near the desk, in arm or desk space             |
| `arm_api`             | the arm's controls: move, jump, cup, suction, RESET, the limits            |
| `page.css`            | the look: dark, Avnet green                                                |
| `page.js`             | the API from the browser, the header with RESET, the state poll, dialogs   |
| `grid.js`, `view.css` | the view from above, the height beside it, the readout                     |
| `shares.js`           | desk space to the camera's picture and back, through the calibration       |
| `marks.js`            | the marks drawn over the camera's picture: lines, areas, the reach, the arm |
| `fullview.js`         | the camera's picture as large as the window allows, to click on            |
"""
