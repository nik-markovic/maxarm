"""What every calibration page shares: the web server, the arm's one thread, its controls.

| module      | what it owns                                                               |
| ----------- | -------------------------------------------------------------------------- |
| `server`    | the web server: the pages' files, from `web/`, and the JSON API            |
| `worker`    | the one thread that talks to the arm; commands, RESET                      |
| `arm_api`   | the arm's controls: move, jump, cup, suction, RESET, the limits            |

How the arm moves near the desk is `applib/motion.py`, the /IOTCONNECT app's too.
The pages' shared look and scripts are in `web/common/`.
"""
