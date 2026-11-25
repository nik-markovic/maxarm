"""What every calibration page shares: the web server and the arm's controls.

| module      | what it owns                                                               |
| ----------- | -------------------------------------------------------------------------- |
| `server`    | the web server: the pages' files, from `web/`, and the JSON API            |
| `arm_api`   | the arm's controls: move, jump, cup, suction, RESET, the limits            |

The one thread that talks to the arm is `applib/worker.py`, and how the arm moves near the desk
`applib/motion.py`: the /IOTCONNECT app's too.
The pages' shared look and scripts are in `web/common/`.
"""
