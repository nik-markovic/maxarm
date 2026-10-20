"""How the calibration pages move the arm: carefully near the desk, in arm or desk coordinates.

Every move is a `MaxArm.move()`, so every limit the library knows is checked.
What this adds is how a person at a web page wants the arm to behave while
finding the desk with it:

- **direct**: straight there. Fast while both ends are above the hover height,
  slow below it; a vertical move through it goes fast down to it and slowly on.
- **jump**: over to another x, y -- slowly up to the hover height first if lower
  -- and stay at that height.
- **approach**: a jump, then down onto the point, the way `grab()` comes down
  onto a piece. Straight down when the point is underneath.
- **glide**: slowly and level, in short legs, to watch the cup's height.

`precise` ends a move with the library's correction (`MoveType.PRECISE`): the
servos stop a pulse or several short of where they are sent, and are sent on
until they are within half a millimetre.

Two spaces. ARM uses no calibration at all: the desk is what is being measured,
so no desk floor from an older calibration may keep the cup off it. DESK uses
the saved `desk.json`, to try it out.
"""

import math
import threading
from typing import List, Optional, Tuple

from maxarm import MaxArm, MoveResult, MoveStatus, MoveType, ProtocolError, Space, Spaces
from maxarm.arm import home_position
from maxarm.geometry import Position, bearing_degrees
from maxarm.path import CROSSING_MM

# Above this nothing on the desk can be hit, so moves go fast; below it they go
# slow. Arm z: the desk is at about 44 under this arm, as for any MaxArm
# standing on the desk it reaches. Desk z: a margin over the arm's +-3 mm.
HOVER_MM = {Space.ARM: 80.0, Space.DESK: 20.0}
# How far apart a glide's commands are. The kinematics model has the tip dip
# between the points it is sent to on moves in or out, with the square of the
# distance (STATUS-falcon 5.4, not measured): 10 mm legs, under 0.05 mm.
GLIDE_STEP_MM = 10.0
# How far a jump lowers its crossing height at a time, looking for one at which
# the far end is in reach: near the reset pose, little is.
TRAVEL_SEARCH_STEP_MM = 5.0

Leg = Tuple[Position, MoveType]


class Motion:
    """The arm, as the calibration pages drive it. Only the worker thread calls in here."""

    def __init__(self, arm: MaxArm, config_dir) -> None:
        self.arm = arm
        self.arm.use_calibration(None)
        self.config_dir = config_dir
        self.stopping = threading.Event()          # set by RESET: no further leg starts
        self.space = Space.ARM
        self.spaces = Spaces(config_dir)           # the saved calibration, for converting
        self.sent: Position = home_position()      # arm space: where the arm was last sent
        self.measured: Optional[Position] = None   # arm space: where its servos last said it was
        self.cup = 0.0                             # the cup servo's angle
        self.is_suction_on = False
        self.is_connected = False

    def connect(self) -> MoveResult:
        """Open the port -- the board resets and homes the arm -- and start from the reset pose."""
        if self.is_connected:
            self.is_connected = False
            self.arm.disconnect()
        self.arm.connect()
        self.is_connected = True
        self.cup, self.is_suction_on = 0.0, False
        self.use(Space.ARM)
        return self._track(self.arm.home())

    def use(self, space: Space) -> None:
        if space is self.space:
            return
        if space is Space.DESK:
            self.arm.use_calibration(self.config_dir)
            self.arm.use_coordinates(Space.DESK)
        else:
            self.arm.use_coordinates(Space.ARM)
            self.arm.use_calibration(None)
        self.space = space

    def reload(self) -> None:
        """`desk.json` has been rewritten: convert with it, and try it out, from now on."""
        self.spaces = Spaces(self.config_dir)
        if self.space is Space.DESK:
            self.arm.use_calibration(self.config_dir)

    def position(self, space: Space) -> Position:
        """Where the arm was last sent, in `space`."""
        return self.sent if space is Space.ARM else self.spaces.convert(self.sent, Space.ARM, space)

    # --- moves ------------------------------------------------------------

    def direct(self, space: Space, target: Position, precise: bool = False) -> MoveResult:
        self.use(space)
        here, hover = self.position(space), HOVER_MM[space]
        if min(here[2], target[2]) >= hover:
            return self._go([(target, MoveType.FAST)], precise)
        if max(here[2], target[2]) > hover and _sideways(here, target) < CROSSING_MM:
            level = (target[0], target[1], hover)
            if target[2] < here[2]:
                return self._go([(level, MoveType.FAST), (target, MoveType.SLOW)], precise)
            return self._go([(level, MoveType.SLOW), (target, MoveType.FAST)], precise)
        return self._go([(target, MoveType.SLOW)], precise)

    def jump(self, space: Space, x: float, y: float) -> MoveResult:
        self.use(space)
        here = self.position(space)
        travel = self._travel_height(here, x, y)
        legs: List[Leg] = [((x, y, travel), MoveType.FAST)]
        if here[2] < travel:
            legs.insert(0, ((here[0], here[1], travel), MoveType.SLOW))
        return self._go(legs)

    def approach(self, space: Space, target: Position, precise: bool = False) -> MoveResult:
        self.use(space)
        if _sideways(self.position(space), target) >= CROSSING_MM:
            result = self.jump(space, target[0], target[1])
            if not result:
                return result
        return self.direct(space, target, precise)

    def glide(self, space: Space, target: Position) -> Optional[MoveResult]:
        """Slowly and level to `target`, corrected there; None, with nothing sent, if a straight line cannot get there."""
        self.use(space)
        here = self.position(space)
        steps = max(1, math.ceil(math.dist(here, target) / GLIDE_STEP_MM))
        legs = [(tuple(a + (b - a) * step / steps for a, b in zip(here, target)), MoveType.SLOW)
                for step in range(1, steps + 1)]
        if not all(self.arm.can_reach(*point) for point, _ in legs):
            return None
        return self._go(legs, precise=True)

    def turn(self, servo: float) -> MoveResult:
        """The cup's servo to `servo` degrees, -90 to 90, where the arm is."""
        self.use(Space.ARM)
        facing = servo + bearing_degrees(self.sent[0], self.sent[1])
        result = self._track(self.arm.move(*self.sent, angle=facing))
        if result:
            self.cup = servo
        return result

    def suction(self, is_on: bool) -> MoveResult:
        """Pump on or off, waited out: the release takes longer, so nothing lifts with the cup."""
        result = self.arm.set_suction(is_on)
        self.is_suction_on = is_on
        return result

    def reset(self) -> MoveResult:
        """Let go, lift clear of the desk, and go home with the cup centred."""
        self.use(Space.ARM)
        if self.is_suction_on:
            self.suction(False)
        x, y, z = self.sent
        if z < HOVER_MM[Space.ARM]:
            self._go([((x, y, HOVER_MM[Space.ARM]), MoveType.SLOW)])   # refused or not, home next
        result = self._track(self.arm.move(*home_position(), angle=0.0))
        if result:
            self.cup = 0.0
        return result

    def hold(self, seconds: float) -> bool:
        """Wait, unless RESET. True if it waited it out."""
        return not self.stopping.wait(seconds)

    def read(self) -> None:
        """Where the servos say the arm is, kept in arm space."""
        try:
            here = self.arm.get_position()
        except ProtocolError:
            self.measured = None
            return
        self.measured = here if self.space is Space.ARM else self.spaces.convert(
            here, self.space, Space.ARM)

    # --- bookkeeping ------------------------------------------------------

    def _go(self, legs: List[Leg], precise: bool = False) -> MoveResult:
        """The legs in turn; `precise` then corrects at the last one, which the arm is already sent to."""
        if precise:
            legs = legs + [(legs[-1][0], MoveType.PRECISE)]
        result = MoveResult(MoveStatus.DONE, self.position(self.space))
        for point, move_type in legs:
            if self.stopping.is_set():
                return MoveResult(MoveStatus.STOPPED, self.position(self.space), "RESET")
            result = self._track(self.arm.move(*point, move_type))
            if not result:
                return result
        return result

    def _track(self, result: MoveResult) -> MoveResult:
        self.sent = (result.position if self.space is Space.ARM
                     else self.spaces.convert(result.position, self.space, Space.ARM))
        return result

    def _travel_height(self, here: Position, x: float, y: float) -> float:
        """The hover height, or the arm's own if higher -- lowered until (x, y) is in reach there."""
        hover = HOVER_MM[self.space]
        height = max(here[2], hover)
        while height > hover and not self.arm.can_reach(x, y, height):
            height = max(height - TRAVEL_SEARCH_STEP_MM, hover)
        return height


def _sideways(a: Position, b: Position) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
