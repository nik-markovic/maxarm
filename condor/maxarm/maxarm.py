#!/usr/bin/env python3
"""`MaxArm` -- the one class a caller needs.

Everything the board does is in `protocol.py`; everything we decide to do about
it is here, in `route.py` and in `motion.py`. The split matters because the
board's behaviour is fixed and ours is not: a future transport (the binary
protocol, if the firmware is ever swapped) replaces the former without touching
the latter.

    from maxarm import MaxArm, Limits

    with MaxArm(limits=Limits(x=(0.0, None))) as arm:
        print(arm.get_position())
        arm.move_to(160.0, -160.0, 120.0)

`move_to()` works out for itself whether a move needs a detour, a lift, a
guarded descent or a nudge at the end. That decision is the library's job --
it is the whole reason there is a library -- so none of it is a knob. What the
caller configures is where the arm is and what the room forbids: `ArmConfig`
has four fields and `Limits` has three.

Reading state is cheap and lock-free: every step of every move updates a cached
snapshot, so a UI can poll `get_state()` at 30 Hz while a move runs without
touching the wire or blocking the motion.
"""

import math
import threading
from typing import Callable, Optional, Sequence, Tuple

from . import geometry, tuning
from .config import ArmConfig, Limits
from .geometry import Angles, JointFrame, Position
from .motion import Mover
from .protocol import (HOME_DURATION_MS, IS_TRACED, RELEASE_HOLD_MS, BoardProtocol,
                       ReplProtocol, trace_note)
from .route import Router
from .status import ArmState, MoveResult, MoveStatus, Step
from .transport import Transport, create_transport
from .zones import ExclusionZone, ZoneSet

StepCallback = Callable[[Step], None]


class NotConnectedError(RuntimeError):
    """A readback or a nozzle command was attempted before `connect()`."""


class MaxArm:
    """A MaxArm on the end of a serial port.

    Configure it in whichever form reads better at the call site -- they are
    the same thing:

        MaxArm()
        MaxArm(limits=Limits(z=(52.0, None)))
        MaxArm(ArmConfig(device="/dev/ttyUSB1", zones=[mug]))

    `transport` and `protocol` exist so the test suite can substitute a fake
    board; normal callers pass neither.
    """

    def __init__(self, config: Optional[ArmConfig] = None,
                 transport: Optional[Transport] = None,
                 protocol: Optional[BoardProtocol] = None,
                 **config_fields) -> None:
        if config is not None and config_fields:
            raise TypeError("pass an ArmConfig or its fields as keywords, not both")
        self.config = (config or ArmConfig(**config_fields)).resolved()
        self.limits: Limits = self.config.limits
        self.zones = ZoneSet(self.config.zones)
        # Pure geometry, so it works before -- and without -- a board. This is
        # what answers check_target() and is_reachable() on an unconnected arm.
        self.router = Router(self.limits, self.zones)

        self._transport = transport
        self._protocol = protocol
        self._mover: Optional[Mover] = None
        self._lock = threading.RLock()
        self._abort = threading.Event()
        self._state = ArmState(is_connected=False, position=None)

    # --- session ----------------------------------------------------------

    def connect(self) -> ArmState:
        """Open the link, check the board is fit to drive, adopt its pose.

        Be aware that this *moves the arm*: opening the port resets the ESP32,
        so the board re-runs its `main.py`, which homes the arm and rotates the
        nozzle. That is the firmware's doing and cannot be suppressed over this
        transport. Expect about 4 seconds, occasionally 15.

        Raises `BoardNotReadyError` rather than hand back an arm that will
        accept every command and move for none of them.
        """
        with self._lock:
            if self._protocol is None:
                self._transport = self._transport or create_transport(self.config)
                self._protocol = ReplProtocol(self._transport)
            self._protocol.connect()
            self._protocol.check_board()
            self._mover = Mover(self._protocol, self.router)
            # `_refresh_state()` reads the pose itself and hands it to the
            # mover, so asking `sync_position()` for it first was a second
            # round trip for the same three servos -- and a servo read retries
            # three times on the board before it gives up, so it is not free.
            # It is still the fallback when nothing answered at all.
            state = self._refresh_state()
            if state.position is None:
                self._mover.sync_position()
                state = self._snapshot(None)
            return state

    def disconnect(self) -> None:
        with self._lock:
            if self._protocol is not None:
                self._protocol.disconnect()
            self._mover = None
            self._state = ArmState(is_connected=False, position=self._state.position)

    def __enter__(self) -> "MaxArm":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._mover is not None

    # --- motion -----------------------------------------------------------

    def move_to(self, x: float, y: float, z: float,
                on_step: Optional[StepCallback] = None) -> MoveResult:
        """Move the nozzle to a position in millimetres.

        Works out its own route: a detour if the straight line crosses an
        exclusion zone, a lift and a guarded descent if it ends near the desk,
        and a nudge if it lands a few millimetres out.

        Never raises for a bad target. "I got within 3 mm" and "I refused to
        try" are both ordinary answers from a machine with a fixed envelope, so
        the reason comes back as a `MoveStatus`. The result is falsy unless the
        arm arrived where it was asked.
        """
        target = (float(x), float(y), float(z))
        if not self.is_connected:
            return MoveResult(MoveStatus.NOT_CONNECTED, target, target,
                              self._state.position or target, "call connect() first")
        with self._lock:
            self._abort.clear()
            result = self._mover.move(target, self._wrap_callback(on_step), self._abort)
            self._state = self._snapshot(result.status)
            return result

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      on_step: Optional[StepCallback] = None) -> MoveResult:
        """Jog from wherever the arm is now -- the operation a UI needs most."""
        base = self._state.position or geometry.HOME_COMMAND
        return self.move_to(base[0] + dx, base[1] + dy, base[2] + dz, on_step)

    def home(self) -> MoveResult:
        """Send the arm to the board's own origin, using the board's routine.

        Returns `STOPPED` rather than `REACHED` if `stop()` cut the wait short;
        the arm will still be travelling when it does.
        """
        if not self.is_connected:
            return MoveResult(MoveStatus.NOT_CONNECTED, geometry.HOME_COMMAND,
                              geometry.HOME_COMMAND, self._state.position or (0.0, 0.0, 0.0))
        with self._lock:
            # Like move_to(): an explicit new instruction re-arms the stop.
            self._abort.clear()
            # The board's routine cannot be called off once sent, but waiting
            # for it can be abandoned -- and the homing at the start of a run
            # is two and a half seconds of the arm swinging with, until now,
            # no way to interrupt it.
            self._protocol.go_home(is_waited=False)
            is_finished = self._wait(HOME_DURATION_MS / 1000.0 + 0.3)
            self._mover.sync_position()
            status = MoveStatus.REACHED if is_finished else MoveStatus.STOPPED
            self._state = self._snapshot(status)
            return MoveResult(status, geometry.HOME_COMMAND,
                              geometry.HOME_COMMAND, self._mover.position)

    def stop(self) -> None:
        """Ask whatever is running to stop. Safe to call from a signal handler.

        It takes effect at the next poll -- about a tenth of a second on a
        flying hop, one 2 mm step on a descent -- by retargeting the arm to the
        pose it currently occupies. There is no interrupt for the servos
        themselves over this transport, so this is a stop, not an e-stop.

        It also abandons every *wait* in this class: the pump settling, the cup
        servo swinging, the board's valve hold, the homing routine. Those are
        seconds each, and a stop that only bit during a move meant a Ctrl-C
        landing in one of them appeared to do nothing at all.
        """
        self._abort.set()

    def _wait(self, seconds: float) -> bool:
        """Sleep, unless and until `stop()` is called. True if it slept it out.

        None of these waits can be *shortened* -- they are a pump pulling down,
        a servo swinging, a valve held open by a thread on the board -- but
        they can all be abandoned, and that is the difference between a stop
        that bites in a tenth of a second and one that bites in two seconds.
        """
        return not self._abort.wait(max(0.0, seconds))

    # --- the nozzle -------------------------------------------------------

    def grip(self) -> None:
        """Suction on, and block while the pump pulls the cup down.

        The board's `nozzle.on()` returns as soon as the duty is written, which
        is not the same thing as a cup that is holding anything.
        """
        self._set_suction(True)

    def release(self, is_waited: bool = True) -> None:
        """Open the valve. The board holds it open for a second, on a thread.

        So a release is not instantaneous and two in quick succession overlap.
        `is_waited` blocks for the rest of that second, which is what a pick-
        and-place wants before it lifts away. The pump's own settling time is
        waited for either way -- see `_set_suction()`.

        It then stands still for `RELEASE_DWELL_MS` before returning, because
        the valve *closing* is what strands the vacuum: lifting away at that
        moment pulls on a sealed cup and deepens it. That dwell is what makes a
        release let go, and it is the whole of `is_waited`.

        This deliberately sends nothing but `nozzle.off()`. A second vent
        pulse was tried here, to bleed down the vacuum the board's one-second
        hold leaves behind, and **measured worse on the arm** -- see
        `protocol.VENT_CUP`, which is still the way to do it if the reason to
        comes back.
        """
        self._set_suction(False)
        if is_waited:
            self._wait(max(0.0, (RELEASE_HOLD_MS - tuning.SUCTION_SETTLE_MS) / 1000.0))
            self._wait(tuning.RELEASE_DWELL_MS / 1000.0)

    def set_nozzle_angle(self, angle_deg: float,
                         duration_ms: Optional[int] = None) -> None:
        """Rotate the cup about Z, -90 to +90 degrees, and wait it out.

        How long it takes depends on how far it has to go -- the cup swings
        180 deg in about 1.2 s and cannot be hurried -- so by default the turn
        is timed from the angle it is actually being asked to cover, measuring
        from where the cup already is. Pass `duration_ms` to say otherwise.

        The waiting is not optional for a reason. The board interpolates the
        cup servo on a 20 ms timer and reports nothing when it arrives, so a
        second `set_angle()` sent before the first finishes simply retargets
        it -- three in a row with no wait look, from outside, like the cup
        twitching and stopping where it started. And a cup still turning when
        the arm moves twists whatever it is touching.
        """
        angle = max(-90.0, min(90.0, float(angle_deg)))
        with self._lock:
            turn_ms = (int(duration_ms) if duration_ms is not None
                       else tuning.nozzle_turn_ms(angle - self._state.nozzle_angle_deg))
            self._require_protocol().set_nozzle_angle(angle, turn_ms)
            self._state = self._snapshot(self._state.last_status, nozzle_angle_deg=angle)
        self._wait((turn_ms + tuning.NOZZLE_SETTLE_MS) / 1000.0)

    def pick_at(self, x: float, y: float, z: float, approach_mm: float = 30.0,
                rotation_deg: Optional[float] = None) -> MoveResult:
        """Suction on, descend onto the target, dwell, lift clear.

        The suction starts before the descent so the cup is already pulling
        when it touches down; on a light piece that is the difference between
        picking it up and pushing it away.
        """
        if rotation_deg is not None:
            self.set_nozzle_angle(rotation_deg)
        result = self.move_to(x, y, z + approach_mm)
        if not result:
            return result
        self.grip()
        result = self.move_to(x, y, z)
        # Touching down early is how a pick is *supposed* to end, so unlike
        # every other early stop it is not a reason to abandon the sequence.
        if not result and result.status is not MoveStatus.DESK_CONTACT:
            return result
        self._wait(tuning.PICK_DWELL_S)
        return self.move_to(x, y, z + approach_mm)

    def place_at(self, x: float, y: float, z: float, approach_mm: float = 30.0,
                 rotation_deg: Optional[float] = None) -> MoveResult:
        """Descend onto the target, release, wait out the valve, lift clear."""
        if rotation_deg is not None:
            self.set_nozzle_angle(rotation_deg)
        result = self.move_to(x, y, z + approach_mm)
        if not result:
            return result
        result = self.move_to(x, y, z)
        if not result and result.status is not MoveStatus.DESK_CONTACT:
            return result
        self.release()
        return self.move_to(x, y, z + approach_mm)

    # --- reading, for UIs -------------------------------------------------

    def get_position(self, is_fresh: bool = True) -> Optional[Position]:
        """Where the nozzle is, by servo feedback. `is_fresh=False` is cached."""
        if not is_fresh:
            return self._state.position
        with self._lock:
            return self._require_protocol().read_settled_position(
                tuning.SETTLE_READS, tuning.SETTLE_MS / 1000.0)

    def get_joint_angles(self, is_fresh: bool = True) -> Optional[Angles]:
        """Measured servo-frame angles, from the servos themselves."""
        if not is_fresh:
            return self._state.joints
        with self._lock:
            return self._require_protocol().read_joint_angles()

    def get_joint_frame(self, is_fresh: bool = False) -> Optional[JointFrame]:
        """Joint positions in 3D -- everything a renderer needs to draw the arm.

        Prefers measured angles; falls back to solving the measured tip pose,
        which is the same chain within the readback's 1 mm quantisation.
        """
        angles = self.get_joint_angles(is_fresh) if self.is_connected else None
        angles = angles or self._state.joints
        if angles is None:
            if self._state.position is None:
                return None
            try:
                angles = geometry.inverse(self._state.position)
            except geometry.Unreachable:
                return None
        return geometry.joint_frame(angles)

    def get_state(self, is_fresh: bool = False) -> ArmState:
        """The last known snapshot, or a freshly measured one.

        The cached form costs nothing and is at most one step old, because
        every step of every move updates it. Ask for a fresh one only when the
        arm has been moved by something other than this object.
        """
        if not is_fresh or not self.is_connected:
            return self._state
        with self._lock:
            return self._refresh_state()

    # --- asking, without moving -------------------------------------------

    def check_target(self, position: Position) -> Tuple[Position, MoveStatus, str]:
        """What a `move_to()` here would actually aim at, and why. No motion.

        Pure arithmetic -- it works before `connect()`, which makes it the
        right way for a UI to grey out a button or for a game to pick a square.
        """
        return self.router.resolve(tuple(float(value) for value in position))

    def is_reachable(self, position: Position) -> bool:
        """True only if the arm can hold exactly this pose, with no retargeting."""
        return self.check_target(position)[1] is MoveStatus.REACHED

    def get_exclusion_zones(self) -> Sequence[ExclusionZone]:
        """Every zone in force, the non-negotiable base square first."""
        return self.zones.zones

    # --- internals --------------------------------------------------------

    def _set_suction(self, is_on: bool) -> None:
        """Switch the pump and wait for it, in both directions.

        The lock is dropped before the wait: nothing is on the wire during it,
        so a UI polling `get_state()` has no reason to block for half a second
        every time the cup is switched.

        Under `MAXARM_TRACE` it also reads the pose either side of the switch
        and reports how far the tip moved.

        That number is worth having, because **it is the only evidence this
        firmware can give that a seal formed.** The cup is a ribbed bellows
        with a couple of millimetres of stroke, and it contracts only when
        there is something sealed against it -- with no seal, air flows in as
        fast as the pump removes it and the pressure never drops. So a tip that
        settles a millimetre or two on `grip()` has sealed, and one that does
        not has not. Nothing else on this board reports on the cup at all.

        Those two reads happen *only* while tracing: a diagnostic that changes
        what the arm does is not one.
        """
        before = self._traced_position()
        with self._lock:
            self._require_protocol().set_suction(is_on)
            self._state = self._snapshot(self._state.last_status, is_suction_on=is_on)
        self._wait(tuning.SUCTION_SETTLE_MS / 1000.0)
        self._report_cup_pull(is_on, before)

    def _traced_position(self) -> Optional[Position]:
        if not IS_TRACED or not self.is_connected:
            return None
        with self._lock:
            return self._protocol.read_position()

    def _report_cup_pull(self, is_on: bool, before: Optional[Position]) -> None:
        after = self._traced_position()
        if before is None or after is None:
            return
        moved = tuple(after[i] - before[i] for i in range(3))
        trace_note(f"suction {'on ' if is_on else 'off'}: "
                   f"tip {_point(before)} -> {_point(after)}, "
                   f"dz {moved[2]:+.0f} mm, moved {math.dist(before, after):.0f} mm")

    def _wrap_callback(self, on_step: Optional[StepCallback]) -> StepCallback:
        def record(step: Step) -> None:
            self._state = self._snapshot(step.status)
            if on_step is not None:
                on_step(step)
        return record

    def _snapshot(self, status: Optional[MoveStatus], **overrides) -> ArmState:
        fields = dict(
            is_connected=self.is_connected,
            position=self._mover.position if self._mover else self._state.position,
            joints=self._state.joints,
            pulses=self._state.pulses,
            is_suction_on=self._state.is_suction_on,
            nozzle_angle_deg=self._state.nozzle_angle_deg,
            last_status=status,
        )
        fields.update(overrides)
        return ArmState(**fields)

    def _refresh_state(self) -> ArmState:
        protocol = self._require_protocol()
        position = protocol.read_position()
        if position is not None:
            self._mover.position = position
        pulses = protocol.read_servo_pulses()
        joints = geometry.pulse_to_deg(pulses) if pulses is not None else None
        # A failed read means the servo bus stayed quiet, not that the arm
        # vanished -- keep the last known values rather than blanking a UI.
        self._state = self._snapshot(self._state.last_status,
                                     position=position or self._state.position,
                                     pulses=pulses or self._state.pulses,
                                     joints=joints or self._state.joints)
        return self._state

    def _require_protocol(self) -> BoardProtocol:
        if self._protocol is None or self._mover is None:
            raise NotConnectedError("connect() first")
        return self._protocol


def _point(position: Position) -> str:
    """Compact pose, for trace lines. Readback is whole millimetres anyway."""
    return "({:.0f}, {:.0f}, {:.0f})".format(*position)
