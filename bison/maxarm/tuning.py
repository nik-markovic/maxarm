#!/usr/bin/env python3
"""The measured numbers, with their provenance. Internal -- not configuration.

Every value here came off this arm or out of its firmware, and every one has
the measurement that produced it written above it. They are constants rather
than fields on a config object on purpose: none of them is a decision the
caller wants to make, and a caller who changes one without reading the comment
above it will hurt the nozzle.

Five of them answer to the environment, because they are the five that a
different desk, a different arm or a bad afternoon could plausibly need:

| variable                 | what it changes                                 |
| ------------------------ | ----------------------------------------------- |
| `MAXARM_DEVICE`          | serial port, when it is not `/dev/ttyUSB0`      |
| `MAXARM_TRAVEL_SPEED`    | mm/s in free space; drop it for a cautious run  |
| `MAXARM_Z_FLOOR`         | default floor, when the desk is not this desk   |
| `MAXARM_DESK_SAG`        | mm of droop that counts as touching down        |
| `MAXARM_RETARGET_LIMIT`  | how far a target may be moved to make it legal  |

Everything else is hardcoded. If one of them turns out to need changing, the
fix is to measure it again and edit the number here, not to grow an API.
"""

import os

DEFAULT_DEVICE = os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0")
DEFAULT_BAUD = 115200

# Opening the port resets the ESP32, so every connect waits out the board's own
# main.py. Measured: 4.7 s once, then 15.1 s twice running -- the long ones are
# the board's BLE init timing out. Nothing sleeps a fixed interval; this is
# only the point at which we give up.
CONNECT_TIMEOUT_S = 25.0
COMMAND_TIMEOUT_S = 5.0


def _env(name: str, default: float) -> float:
    """A tuning value the environment may override. Bad values are ignored."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- how fast -------------------------------------------------------------

# The owner's yardstick: a second to cross from one safe half of the desk to
# the other. That is roughly 200 mm, so 200 mm/s, and a move's duration is
# simply its length at this speed. Short hops hit the floor below instead.
TRAVEL_SPEED_MM_S = _env("MAXARM_TRAVEL_SPEED", 200.0)
# Descents into the desk band are stepped, and a 2 mm step at travel speed
# would be over before the servos left their deadband.
DESCENT_SPEED_MM_S = 40.0
MIN_MOVE_MS = 150
MAX_MOVE_MS = 2500

# How often a flying hop is looked at. Reads only, no commands, so the cost is
# serial time -- and it is what lets stop() bite in the middle of a long move.
POLL_MS = 120

# The error keeps shrinking for about this long after a command lands, worth
# ~0.8 mm. Re-commanding the same coordinate instead of waiting was measured to
# gain nothing, so waiting is free. The readback jitters by ~1 mm, hence the
# median of several.
SETTLE_MS = 400
SETTLE_READS = 5


# --- how close is close enough --------------------------------------------

# The owner's figure, from placing 40 mm cubes: 2 mm is the difference between
# a cube that sits where it was put and one that has to be nudged by hand. The
# readback is quantised to 1 mm and jitters by about that much, so this is
# close to the floor of what can be *measured*, never mind commanded -- which
# is why the correction below gives up rather than chasing forever.
ARRIVAL_TOLERANCE_MM = 2.0
# When it has not arrived, nudge: aim past the target by the size of the miss.
# Four attempts rather than two, because the first one usually overshoots (the
# deadband means a mirrored command moves further than the error) and it takes
# another two to settle out of that.
CORRECTION_ATTEMPTS = 4
# How far past the target a nudge aims, as a multiple of the miss. Full mirror
# to begin with, halved after an overshoot, and never raised: a nudge that
# moved the arm nowhere means something is in the way, and the only safe answer
# to that is to stop. See `motion._next_gain()`.
CORRECTION_GAIN = 1.0
CORRECTION_GAIN_MIN = 0.25
# Do not chase a large miss with a nudge -- that is a blockage, not an aim
# error, and the result says so instead.
MAX_CORRECTION_MM = 12.0

# Command-to-readback distance that counts as tracking lost outright.
# Deliberately loose: every axis carries a standing offset, and a genuine limit
# is found by loss of progress, which needs no calibration.
TRACKING_TOLERANCE_MM = 12.0
# Readback is quantised to 1 mm and jitters by about that much, so movement
# under this is indistinguishable from the arm standing perfectly still.
NOISE_FLOOR_MM = 1.5
# Consecutive still polls before a flying hop is called off. Three at 120 ms is
# about a third of a second of nothing happening.
IDLE_POLLS_BEFORE_BLOCKED = 3


# --- margins, in millimetres ----------------------------------------------

# 4 mm of measured gravity sag plus ~1.3 mm of spread. Keeps planned targets
# inside the envelope the arm can *hold*, not merely solve for.
ENVELOPE_MARGIN_MM = 5.0
# The base fan boundary is knife-edge: single and double precision disagree
# within 1e-5 degrees of it. One degree is about 5 mm of arc at full reach.
FAN_MARGIN_DEG = 1.0
# Planning stays this far outside an exclusion zone. The arm lands within about
# 2.6 mm of its command, so aiming at the boundary itself lands inside it
# roughly half the time.
NO_FLY_MARGIN_MM = 8.0

# How far a target may be pulled to make it legal before it is refused instead.
# Past this the caller meant something else.
RETARGET_LIMIT_MM = _env("MAXARM_RETARGET_LIMIT", 20.0)


# --- the desk -------------------------------------------------------------

# The nozzle scrapes at about z=34 and barely touches at z=46; 48 leaves room
# for the arm's own error. This is only the default -- Limits(z=...) is the
# caller's to set, and that is the one limit they genuinely care about.
DEFAULT_Z_FLOOR = _env("MAXARM_Z_FLOOR", 48.0)

# Height above the floor to cross the desk at. An XY move commanded near the
# floor droops into it mid-path even when both endpoints clear it.
DESK_CLEARANCE_MM = 10.0
# Band above the floor in which a descent is stepped and watched rather than
# flown in one go.
DESK_GUARD_MM = 15.0
DESCENT_STEP_MM = 2.0
# Droop below the commanded z, net of the droop the first step of the same
# descent showed, that means the cup is on the surface. Real contact measures
# about 4 mm and the noise floor is about 2.4 mm, so it is re-read before it is
# believed rather than given a wider threshold.
DESK_SAG_MM = _env("MAXARM_DESK_SAG", 3.0)
# Droop in free space is gravity: real, smooth and expected. Only a large one
# means the arm is fighting something.
FREE_SAG_MM = 10.0


# --- routing --------------------------------------------------------------

# How finely a hop's swing is checked. Twelve samples over a 200 mm hop is a
# point every 17 mm, far below anything the arm could squeeze through.
SWING_SAMPLES = 12
# A hop whose swing leaves safe space is halved until each half is safe. Four
# halvings is a sixteenth of the original bow, which has always been enough;
# past that the route is refused rather than chopped into a crawl.
MAX_SPLITS = 4
# Arc spacing when detouring around a zone. Coarse on purpose: every chord is
# then checked against the zones by the swing test above and halved where it
# needs to be, so this only has to be fine enough to be a sensible starting
# shape. Fifteen degrees produced a dozen separate hops for one crossing.
ORBIT_STEP_DEG = 45.0

# How long to hold still on the target before trusting a pick. The cup needs a
# moment to seal. The release valve is held open for a second by a thread on
# the board regardless of what we do.
PICK_DWELL_S = 0.6


# --- the pump -------------------------------------------------------------

# `nozzle.on()` and `nozzle.off()` return the moment the H-bridge duty is
# written, and the pump is a physical thing: it takes this long to pull the cup
# down onto a piece, and as long again to let go of one. Returning from grip()
# before that means the next move starts while there is still no vacuum, which
# on a light piece is the difference between picking it up and pushing it away.
# The board reports nothing about the pump, so there is nothing to poll -- the
# only way to honour it is to wait.
#
# **Currently 2000 as an experiment, not as a measurement.** The owner sees a
# release let go cleanly and then something take hold of the cube again about a
# second later, and wants to know whether anything of ours is responsible for
# it. Nothing is sent to the board during this wait either way -- run with
# `MAXARM_TRACE=1` and read the timestamps -- so if the re-grip still happens
# at the same moment, it is the board or the physics. 500 was the working value
# and is what to go back to.
SUCTION_SETTLE_MS = 2000

# How long to stand still after a release before moving away.
#
# The board's vent valve is open for one second and then **closes**, which
# leaves the cup and its line a sealed dead volume with the pump off. Lifting
# at that moment is the worst thing that can be done with it: the cup works
# like a syringe, so pulling on a sealed volume deepens the vacuum rather than
# breaking it, and the piece comes up stuck to the cup. Standing still lets the
# residual bleed away through ordinary leakage instead.
#
# The owner found this by duplicating the placement row in his scene, which
# made releases reliable. That second row is not a second vent -- the board's
# `off()` is guarded by `nozzle_st`, already False by then, so it does nothing
# at all. What it bought was about three seconds of dwell.
#
# Three seconds is reliable and zero is not; nothing in between has been tried,
# so this sits inside that gap. If a piece still lifts away stuck, raise it.
RELEASE_DWELL_MS = 2000

# The cup servo reports nothing back, so the only way to know it has arrived is
# to wait -- and a cup still turning when the arm descends twists whatever it
# lands on.
#
# Measured on this arm: the cup swings the full 180 deg, -90 to +90, in about
# 1.2 s. That is the *servo's* rate, not the board's. `PWMServo.run()` ramps
# the pulse width over whatever duration it is given, on a 20 ms timer, so a
# duration shorter than this does not make the cup turn faster -- it makes the
# pulse arrive somewhere the horn has not reached yet. Hence deriving the
# duration from the size of the turn: the ramp and the horn then move together,
# and the wait is the same number.
NOZZLE_MS_PER_DEG = 1200.0 / 180.0
# The board clamps anything under 20 ms and steps on a 20 ms timer, so even a
# few degrees wants a handful of ticks to be a movement rather than a jump.
NOZZLE_MIN_TURN_MS = 200
# Margin for the horn to stop after the pulse stops changing. A guess, unlike
# the rate above -- if the cup is still creeping when the arm moves, raise it.
NOZZLE_SETTLE_MS = 150


def nozzle_turn_ms(degrees: float) -> int:
    """How long the cup needs to swing this far, at its measured rate."""
    return int(max(NOZZLE_MIN_TURN_MS, abs(degrees) * NOZZLE_MS_PER_DEG))
