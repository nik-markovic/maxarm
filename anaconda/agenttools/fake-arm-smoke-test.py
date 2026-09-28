#!/usr/bin/env python3
"""Exercise the reach tooling against a fake arm -- no hardware, no motion.

The fake board is driven by pulses, like the real one: an IK that throws
outside a reach shell, `set_servo_in_range` clamping servo 2 above 700 and
servo 3 below 470 while still reporting success, and a gravity sag near the
desk that no model can see. Those are the failure modes the tooling exists to
catch. The geometry is invented -- only a physical sweep produces real numbers.

Everything runs through the real MaxArmLink over a fake serial port, so the raw
REPL framing is covered too.
"""

import contextlib
import csv
import importlib.util
import io
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

DATACOLLECTION = Path(__file__).resolve().parent.parent / "datacollection"
sys.path.insert(0, str(DATACOLLECTION))

import maxarm_link  # noqa: E402
from envelope_model import EnvelopeModel  # noqa: E402
from reach_probe import (NO_FLY_HALF, NO_FLY_PLANNING_HALF, Outcome,  # noqa: E402
                         ReachProbe, is_in_no_fly, is_segment_clear)

SHOULDER_HEIGHT = 84.0
REACH_MIN = 60.0
REACH_MAX = 230.0      # beyond this the IK throws
SAG_Z = 55.0           # below this the nozzle starts hitting the desk
SAG_REACH = 190.0
HOME = (0.0, -163.0, 212.0)
TRACKING_GAIN = 0.8    # fraction of remaining distance covered per move


class FakeBoardArm:
    """What `arm` looks like in the board's namespace, pulses and all."""

    def __init__(self) -> None:
        self.position = HOME   # what espmax was last ASKED for
        self.actual = HOME     # where the servos really are
        self.move_count = 0

    def _pulses(self, position) -> Optional[Tuple[float, float, float]]:
        """None where the real __espmax.inverse() would throw."""
        x, y, z = position
        reach = math.sqrt(x ** 2 + y ** 2 + (z - SHOULDER_HEIGHT) ** 2)
        if not REACH_MIN <= reach <= REACH_MAX:
            return None
        stretch = reach - REACH_MIN
        return (500.0, 400.0 + stretch * 1.8, 900.0 - stretch * 2.6)

    def position_to_pulses(self, position):
        pulses = self._pulses(position)
        if pulses is None:
            # The real __espmax.inverse() prints the required reach and the link
            # lengths to stdout before failing, polluting the reply.
            print(f"{math.dist(position, (0, 0, SHOULDER_HEIGHT)):.4f}")
            print("128.4 138.0")
            raise ValueError("no solution")
        return pulses

    def verify_position(self, x, y, z) -> bool:
        solved = self._pulses((x, y, z)) is not None
        if not solved:
            print(f"{math.dist((x, y, z), (0, 0, SHOULDER_HEIGHT)):.4f}")
            print("128.4 138.0")
        return solved

    def set_position(self, position, duration_ms):
        x, y, z = position
        if math.hypot(x, y) < 50.0:
            return None
        pulses = self._pulses((-x, y, z))
        if pulses is None:
            return False

        self.move_count += 1
        if pulses[1] > 700.0 or pulses[2] < 470.0:
            # Clamped: the servo is pinned, so further commands change nothing
            # while set_position still reports success. This is the stall an
            # edge actually looks like.
            return True

        self.position = (x, y, z)
        target = (x, y, z - 4.0) if z < SAG_Z and math.hypot(x, y) > SAG_REACH else (x, y, z)
        # First-order lag on every axis: the arm closes most, not all, of the
        # remaining distance each move. That is the drift the tooling must
        # tolerate without mistaking it for a limit.
        self.actual = tuple(self.actual[i] + TRACKING_GAIN * (target[i] - self.actual[i])
                            for i in range(3))
        return True

    def read_position(self):
        return tuple(int(round(v)) for v in self.actual)

    def go_home(self, duration_ms=2000) -> None:
        self.position = self.actual = HOME


class FakeBoardTime:
    def __init__(self) -> None:
        self.slept_ms: List[int] = []

    def sleep_ms(self, ms: int) -> None:
        self.slept_ms.append(ms)


class FakeSerial:
    """A MicroPython v1.12 raw REPL on a wire. Executes what it is sent."""

    def __init__(self, *args, **kwargs) -> None:
        self.port = self.baudrate = self.timeout = None
        self.dtr = self.rts = None
        self.outbox = b""
        self.buffered_code = b""
        self.is_raw = False
        self.arm = FakeBoardArm()
        self.board_time = FakeBoardTime()
        self.namespace = {"arm": self.arm, "time": self.board_time}

    def open(self) -> None:
        self.outbox += b"\r\nMicroPython v1.12 on 2019-12-20; ESP32 module\r\n>>> "

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.outbox = b""

    @property
    def in_waiting(self) -> int:
        return len(self.outbox)

    def read(self, size: int = 1) -> bytes:
        taken, self.outbox = self.outbox[:size], self.outbox[size:]
        return taken

    def write(self, data: bytes) -> int:
        for byte in bytes(data):
            if byte == 0x01:
                self.is_raw = True
                self.outbox += b"raw REPL; CTRL-B to exit\r\n>"
            elif byte == 0x02:
                self.is_raw = False
                self.outbox += b"\r\n>>> "
            elif byte == 0x03:
                self.buffered_code = b""
            elif byte == 0x04 and self.is_raw:
                self._execute()
            elif byte in (0x0A, 0x0D) and not self.is_raw:
                self.outbox += b"\r\n>>> "
            else:
                self.buffered_code += bytes([byte])
        return len(data)

    def _execute(self) -> None:
        source, self.buffered_code = self.buffered_code.decode(), b""
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(source, self.namespace)  # noqa: S102 - that is the point
            self.outbox += b"OK" + captured.getvalue().encode() + b"\x04" + b"\x04>"
        except Exception as exc:
            self.outbox += b"OK\x04" + repr(exc).encode() + b"\x04>"


@contextlib.contextmanager
def fake_board():
    """Run a block with MaxArmLink talking to a fake board."""
    real_serial = maxarm_link.serial.Serial
    maxarm_link.serial.Serial = FakeSerial
    try:
        with maxarm_link.MaxArmLink("/dev/fake") as link:
            yield link
    finally:
        maxarm_link.serial.Serial = real_serial


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  DATACOLLECTION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(name: str, argv: List[str]) -> Tuple[int, FakeSerial]:
    """Run a CLI end to end against a fake board; return its exit code."""
    script = load_script(name)
    real_serial = maxarm_link.serial.Serial
    maxarm_link.serial.Serial = FakeSerial
    wire_holder = {}

    original_open = maxarm_link.MaxArmLink.open

    def remember_open(self):
        original_open(self)
        wire_holder["wire"] = self.port

    maxarm_link.MaxArmLink.open = remember_open
    saved_argv = sys.argv
    sys.argv = [f"{name}.py"] + argv
    try:
        code = script.main()
    finally:
        sys.argv = saved_argv
        maxarm_link.MaxArmLink.open = original_open
        maxarm_link.serial.Serial = real_serial
    return code, wire_holder.get("wire")


def check(label: str, is_ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if is_ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(is_ok)


def test_raw_repl_framing() -> bool:
    print("raw REPL framing:")
    with fake_board() as link:
        wire = link.port
        is_raw = wire.is_raw
        verdict, measured = link.probe((10.0, -160.0, 150.0), 80, 40)
        cached = link.get_cached_position()
        names_left = {n for n in wire.namespace if not n.startswith("__")}
        slept = list(wire.board_time.slept_ms)

    return all([
        check("entered raw mode", is_raw),
        check("probe round trip", verdict is True and measured is not None,
              f"{verdict} {measured}"),
        check("settle delay ran on the board", slept == [120], f"{slept}"),
        check("cached position is the command, not the readback",
              cached == (10.0, -160.0, 150.0), f"{cached}"),
        check("board namespace gained no names", names_left == {"arm", "time"},
              f"{sorted(names_left)}"),
    ])


def test_probe_outcomes() -> bool:
    print("reach_probe outcomes:")
    results = []

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        reached = probe.walk_to((0.0, -160.0, 150.0))
        results.append(check("reachable target", reached.outcome is Outcome.REACHED,
                             f"{len(reached.steps)} steps"))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        far = probe.walk_to((240.0, -20.0, 200.0))
        results.append(check("far target stops short", not far.is_reached,
                             f"{far.outcome.value} at x={far.last_good[0]:.1f}"))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        sagged = probe.walk_to((100.0, -180.0, 50.0))
        results.append(check("desk contact detected and backed out",
                             sagged.outcome is Outcome.Z_DESK,
                             f"{sagged.outcome.value} at z={sagged.last_good[2]:.1f}"))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        out = probe.walk_to((300.0, 0.0, 300.0))
        results.append(check("envelope guard rejects before moving",
                             out.outcome is Outcome.OUT_OF_BOUNDS and not out.steps))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        probe.walk_to((0.0, -160.0, 150.0))
        # Feedback reading ABOVE the command is lag or offset, never a collision.
        high = probe._classify((0.0, -160.0, 150.0), True, (0.0, -160.0, 153.5))
        low = probe._classify((0.0, -160.0, 150.0), True, (0.0, -160.0, 146.5))
        results.append(check("reading high is never a sag", high not in
                             (Outcome.Z_SAG, Outcome.Z_DESK), high.value))
        results.append(check("modest droop in free space is tolerated",
                             low is Outcome.REACHED, low.value))
        # Free space allows droop up to --z-drift; the desk band does not.
        deep = probe._classify((0.0, -160.0, 150.0), True, (0.0, -160.0, 137.0))
        near_desk = probe._classify((0.0, -160.0, 55.0), True, (0.0, -160.0, 50.5))
        results.append(check("large droop in free space is Z_SAG",
                             deep is Outcome.Z_SAG, deep.value))
        results.append(check("small droop near the desk is still Z_DESK",
                             near_desk is Outcome.Z_DESK, near_desk.value))

        # A constant Z error must stop registering once it is the baseline.
        probe.z_baseline = 4.0
        absorbed = probe._classify((0.0, -160.0, 150.0), True, (0.0, -160.0, 146.5))
        results.append(check("baseline absorbs ordinary Z error",
                             absorbed is Outcome.REACHED, absorbed.value))
        probe.z_baseline = 0.0

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        # Drive outward until the servo clamp pins the arm.
        stalled = probe.walk_to((240.0, -20.0, 200.0))
        results.append(check("clamp is detected as a stall",
                             stalled.outcome is Outcome.STALLED, stalled.outcome.value))
        results.append(check("stall reports the measured pose, not the command",
                             stalled.last_measured is not None
                             and stalled.last_measured != stalled.last_good,
                             f"measured {stalled.last_measured}"))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        tracking = probe.walk_to((0.0, -160.0, 150.0))
        lag = max(abs(tracking.edge[i] - tracking.target[i]) for i in range(3))
        results.append(check("lagging arm still counts as reached",
                             tracking.outcome is Outcome.REACHED, f"lag {lag:.1f} mm"))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        probe.walk_to((150.0, -120.0, 150.0), step_mm=15.0)
        before = math.hypot(probe.position[0], probe.position[1])
        backed = probe.back_off_from_limit(20.0)
        after = math.hypot(probe.position[0], probe.position[1])
        results.append(check("stall back-off unloads the arm inward",
                             backed and after < before - 5.0,
                             f"reach {before:.0f} -> {after:.0f} mm"))


    return all(results)


def test_survives_board_chatter() -> bool:
    """Unsolvable targets make the board print diagnostics over our reply."""
    print("board chatter on unsolvable targets:")
    from envelope_model import EnvelopeModel
    with fake_board() as link:
        model = EnvelopeModel(link)
        far = model.is_predicted_reachable((240.0, -240.0, 200.0))
        near = model.is_predicted_reachable((0.0, -160.0, 150.0))
        # A line that straddles the limit: chatter lands mid-reply.
        line = model.predict_line(-160.0, 150.0, [float(x) for x in range(0, 248, 4)])
    return all([
        check("unreachable point parses through the noise", far is False),
        check("reachable point still parses", near is True),
        check("mixed line returns one verdict per point", len(line) == 62, f"{len(line)}"),
        check("line has both reachable and unreachable", any(line) and not all(line)),
    ])


def test_no_fly_zone() -> bool:
    """The base exclusion square must block targets AND paths across it."""
    print("no-fly zone around the robot base:")
    results = []
    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        arm = link.port.arm

        before = arm.move_count
        inside = probe.walk_to((30.0, -40.0, 120.0))
        results.append(check("target inside the square is refused without moving",
                             inside.outcome is Outcome.NO_FLY
                             and arm.move_count == before, inside.outcome.value))

        # Straddling the base: both ends legal (x >= 0, outside the square),
        # but the straight line between them clips the corner.
        probe.position = (100.0, -100.0, 120.0)
        before = arm.move_count
        across = probe.walk_to((20.0, 100.0, 120.0))
        results.append(check("path across the square is refused without moving",
                             across.outcome is Outcome.NO_FLY
                             and arm.move_count == before, across.outcome.value))

        detour = probe.route_around((20.0, 100.0, 120.0))
        legs = [(100.0, -100.0, 120.0)] + list(detour)
        clear = all(is_segment_clear(legs[n], legs[n + 1]) for n in range(len(legs) - 1))
        results.append(check("routed detour clears the square on every leg", clear,
                             f"{len(detour)} waypoints"))
        results.append(check("no detour waypoint sits inside the square",
                             not any(is_in_no_fly(p[0], p[1]) for p in detour)))

    with fake_board() as link:
        probe = ReachProbe(link, step_ms=0, settle_ms=0)
        # Park the arm inside the zone, as a sweep once did by aiming at x=71
        # and landing at 70. Before the escape existed this deadlocked: the
        # guard refuses any move that starts inside, so it could not get out.
        probe.step_to((70.0, 20.0, 120.0))
        trapped = is_in_no_fly(probe.position[0], probe.position[1])
        escaped = probe.escape_no_fly()
        landed = probe.position
        free = not is_in_no_fly(landed[0], landed[1])
        after = probe.walk_around_to((150.0, -150.0, 120.0), step_mm=15.0)
        results.append(check("arm can be parked inside the zone", trapped))
        results.append(check("escape gets it out", escaped and free,
                             f"landed at {tuple(round(v) for v in landed)}"))
        results.append(check("escape does not overshoot far",
                             max(abs(landed[0]), abs(landed[1])) < NO_FLY_PLANNING_HALF + 15,
                             f"max-norm {max(abs(landed[0]), abs(landed[1])):.0f}"))
        results.append(check("and it can move normally afterwards",
                             after.is_reached, after.outcome.value))

    results.append(check("planning floor clears the arm's own error",
                         NO_FLY_PLANNING_HALF - NO_FLY_HALF >= 3.0,
                         f"{NO_FLY_PLANNING_HALF - NO_FLY_HALF:.0f} mm margin"))

    return all(results)


def test_model_predicts_without_moving() -> bool:
    print("envelope model (motion-free):")
    with fake_board() as link:
        model = EnvelopeModel(link)
        arm = link.port.arm
        moves_before = arm.move_count

        grid = [float(x) for x in range(0, 250, 4)]
        span = model.find_line_span(-160.0, 150.0, grid)
        blind = model.is_predicted_reachable((0.0, -20.0, 150.0))
        empty = model.find_line_span(-280.0, 200.0, grid)
        moves_after = arm.move_count

    return all([
        check("predicting moved the arm zero times", moves_after == moves_before,
              f"{moves_after - moves_before} moves"),
        check("found a span on a good line", span is not None, f"{span}"),
        check("rejects the blind cylinder", blind is False),
        check("unreachable line returns no span", empty is None, f"{empty}"),
    ])


def test_cube_calibration() -> bool:
    """The cube probe must actually collect moves, not silently skip them all."""
    print("cube calibration:")
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "cube.csv"
        code, wire = run_script(
            "calibrate-readback",
            ["--mode", "cube", "--rounds", "2", "--sizes", "1,2,5,20", "--half", "35",
             "--step-ms", "0", "--settle-ms", "0", "--extra-reads", "0",
             "--read-gap-ms", "0", "--out", str(out)])
        rows = list(csv.DictReader(out.read_text().splitlines()))

    sizes = {float(r["requested_mm"]) for r in rows}
    ratios = [float(r["travel_ratio"]) for r in rows if float(r["requested_mm"]) >= 20]
    return all([
        check("exits cleanly", code == 0),
        check("collected moves at every size", sizes == {1.0, 2.0, 5.0, 20.0}, f"{sorted(sizes)}"),
        check("cube stayed inside the envelope",
              all(float(r["commanded_z"]) <= 209.0 for r in rows)),
        check("lagging arm shows a travel ratio below 1",
              bool(ratios) and max(ratios) < 1.0, f"max {max(ratios):.2f}" if ratios else "none"),
    ])


def test_endpoint_cli() -> bool:
    print("test-endpoint CLI:")
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "results.csv"
        code, _ = run_script("test-endpoint", ["0", "-160", "150", "--out", str(out),
                                               "--step-ms", "0", "--settle-ms", "0"])
        lines = out.read_text().strip().splitlines()

    return all([
        check("exits cleanly", code == 0),
        check("one result row", len(lines) == 2, f"{len(lines)} lines"),
        check("target reported reached", "reached" in lines[1]),
    ])


def test_contour_tracer() -> bool:
    """The polar tracer must find both edges of a ring and skip our own fences."""
    print("contour tracer:")
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "reach.csv"
        code, wire = run_script(
            "trace-contour",
            ["--out", str(out), "--z", "150", "--z-to", "150", "--angle-step", "30",
             "--grid", "8", "--step-ms", "0", "--settle-ms", "0"])
        rows = list(csv.DictReader(out.read_text().splitlines()))

    arm = [r for r in rows if r["limit_kind"] == "arm"]
    skipped = [r for r in rows if r["outcome"] == "skipped"]
    radii = [math.hypot(float(r["x"]), float(r["y"])) for r in arm]
    return all([
        check("exits cleanly", code == 0),
        check("found outer-edge arm limits",
              any(r["edge"] == "r_max" for r in arm), f"{len(arm)} arm rows"),
        check("fence points are skipped without moving", bool(skipped),
              f"{len(skipped)} skipped"),
        check("no measured point sits in the no-fly square",
              not any(is_in_no_fly(float(r["x"]), float(r["y"])) for r in rows)),
        check("outer radius is near-constant across bearings",
              bool(radii) and max(radii) - min(radii) < 10.0,
              f"spread {max(radii) - min(radii):.1f} mm" if radii else "none"),
    ])


def test_extends_past_a_bad_prediction() -> bool:
    """A probe must not stop on our own overshoot cap while the arm still moves."""
    print("probe extension:")
    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "tiny.csv"
        # 4 mm overshoot is far too small; without extension every probe would
        # end on our cap rather than at the arm's limit.
        run_script("trace-contour",
                   ["--out", str(out), "--z", "150", "--z-to", "150",
                    "--angle-step", "45", "--grid", "8", "--overshoot", "4",
                    "--step-ms", "0", "--settle-ms", "0"])
        rows = [r for r in csv.DictReader(out.read_text().splitlines())
                if r["outcome"] != "skipped"]
    return all([
        check("probes ran", bool(rows), f"{len(rows)} rows"),
        check("all reached the arm's limit, not our cap",
              all(r["limit_kind"] == "arm" for r in rows),
              f"{sum(1 for r in rows if r['limit_kind'] == 'arm')} of {len(rows)}"),
    ])


def main() -> int:
    passed = all([test_raw_repl_framing(),
                  test_probe_outcomes(),
                  test_survives_board_chatter(),
                  test_no_fly_zone(),
                  test_model_predicts_without_moving(),
                  test_cube_calibration(),
                  test_endpoint_cli(),
                  test_contour_tracer(),
                  test_extends_past_a_bad_prediction()])
    print("\nALL PASS" if passed else "\nFAILURES")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
