#!/usr/bin/env python3
"""Routing checks: retargeting, zones, staging and swing splitting.

All of this is arithmetic, so it runs with no board and no fake board. If the
routing is wrong every other layer inherits the mistake, which makes these the
cheapest checks in the project and the most worth having.

`Router` has no protocol to drive even if it wanted one, which is why these
are the cheap checks: no board, no fake board, no waiting.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, tuning                            # noqa: E402
from maxarm.config import Limits                               # noqa: E402
from maxarm.route import NoRoute, Router                       # noqa: E402
from maxarm.status import MoveStatus                           # noqa: E402
from maxarm.zones import BASE_HALF_MM, ExclusionZone, ZoneSet  # noqa: E402
from harness import check                                      # noqa: E402


def make_router(limits: Limits = None, zones=()) -> Router:
    return Router(limits or Limits(), ZoneSet(zones))


def is_path_clear(router: Router, start, hops) -> bool:
    """True if every hop of a route clears the zones end to end."""
    cursor = start
    for hop in hops:
        if not router.zones.is_path_clear(cursor, hop.position, margin=0.0):
            return False
        cursor = hop.position
    return True


def test_target_verdicts() -> bool:
    print("target verdicts:")
    router = make_router()
    _, inside, _ = router.resolve((30.0, -40.0, 120.0))
    good, good_status, _ = router.resolve((150.0, -150.0, 60.0))
    edge, edge_status, edge_detail = router.resolve((0.0, -292.0, 60.0))
    _, far, _ = router.resolve((0.0, -400.0, 60.0))
    low, low_status, _ = router.resolve((150.0, -150.0, 40.0))
    _, sunk, _ = router.resolve((150.0, -150.0, 20.0))
    return all([
        check("a target in the base square is refused", inside is MoveStatus.NO_FLY,
              inside.value),
        check("a good target is passed through unchanged",
              good_status is MoveStatus.REACHED
              and math.dist(good, (150.0, -150.0, 60.0)) < 0.05),
        check("a target just past the envelope is pulled in",
              edge_status is MoveStatus.APPROXIMATED, f"{edge_status.value}: {edge_detail}"),
        check("and the pull is within the retarget budget",
              math.dist(edge, (0.0, -292.0, 60.0)) <= tuning.RETARGET_LIMIT_MM,
              f"{math.dist(edge, (0.0, -292.0, 60.0)):.1f} mm"),
        check("a target far outside is refused outright", far is MoveStatus.UNREACHABLE,
              far.value),
        check("a target just below the z floor is lifted to it",
              low_status is MoveStatus.APPROXIMATED and abs(low[2] - Limits().z_floor) < 0.05,
              f"z {low[2]:.1f}"),
        check("a target well below it is refused rather than quietly moved 28 mm",
              sunk is MoveStatus.UNREACHABLE, sunk.value),
    ])


def test_operator_limits() -> bool:
    """Operator limits are the desk, not the arm: they clip, they do not refuse."""
    print("operator limits:")
    router = make_router(Limits(x=(0.0, None), z=(48.0, 150.0)))
    left, left_status, _ = router.resolve((-10.0, -200.0, 60.0))
    high, high_status, _ = router.resolve((0.0, -200.0, 158.0))
    _, open_status, _ = make_router().resolve((-10.0, -200.0, 60.0))
    return all([
        check("a target left of the fence is clipped to it",
              left_status is MoveStatus.APPROXIMATED and left[0] >= -0.05, f"x {left[0]:.1f}"),
        check("the same target is fine without the fence",
              open_status is MoveStatus.REACHED, open_status.value),
        check("a z ceiling clips too",
              high_status is MoveStatus.APPROXIMATED and high[2] <= 150.05, f"z {high[2]:.1f}"),
    ])


def test_ordinary_move_is_one_hop() -> bool:
    """The case that matters most: a plain move in open space is one command.

    Chunking a safe move into waypoints is what made the first version of this
    library stutter. Splitting is for swings that leave safe space, and nothing
    else.
    """
    print("an ordinary move is a single hop:")
    router = make_router()
    plain = router.route((160.0, -160.0, 120.0), (200.0, -120.0, 110.0))
    across = router.route((160.0, -160.0, 150.0), (-160.0, -160.0, 150.0))
    nudge = router.route((150.0, -150.0, 90.0), (152.0, -150.0, 90.0))
    return all([
        check("a short move high over the desk is one hop", len(plain) == 1,
              f"{[hop.purpose for hop in plain]}"),
        check("and it is flown, not stepped", not plain[0].is_stepped),
        check("a 2 mm jog is one hop", len(nudge) == 1),
        check("a wide sweep is split only as far as its swing needs",
              1 <= len(across) <= 4, f"{len(across)} hops"),
    ])


def test_unsafe_swing_is_split() -> bool:
    """The arm does not move in straight lines, so the chord is not the path.

    An obstacle that the straight line misses but the swing runs through is the
    case that isolates this: the route has to notice the *swing*, and it fixes
    it by halving the hop rather than by inventing a third way to drive.
    """
    print("splitting a hop whose swing leaves safe space:")
    start, target = (-170.0, -120.0, 90.0), (170.0, -120.0, 90.0)
    swing = geometry.pulse_path(start, target, samples=tuning.SWING_SAMPLES)
    deviation = geometry.path_deviation(start, target)
    # A box the straight chord clears but the bowed swing passes through.
    bulge = min(swing, key=lambda point: point[1])
    wall = ExclusionZone("wall", bulge[0] - 40.0, bulge[0] + 40.0,
                         bulge[1] - 15.0, bulge[1] + 15.0)
    open_route = make_router().route(start, target)
    walled = make_router(zones=(wall,))
    guarded_route = walled.route(start, target)

    cursor = start
    is_swing_clear = True
    for hop in guarded_route:
        for point in geometry.pulse_path(cursor, hop.position, samples=24):
            is_swing_clear &= walled.zones.is_clear(point, margin=0.0)
        cursor = hop.position
    return all([
        check("the swing really does leave the straight line", deviation > 15.0,
              f"{deviation:.0f} mm off the chord"),
        check("the obstacle is off the straight line",
              make_router().zones.is_path_clear(start, target)
              and not wall.contains(_midpoint(start, target)),
              "chord misses it"),
        check("without the obstacle the sweep is not split further than needed",
              len(open_route) <= len(guarded_route),
              f"{len(open_route)} vs {len(guarded_route)} hops"),
        check("with it, the route grows extra hops", len(guarded_route) > len(open_route),
              f"{len(guarded_route)} hops"),
        check("and the arm's actual swung path clears the obstacle", is_swing_clear),
    ])


def test_routes_around_the_base() -> bool:
    print("routing around the base square:")
    router = make_router(Limits(z=(48.0, None)))
    start, target = (-150.0, -30.0, 60.0), (150.0, -30.0, 60.0)
    hops = router.route(start, target)
    closest = min(max(abs(hop.position[0]), abs(hop.position[1])) for hop in hops)

    cursor, swing_clearance = start, 1e9
    for hop in hops:
        if not hop.is_stepped:
            for point in geometry.pulse_path(cursor, hop.position, samples=24):
                swing_clearance = min(swing_clearance, max(abs(point[0]), abs(point[1])))
        cursor = hop.position
    return all([
        check("the straight path really does cross the base",
              not router.zones.is_path_clear(start, target, margin=0.0)),
        check("every hop clears the square", is_path_clear(router, start, hops),
              f"{len(hops)} hops"),
        check("no waypoint comes near the base", closest > BASE_HALF_MM,
              f"closest max-norm {closest:.1f} mm"),
        check("and neither does the swung path between them",
              swing_clearance > BASE_HALF_MM, f"{swing_clearance:.1f} mm"),
        check("every waypoint stays reachable",
              all(geometry.is_firmware_reachable(hop.position) for hop in hops)),
    ])


def test_detours_from_outside_the_orbit() -> bool:
    """A detour that starts further out than the orbit must still go radial first.

    Going straight from a distant start to the first arc point is a chord
    across two bearings, and it cuts the corner -- this exact route (r=161 in,
    orbit at r=114) refused itself as unroutable until the radial leg was made
    unconditional.
    """
    print("detouring from outside the orbit radius:")
    router = make_router(Limits(z=(52.0, None)))
    start, target = (-150.0, -60.0, 120.0), (200.0, -120.0, 52.0)
    start_radius = math.hypot(start[0], start[1])
    try:
        hops = router.route(start, target)
        refused = ""
    except NoRoute as error:
        hops, refused = [], str(error)
    return all([
        check("the start really is outside the orbit radius",
              start_radius > router.zones.orbit_radius(),
              f"r={start_radius:.0f} against orbit r={router.zones.orbit_radius():.0f}"),
        check("the straight path really does cross the base",
              not router.zones.is_path_clear(start, target)),
        check("a route is found rather than refused", not refused, refused),
        check("and every hop of it clears the zones",
              is_path_clear(router, start, hops), f"{len(hops)} hops"),
    ])


def test_desk_descent_is_staged() -> bool:
    """Travel high, drop to the guard band flying, walk the last few millimetres."""
    print("descending to the desk:")
    router = make_router()
    hops = router.route((160.0, -160.0, 120.0), (200.0, -120.0, 50.0))
    purposes = [hop.purpose for hop in hops]
    stepped = [hop for hop in hops if hop.is_stepped]
    guard_top = Limits().z_floor + tuning.DESK_GUARD_MM
    walked_mm = (math.dist(hops[-2].position, hops[-1].position)
                 if len(hops) > 1 else float("inf"))
    return all([
        check("it travels before it descends", purposes[0] == "travel", f"{purposes}"),
        check("the travel leg clears the floor by the desk clearance",
              hops[0].position[2] >= Limits().z_floor + tuning.DESK_CLEARANCE_MM,
              f"travel z {hops[0].position[2]:.1f}"),
        check("exactly one hop is stepped, and it is the last", len(stepped) == 1
              and stepped[0] is hops[-1], f"{purposes}"),
        check("the stepped part starts at the top of the guard band",
              abs(hops[-2].position[2] - guard_top) < 0.05, f"z {hops[-2].position[2]:.1f}"),
        check("so the walk is short, not the whole descent",
              walked_mm <= tuning.DESK_GUARD_MM + 0.05, f"{walked_mm:.0f} mm walked"),
    ])


def test_never_crosses_low() -> bool:
    """The invariant: cross above the lower end of the move, never along it.

    Both endpoints clearing the desk is not enough -- the arm sags as it
    extends, so an XY path flown at the lower endpoint's height can scrape
    between them.
    """
    print("crossing height:")
    clearance = tuning.DESK_CLEARANCE_MM
    cases = [((100.0, -120.0, 50.0), (180.0, -180.0, 50.0), "both low", Limits()),
             ((160.0, -160.0, 90.0), (200.0, -120.0, 50.0), "high to low", Limits()),
             ((100.0, -120.0, 50.0), (180.0, -180.0, 120.0), "low to high", Limits()),
             (geometry.HOME_COMMAND, (160.0, -160.0, 90.0), "from home", Limits()),
             ((-150.0, -30.0, 55.0), (150.0, -30.0, 55.0), "around the base",
              Limits(z=(48.0, None)))]

    faults, lifted = [], []
    for start, target, label, limits in cases:
        hops = make_router(limits).route(start, target)
        crossing = [hop for hop in hops if hop.purpose == "travel"]
        floor = min(start[2], target[2]) + clearance
        if any(hop.position[2] < floor - 0.01 for hop in crossing):
            faults.append(f"{label}: {[round(hop.position[2], 1) for hop in crossing]}")
        if crossing and crossing[0].position[2] > start[2] + 0.01:
            lifted.append(label)

    flat = make_router().route((100.0, -120.0, 50.0), (180.0, -180.0, 50.0))
    return all([
        check("no crossing hop runs at or below the lower endpoint", not faults,
              "; ".join(faults)),
        check("a low move is lifted before it travels", "both low" in lifted, f"{lifted}"),
        check("and the lift is the first thing it does", flat[0].purpose == "lift",
              f"{[hop.purpose for hop in flat]}"),
        check("then it comes down stepped at the far end", flat[-1].is_stepped,
              f"{[hop.purpose for hop in flat]}"),
    ])


def test_travel_height_stays_reachable() -> bool:
    """Lifting shrinks the reachable radius -- a naive lift strands the arm."""
    print("travel height against the envelope:")
    router = make_router(Limits(z=(48.0, None)))
    start = (0.0, -280.0, 50.0)
    target = (280.0 * math.sin(math.radians(30)), -280.0 * math.cos(math.radians(30)), 50.0)
    hops = router.route(start, router.resolve(target)[0])
    heights = [hop.position[2] for hop in hops]
    span_at_top = router.envelope_span(max(heights))
    return all([
        check("every waypoint is still reachable",
              all(geometry.is_firmware_reachable(hop.position) for hop in hops),
              f"heights {[round(h) for h in heights]}"),
        check("the travel height keeps 280 mm in range",
              span_at_top is not None and span_at_top[1] >= 279.0,
              f"r_max {span_at_top[1]:.1f} at z={max(heights):.0f}" if span_at_top else "none"),
    ])


def test_operator_zone() -> bool:
    """An extra zone behaves like the base one, and the base one stays regardless."""
    print("operator exclusion zones:")
    mug = ExclusionZone("mug", 80.0, 140.0, -220.0, -160.0)
    router = make_router(zones=(mug,))
    _, status, _ = router.resolve((110.0, -190.0, 60.0))
    start = (60.0, -200.0, 90.0)
    try:
        hops = router.route(start, (200.0, -190.0, 90.0))
        is_routed = is_path_clear(router, start, hops)
    except NoRoute:
        is_routed = True             # refusing is also an acceptable answer
    tall = ExclusionZone("card box", 80.0, 140.0, -220.0, -160.0, z_max=70.0)
    over = make_router(zones=(tall,))
    return all([
        check("a target inside the zone is refused", status is MoveStatus.NO_FLY,
              status.value),
        check("a path through it is detoured or refused, never driven", is_routed),
        check("the base square is present even with operator zones",
              any(zone.name == "base" for zone in router.zones.zones)),
        check("a zone with a ceiling can be flown over",
              over.zones.is_path_clear((60.0, -200.0, 90.0), (200.0, -190.0, 90.0))),
        check("but not driven through at its own height",
              not over.zones.is_path_clear((60.0, -200.0, 60.0), (200.0, -190.0, 60.0))),
    ])


def test_escapes_a_zone_it_starts_in() -> bool:
    """Somebody pushed the arm over the base by hand. Get out before anything else."""
    print("starting inside a zone:")
    router = make_router()
    start = (40.0, -50.0, 120.0)
    hops = router.route(start, (200.0, -120.0, 120.0))
    return all([
        check("the start really is inside a zone", not router.zones.is_clear(start)),
        check("the first hop is an escape", hops[0].purpose == "escape",
              f"{[hop.purpose for hop in hops]}"),
        check("it is one hop, not halved back into the zone",
              [hop.purpose for hop in hops].count("escape") == 1),
        check("the escape leaves the zone", router.zones.is_clear(hops[0].position),
              "({:.0f}, {:.0f}, {:.0f})".format(*hops[0].position)),
        check("and it goes straight out, not around",
              abs(geometry.bearing_degrees(*hops[0].position[:2])
                  - geometry.bearing_degrees(*start[:2])) < 0.5),
    ])


def _midpoint(a, b):
    return tuple((a[i] + b[i]) / 2.0 for i in range(3))


TESTS = [test_target_verdicts, test_operator_limits, test_ordinary_move_is_one_hop,
         test_unsafe_swing_is_split, test_routes_around_the_base,
         test_detours_from_outside_the_orbit,
         test_desk_descent_is_staged, test_never_crosses_low,
         test_travel_height_stays_reachable, test_operator_zone,
         test_escapes_a_zone_it_starts_in]
