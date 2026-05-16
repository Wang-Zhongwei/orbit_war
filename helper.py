"""
helper.py — shared constants and utility functions used by both
rule_based.py and rl.py.

Observation schema (from kaggle-environments orbit_wars):
  planets           : list of [id, owner, x, y, radius, ships, production]
  fleets            : list of [id, owner, x, y, angle, from_planet_id, ships]
  player            : int  (0–3)
  angular_velocity  : float  (radians / turn, for inner rotating planets)
  comet_planet_ids  : list of int  (planet IDs that are comets)
  comets            : list of comet group dicts  {planet_ids, paths, path_index}

Action format: [from_planet_id, angle_in_radians, num_ships]
"""

from functools import cache
import math
from typing import NamedTuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, SUN_RADIUS, CENTER, ROTATION_RADIUS_LIMIT, point_to_segment_distance

DirectedFleet = NamedTuple("DirectedFleet", [("src_id", int), ("tgt_id", int), ("angle", float), ("ships", int), ("arrive_tick", int)])

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
MAX_SPEED: float = 6.0
MAX_TICKS: int = 25


# ---------------------------------------------------------------------------
# Observation unpacking
# ---------------------------------------------------------------------------

def unpack_obs(obs) -> tuple[int, list[Planet], set[int], float, list, dict[int, list[list[float]]]]:
    """
    Parse an observation (dict or namedtuple) into typed components.

    Returns
    -------
    player           : int                          — this agent's player ID
    planets          : list[Planet]                 — all planets as namedtuples
    comet_ids        : set[int]                     — planet IDs that are comets this turn
    angular_velocity : float                        — inner-planet rotation speed (rad/turn)
    fleets           : list[Fleet]                  — all active fleets as namedtuples
    comet_paths      : dict[int, list[list[float]]] — maps each comet planet ID to its
                       remaining waypoints [[x, y], ...] (index path_index+1 onward).
                       An empty list means the comet expires next tick.
    """
    player           = obs.get("player", 0)
    raw_planets      = obs.get("planets", [])
    comet_ids        = set(obs.get("comet_planet_ids", []))
    angular_velocity = obs.get("angular_velocity", 0.0)
    raw_fleets       = obs.get("fleets", [])
    raw_comets       = obs.get("comets", [])

    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f) for f in raw_fleets]

    # Build a per-planet remaining-path lookup from the comet group bookkeeping.
    # path_index is the step that was just applied, so future positions start
    # at path_index + 1.
    comet_paths: dict[int, list[list[float]]] = {}
    for group in raw_comets:
        idx        = group.get("path_index", -1)
        pid_list   = group.get("planet_ids", [])
        paths_list = group.get("paths", [])
        for pid, path in zip(pid_list, paths_list):
            comet_paths[pid] = path[idx + 1:]

    return player, planets, comet_ids, angular_velocity, fleets, comet_paths

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

@cache
def fleet_speed(num_ships: int) -> float:
    """Speed of a fleet of `num_ships`."""
    num_ships = min(max(num_ships, 1), 1000)  # clamp to 1-1000
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(num_ships) / math.log(1000.0)) ** 1.5


def hits_sun(src_x: float, src_y: float, angle: float, spd: float, ticks: int = MAX_TICKS) -> bool:
    """Return True if a straight fleet trajectory clips the sun danger zone."""
    end_x = src_x + math.cos(angle) * spd * ticks
    end_y = src_y + math.sin(angle) * spd * ticks
    return point_to_segment_distance((CENTER, CENTER), (src_x, src_y), (end_x, end_y)) <= SUN_RADIUS


def planet_trajectory(p, ticks: int, angular_velocity: float) -> list[tuple[float, float]]:
    """Return a list of (x, y) positions for a moving planet over `ticks` turns."""
    return [predict_planet_position(p, t, angular_velocity) for t in range(1, ticks + 1)]

def _seg_seg_min_dist(ax: float, ay: float, bx: float, by: float,
                      cx: float, cy: float, dx: float, dy: float) -> float:
    """Minimum distance between segment AB and segment CD."""
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) > 1e-10:
        t = ((cx - ax) * sy - (cy - ay) * sx) / denom
        u = ((cx - ax) * ry - (cy - ay) * rx) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(
        point_to_segment_distance((ax, ay), (cx, cy), (dx, dy)),
        point_to_segment_distance((bx, by), (cx, cy), (dx, dy)),
        point_to_segment_distance((cx, cy), (ax, ay), (bx, by)),
        point_to_segment_distance((dx, dy), (ax, ay), (bx, by)),
    )


def path_blocked(sx: float, sy: float, ex: float, ey: float, ticks: int, spd: float,
                  blockers: list, traj_cache: dict) -> bool:
    """Return True if the fleet collides with any blocker during any tick interval [t-1, t].

    At each tick interval, checks the minimum distance between the fleet's path segment
    and the planet's path segment (or point for static planets) against p.radius.
    traj_cache[p.id][t-1] is the planet's position at tick t; its position at tick 0
    is (p.x, p.y).
    """
    dist_total = math.hypot(ex - sx, ey - sy)
    if dist_total == 0:
        return False
    ux = (ex - sx) / dist_total
    uy = (ey - sy) / dist_total
    for p in blockers:
        ptraj = traj_cache.get(p.id, [])
        for t in range(1, ticks + 1):
            pfx = sx + ux * spd * (t - 1)
            pfy = sy + uy * spd * (t - 1)
            fx  = sx + ux * spd * t
            fy  = sy + uy * spd * t
            if ptraj:
                if t > len(ptraj):
                    break
                px, py   = ptraj[t - 1]
                ppx, ppy = ptraj[t - 2] if t > 1 else (p.x, p.y)
                if _seg_seg_min_dist(pfx, pfy, fx, fy, ppx, ppy, px, py) <= p.radius:
                    return True
            else:
                if point_to_segment_distance((p.x, p.y), (pfx, pfy), (fx, fy)) <= p.radius:
                    return True
    return False


def find_angle(src, tgt, ships: int, traj_cache: dict,
               blockers: list | None = None) -> tuple[float|None, int|None]:
    """Angle + arrive tick toward `tgt`, handling both moving and static targets.

    Returns (None, None) when no intercept is found or every candidate path clips
    the sun or is occluded by a planet.  src and tgt are excluded from blockers
    automatically.  For moving targets, blocked intercepts are skipped (a blocking
    planet may rotate clear at a later tick); sun blocks abort immediately.
    """
    spd = fleet_speed(ships)

    if tgt.id in traj_cache:
        traj = traj_cache[tgt.id]
        n = len(traj)

        def gap(idx):
            tick = idx + 1
            tx, ty = traj[idx]
            dx, dy = tx - src.x, ty - src.y
            return spd * tick - (math.hypot(dx, dy) - src.radius)

        lo, hi, first = 0, n - 1, n
        while lo <= hi:
            mid = (lo + hi) // 2
            if gap(mid) >= -tgt.radius:
                first = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if first >= n or abs(gap(first)) >= tgt.radius:
            return None, None
        tick = first + 1
        tx, ty = traj[first]
        dx, dy = tx - src.x, ty - src.y
        angle = math.atan2(dy, dx)
        lx = src.x + math.cos(angle) * src.radius
        ly = src.y + math.sin(angle) * src.radius
        if hits_sun(lx, ly, angle, spd, tick):
            return None, None
        if blockers and path_blocked(lx, ly, tx, ty, tick, spd, blockers, traj_cache):
            return None, None
        return angle, tick
    else:
        dx, dy = tgt.x - src.x, tgt.y - src.y
        angle = math.atan2(dy, dx)
        dist = math.hypot(dx, dy) - src.radius
        tick = math.ceil((dist - tgt.radius) / spd)

        lx = src.x + math.cos(angle) * src.radius
        ly = src.y + math.sin(angle) * src.radius
        if hits_sun(lx, ly, angle, spd, tick):
            return None, None

        if blockers and path_blocked(lx, ly, tgt.x, tgt.y, tick, spd, blockers, traj_cache):
            return None, None

        return angle, tick


def reinforcement_plans(
    mine, under_attack: dict[int, list[DirectedFleet]], my_fleets: list[DirectedFleet], min_ships: int
) -> dict:
    """Identify planets that need reinforcement and how many ships are required."""
    plans: dict = {}
    for p in mine:
        if p.id not in under_attack:
            continue

        friendly = [(f.arrive_tick, +f.ships) for f in my_fleets if f.tgt_id == p.id]
        hostile  = [(f.arrive_tick, -f.ships) for f in under_attack[p.id]]
        events   = sorted(friendly + hostile, key=lambda e: (e[0], -e[1]))

        avail     = float(p.ships)
        prev_tick = 0
        for tick, delta in events:
            avail    += (tick - prev_tick) * p.production + delta
            prev_tick = tick
            if delta < 0 and avail < 0:  # just processed a hostile attack
                plans[p] = {
                    "ships_needed":   max(min_ships, int(abs(avail))),
                    "needed_by_tick": tick,
                }
                break
    return plans


def predicted_garrison(tgt: Planet, turns: float) -> float:
    """
    Estimate how many ships defend `tgt` after `turns` turns of production
    (only relevant if the planet still belongs to a non-neutral owner; neutral
    planets don't produce, but we err on the safe side).
    """

    return tgt.ships + (tgt.owner > -1) * tgt.production * turns


def distance(a: Planet, b: Planet) -> float:
    """Euclidean distance between two planets."""
    return math.hypot(a.x - b.x, a.y - b.y)


def predict_planet_position(planet, turns_ahead, angular_velocity):
    """
    Predict the position of a planet after a certain number of turns.

    Args:
        planet: The Planet object.
        turns_ahead: Number of turns to predict ahead.
        angular_velocity: The angular velocity of the planet.

    Returns:
        Tuple (x, y) representing the predicted position.
    """
    dx = planet.x - CENTER
    dy = planet.y - CENTER
    orbital_r = math.sqrt(dx * dx + dy * dy)
    if orbital_r + planet.radius >= ROTATION_RADIUS_LIMIT:
        return planet.x, planet.y
    initial_angle = math.atan2(dy, dx)
    current_angle = initial_angle + angular_velocity * turns_ahead
    return CENTER + orbital_r * math.cos(current_angle), CENTER + orbital_r * math.sin(current_angle)

# todo: use static angle and find angle later
def aim_for_planet(target: Planet, my_planet: Planet, angular_velocity, is_my: bool=False) -> tuple[float, int, bool, int] | None:
    """
    Determine the angle and number of ships to send to a target planet.

    Args:
        target: The target Planet object.
        my_planet: The source Planet object.
        angular_velocity: The angular velocity of the planets.

    Returns:
        Tuple (angle, ships_to_send) if valid, None otherwise.
    """
    my_x, my_y = my_planet.x, my_planet.y
    max_available: int = my_planet.ships

    # Initial guess for position and ships
    predicted_x, predicted_y = target.x, target.y
    est_ships: int = max(10, max_available // 2)
    can_capture = False

    # Iterative refinement to predict target position accurately
    converged = False
    for _ in range(10):
        # Current assumption
        dx = predicted_x - my_x
        dy = predicted_y - my_y
        distance = math.sqrt(dx * dx + dy * dy) - my_planet.radius
        est_speed = fleet_speed(est_ships)
        turns_to_arrive = max(1.0, int(distance / est_speed))

        # Update estimated ships
        if not is_my:
            new_est_ships = predicted_garrison(target, turns_to_arrive) + 1
        else:
            new_est_ships = max(1, max_available // 2)

        # Can we capture?
        if new_est_ships < max_available:
            can_capture = True
        else:
            new_est_ships = max_available - 1
            can_capture = False

        # Update position
        est_speed = fleet_speed(new_est_ships)
        turns_to_arrive = max(1.0, int(distance / est_speed))
        new_predicted_x, new_predicted_y = predict_planet_position(target, turns_to_arrive, angular_velocity)

        # Stopping criteria
        error = math.sqrt((predicted_x - new_predicted_x) ** 2 + (predicted_y - new_predicted_y) ** 2)
        delta_ships = abs(new_est_ships - est_ships)
        est_ships = new_est_ships
        predicted_x, predicted_y = new_predicted_x, new_predicted_y

        if error < 0.01 and delta_ships == 0:
            converged = True
            break

    # Check reachability
    if not converged:
        return None

    # Check sun collision against the actual flight path (src → predicted position).
    if point_to_segment_distance((CENTER, CENTER), (my_x, my_y), (predicted_x, predicted_y)) < SUN_RADIUS + 1:
        return None

    angle: float = math.atan2(predicted_y - my_y, predicted_x - my_x)

    return angle, est_ships, can_capture, turns_to_arrive
