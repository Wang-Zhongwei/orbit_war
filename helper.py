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

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, SUN_RADIUS, CENTER, ROTATION_RADIUS_LIMIT, point_to_segment_distance

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
# Conservative radius of the sun's danger zone (fleets clipping this are lost)

# Maximum fleet speed (config default: shipSpeed = 6.0).
# Speed scales with fleet size: 1 ship → 1 unit/turn, large fleets → MAX_SPEED.
MAX_SPEED: float = 6.0

# todo: to be tuned
# Comet ROI multiplier — comets are temporary so we prioritise them slightly
COMET_ROI_BONUS: float = 1.5

# Never spend more than this fraction of a planet's garrison on one attack
MAX_COMMIT_FRACTION: float = 0.8


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

def fleet_speed(num_ships: int) -> float:
    """Speed of a fleet of `num_ships`."""
    num_ships = min(max(num_ships, 1), 1000)  # clamp to 1-1000
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(num_ships) / math.log(1000.0)) ** 1.5


def travel_time(src: Planet, tgt: Planet, num_ships: int) -> float:
    """Turns required for a fleet of `num_ships` to fly from src to tgt."""
    dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
    speed = fleet_speed(num_ships)
    return dist / speed if speed > 0 else float('inf')


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


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------

def score_target(src: Planet, tgt: Planet, comet_ids: set[int]) -> float | None:
    """
    Compute a scalar ROI score for attacking `tgt` from `src`.
    Returns None when the attack should be skipped entirely.

    Higher is better.  Score = production / ships_needed  (× comet bonus).
    """
    dist = distance(src, tgt)
    if dist == 0:
        return None

    if point_to_segment_distance((CENTER, CENTER), (src.x, src.y), (tgt.x, tgt.y)) < SUN_RADIUS + 1:
        return None

    fleet_size = src.ships // 2
    if fleet_size < 1:
        return None

    tt = dist / fleet_speed(fleet_size)
    garrison = predicted_garrison(tgt, tt)
    ships_needed = int(garrison) + 1

    if ships_needed > src.ships * MAX_COMMIT_FRACTION:
        return None

    # todo: add distance penalty
    roi = tgt.production / ships_needed if ships_needed > 0 else 0.0
    if tgt.id in comet_ids:
        roi *= COMET_ROI_BONUS

    return roi


def best_target(planets: list[Planet], src: Planet,
                player: int, assigned: set[int],
                comet_ids: set[int]) -> Planet | None:
    """
    Return the highest-ROI planet that `src` should attack, or None.

    Planets already in `assigned` are skipped to avoid double-sending.
    """
    best_score = -float('inf')
    best: Planet | None = None

    for tgt in planets:
        if tgt.owner == player or tgt.id in assigned:
            continue
        roi = score_target(src, tgt, comet_ids)
        if roi is not None and roi > best_score:
            best_score = roi
            best = tgt

    return best
