"""
Orbit Wars game physics — a faithful Python port of the kaggle interpreter.

Public API
----------
init_game(seed, num_agents) -> GameState
    Initialise a new episode.

step_game(state, actions, ...) -> (statuses, rewards)
    Advance the game by one tick in-place.
    Returns per-agent status ("ACTIVE" | "DONE") and reward (None until done).

All internal helpers are prefixed with ``_`` and not exported.
"""

from __future__ import annotations

import math
import random as _stdlib_random
from typing import Sequence

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    EPISODE_STEPS,
    MAX_SHIP_SPEED,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    point_to_segment_distance,
    swept_pair_hit,
)
from .generation import generate_comet_paths, generate_planets
from .state import GameState


# ---------------------------------------------------------------------------
# Public: episode initialisation
# ---------------------------------------------------------------------------

def init_game(seed: int | None = None, num_agents: int = 2) -> GameState:
    """Create a fresh GameState for a new episode.

    Planet positions are at their step-0 (initial) locations.  No production
    or movement has been applied.
    """
    if seed is None:
        seed = _stdlib_random.randrange(2 ** 31)
    rng = _stdlib_random.Random(seed)

    angular_velocity = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    initial_planets = [p[:] for p in planets]

    # Assign home planets from a randomly chosen symmetric group.
    num_groups = len(planets) // 4
    if num_groups > 0:
        home_group = rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_agents == 2:
            planets[base][1] = 0       # Q1 → player 0
            planets[base][5] = 10
            planets[base + 3][1] = 1   # Q4 → player 1
            planets[base + 3][5] = 10
        elif num_agents == 4:
            for j in range(4):
                planets[base + j][1] = j
                planets[base + j][5] = 10

    return GameState(
        planets=planets,
        fleets=[],
        next_fleet_id=0,
        step=0,
        angular_velocity=angular_velocity,
        initial_planets=initial_planets,
        comets=[],
        comet_planet_ids=[],
        num_agents=num_agents,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Public: single game tick
# ---------------------------------------------------------------------------

def step_game(
    state: GameState,
    actions: Sequence[list],
    *,
    ship_speed: float = MAX_SHIP_SPEED,
    episode_steps: int = EPISODE_STEPS,
    comet_speed: float = COMET_SPEED,
) -> tuple[list[str], list[float | None]]:
    """Advance *state* by one tick **in-place**.

    Parameters
    ----------
    state:
        Mutable game state; updated in-place to reflect the tick result.
    actions:
        One action list per agent.  Each action is a list of moves, where
        each move is ``[from_planet_id, angle_radians, num_ships]``.
    ship_speed, episode_steps, comet_speed:
        Configuration knobs that mirror the kaggle environment configuration.

    Returns
    -------
    statuses : list[str]
        ``"ACTIVE"`` or ``"DONE"`` for each agent.
    rewards : list[float | None]
        ``None`` while active; ``+1.0`` (win) or ``-1.0`` (lose) when done.
        Both players get ``+1.0`` on a draw (matched by ``terminal_reward``
        in env.py which converts that to 0).
    """
    # Step 0a: remove comets whose path was exhausted before fleet launch
    # (matches kaggle ordering: expired comets are invisible to agents this tick)
    _remove_expired_comets(state)

    # Step 0b: spawn new comet group when the *next* completed step would
    # hit a designated spawn step.  The comet is placed off-board this tick
    # and moved to path[0] during fleet-movement.
    _maybe_spawn_comets(state, next_step=state.step + 1, comet_speed=comet_speed)

    # Step 1: fleet launch — validate and deduct ships, create new fleet rows
    for player_id, action in enumerate(actions):
        _process_moves(state, player_id, action)

    # Step 2: production — owned planets gain ships
    for planet in state.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    # Step 3: compute old→new planet positions for continuous collision checks
    # Inner planets rotate; comets follow their waypoint paths.
    planet_paths = _compute_planet_paths(state)

    # Step 4: move fleets one tick; collect those that hit planets / sun / OOB
    fleets_to_remove, combat_lists = _move_fleets(state, planet_paths, ship_speed)

    # Step 5: apply planet position update (rotation + comet advancement)
    for planet in state.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]  # new_pos

    # Remove comets that exhausted their path this tick (after position update)
    _remove_freshly_expired_comets(state)

    # Remove fleets consumed by collisions
    hit_ids = {id(f) for f in fleets_to_remove}
    state.fleets = [f for f in state.fleets if id(f) not in hit_ids]

    # Step 6: combat resolution at each contested planet
    _resolve_combat(state, combat_lists)

    # Advance the step counter
    state.step += 1

    return _check_termination(state, episode_steps)


# ---------------------------------------------------------------------------
# Internal helpers — comet lifecycle
# ---------------------------------------------------------------------------

def _remove_expired_comets(state: GameState) -> None:
    """Purge comet planets that have already reached the end of their path."""
    expired: list[int] = []
    for group in state.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired.append(pid)
    if expired:
        _purge_comet_planets(state, set(expired))


def _remove_freshly_expired_comets(state: GameState) -> None:
    """Purge comets whose path_index was just advanced past the end this tick."""
    expired: list[int] = []
    for group in state.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired.append(pid)
    if expired:
        _purge_comet_planets(state, set(expired))


def _purge_comet_planets(state: GameState, expired_set: set[int]) -> None:
    state.planets = [p for p in state.planets if p[0] not in expired_set]
    state.initial_planets = [p for p in state.initial_planets if p[0] not in expired_set]
    state.comet_planet_ids = [pid for pid in state.comet_planet_ids if pid not in expired_set]
    for group in state.comets:
        group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
    state.comets = [g for g in state.comets if g["planet_ids"]]


def _maybe_spawn_comets(state: GameState, next_step: int, comet_speed: float) -> None:
    """Spawn a comet group when *next_step* is a designated spawn step.

    Uses a deterministic RNG seeded from the episode seed + spawn step so
    comet shapes are reproducible — identical to the kaggle implementation.
    """
    if next_step not in COMET_SPAWN_STEPS:
        return
    comet_rng = _stdlib_random.Random(f"orbit_wars-comet-{state.seed}-{next_step}")
    paths = generate_comet_paths(
        state.initial_planets,
        state.angular_velocity,
        next_step,
        state.comet_planet_ids,
        comet_speed,
        rng=comet_rng,
    )
    if paths is None:
        return

    next_id = (max(p[0] for p in state.planets) + 1) if state.planets else 0
    comet_ships = min(
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
    )
    group: dict = {"planet_ids": [], "paths": paths, "path_index": -1}
    for i in range(4):
        pid = next_id + i
        group["planet_ids"].append(pid)
        state.comet_planet_ids.append(pid)
        # Planet starts off-board; first _compute_planet_paths call places it at path[0]
        planet = [pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
        state.planets.append(planet)
        state.initial_planets.append(planet[:])
    state.comets.append(group)


# ---------------------------------------------------------------------------
# Internal helpers — per-tick physics
# ---------------------------------------------------------------------------

def _process_moves(state: GameState, player_id: int, action: list) -> None:
    """Validate and launch fleets for one player."""
    if not action or not isinstance(action, list):
        return
    for move in action:
        if len(move) != 3:
            continue
        from_id, angle, ships = move
        ships = int(ships)
        from_planet = next((p for p in state.planets if p[0] == from_id), None)
        if from_planet is None or from_planet[1] != player_id:
            continue
        if ships <= 0 or from_planet[5] < ships:
            continue
        from_planet[5] -= ships
        # Start just outside the planet surface so it doesn't collide with origin
        start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
        start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
        state.fleets.append([
            state.next_fleet_id, player_id,
            start_x, start_y,
            float(angle), from_id, ships,
        ])
        state.next_fleet_id += 1


def _compute_planet_paths(state: GameState) -> dict[int, tuple]:
    """Return ``{planet_id: (old_pos, new_pos, check_collision)}`` for this tick.

    ``check_collision=False`` on the first placement of a comet (its old_pos
    is the off-board placeholder), matching the kaggle safety check.
    """
    comet_pid_set = set(state.comet_planet_ids)
    initial_by_id = {p[0]: p for p in state.initial_planets}
    planet_paths: dict[int, tuple] = {}
    # Kaggle computes target positions using the CURRENT obs.step value, which
    # equals state.step (the number of completed ticks).  For the first real
    # tick (state.step=1) this gives angular_velocity*1; for the init step
    # (state.step=0) this gives angular_velocity*0 — i.e. no rotation, which
    # matches kaggle's observed init-step behaviour.
    cur_step = state.step

    # Regular (non-comet) planets
    for planet in state.planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        new_pos = old_pos
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            orbital_r = math.sqrt(dx * dx + dy * dy)
            if orbital_r + planet[4] < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + state.angular_velocity * cur_step
                new_pos = (
                    CENTER + orbital_r * math.cos(cur_angle),
                    CENTER + orbital_r * math.sin(cur_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # Comet planets — advance one step along their pre-computed waypoints
    for group in state.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in state.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                # Expired this tick — stays put; will be pruned after combat
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                new_pos = (p_path[idx][0], p_path[idx][1])
                # First placement: old_pos is the off-board placeholder → skip collision
                check = old_pos[0] >= 0.0
                planet_paths[pid] = (old_pos, new_pos, check)

    return planet_paths


def _fleet_speed(ships: int, max_speed: float) -> float:
    ships = max(ships, 1)
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _move_fleets(
    state: GameState,
    planet_paths: dict[int, tuple],
    max_speed: float,
) -> tuple[list, dict[int, list]]:
    """Move all fleets one tick; return (fleets_to_remove, combat_lists).

    Continuous swept-pair collision detection (fleet × planet) runs for every
    fleet–planet pair.  A fleet is removed if it hits a planet, goes out of
    bounds, or crosses the sun.
    """
    fleets_to_remove: list = []
    combat_lists: dict[int, list] = {p[0]: [] for p in state.planets}

    for fleet in state.fleets:
        angle = fleet[4]
        speed = _fleet_speed(int(fleet[6]), max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # Check planet collisions first (continuous swept-pair test)
        hit = False
        for planet in state.planets:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:   # path[2] = check_collision flag
                continue
            p_old, p_new, _ = path
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit = True
                break
        if hit:
            continue

        # Out of board bounds
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue

        # Sun collision (continuous segment check)
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)

    return fleets_to_remove, combat_lists


def _resolve_combat(state: GameState, combat_lists: dict[int, list]) -> None:
    """Resolve fleet combat at each planet.

    Ships from each player are summed; the top player beats the second by the
    difference.  Ties leave the planet owner unchanged with 0 ships.
    """
    for pid, planet_fleets in combat_lists.items():
        if not planet_fleets:
            continue
        planet = next((p for p in state.planets if p[0] == pid), None)
        if planet is None:
            continue

        player_ships: dict[int, int] = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

        sorted_players = sorted(player_ships.items(), key=lambda kv: kv[1], reverse=True)
        top_owner, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            survivor_owner = top_owner if survivor_ships > 0 else -1
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships

        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])


def _check_termination(
    state: GameState,
    episode_steps: int,
) -> tuple[list[str], list[float | None]]:
    """Determine whether the episode has ended and compute final rewards."""
    # Terminates when the step count has reached the episode limit, or when at
    # most one player still controls planets / has fleets in flight.
    terminated = state.step >= episode_steps - 1

    if not terminated:
        alive: set[int] = set()
        for p in state.planets:
            if p[1] != -1:
                alive.add(p[1])
        for f in state.fleets:
            alive.add(f[1])
        if len(alive) <= 1:
            terminated = True

    if not terminated:
        return ["ACTIVE"] * state.num_agents, [None] * state.num_agents

    # Tally ships controlled by each player
    scores = [0] * state.num_agents
    for p in state.planets:
        if 0 <= p[1] < state.num_agents:
            scores[p[1]] += p[5]
    for f in state.fleets:
        if 0 <= f[1] < state.num_agents:
            scores[f[1]] += f[6]

    max_score = max(scores) if scores else 0
    rewards: list[float] = []
    for i in range(state.num_agents):
        # Both win (+1) on a draw; env.py's terminal_reward converts both-positive to 0
        rewards.append(1.0 if (scores[i] == max_score and max_score > 0) else -1.0)

    return ["DONE"] * state.num_agents, rewards
