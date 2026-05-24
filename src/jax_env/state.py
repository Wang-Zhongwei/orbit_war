"""
GameState: mutable game state for one Orbit Wars episode.

The observation dict returned by ``make_observation`` mirrors the schema that
kaggle_environments exposed — downstream code (parse_observation, helper.py)
reads the same keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameState:
    """Single source of truth for all game physics.  Mutated in-place by ``step_game``."""

    # Each planet row: [id, owner, x, y, radius, ships, production]
    planets: list[list]

    # Each fleet row: [id, owner, x, y, angle, from_planet_id, ships]
    fleets: list[list]

    next_fleet_id: int
    step: int                   # completed ticks (0 after init_game, N after N calls to step_game)
    angular_velocity: float

    # Planet positions at step=0, used to derive orbital angles.
    # Frozen: never mutated after init_game.
    initial_planets: list[list]

    # Comet groups: [{"planet_ids": [...], "paths": [...], "path_index": int}]
    comets: list[dict]

    # Flat list of active comet planet IDs (duplicate for fast lookup)
    comet_planet_ids: list[int]

    num_agents: int = 2

    # Episode seed — used to reproduce comet RNG; never exposed to agents
    seed: int | None = None


def make_observation(state: GameState, player: int) -> dict:
    """Build a per-player observation dict.

    The returned dict has the same keys that kaggle's observation object
    exposed, so ``parse_observation`` (game_types.py) and ``unpack_obs``
    (helper.py) work unchanged.
    """
    return {
        "planets": state.planets,
        "fleets": state.fleets,
        "player": player,
        "step": state.step,
        "angular_velocity": state.angular_velocity,
        "initial_planets": state.initial_planets,
        "comets": state.comets,
        "comet_planet_ids": state.comet_planet_ids,
    }
