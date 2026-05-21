"""
stateless_agents.py — rule-based agents sharing a common configurable pipeline.

All three canonical agents (sniper, roi, never_miss) are thin wrappers over
``run_agent``, which executes this shared decision pipeline each turn:

    1. Parse obs → ``GameContext``
    2. Order source planets  (``AgentConfig.source_sort``)
    3. For each source planet:
        a. Select one target, skipping already-assigned and comet planets.
        b. Mark target as assigned (prevents two sources hitting the same planet).
        c. Call ``aim_for_planet`` to compute angle, ship count, and ETA.
        d. Accept when capturable, or when source is a comet and
           ``AgentConfig.accept_comet_src`` is True.
        e. Record the move with its cost scalar.
    4. Sort candidates by cost (ascending) and cap at ``AgentConfig.max_moves``.

Adding a new agent
------------------
Define an ``AgentConfig`` preset and a 1-arg wrapper::

    MY_CONFIG = AgentConfig(target_mode="nearest", max_moves=4)

    def my_agent(obs) -> list:
        return run_agent(obs, MY_CONFIG)

To make it tunable, register it in ``tune.py``::

    REGISTRY.register("my_agent", TunableStrategy(
        factory=lambda cfg: (lambda obs: run_agent(obs, AgentConfig(**cfg))),
        params=[CategoricalParam("target_mode", ["nearest", "roi"]), ...],
        agent_fn_name="run_agent",
        description="my_agent: ...",
    ))
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, SUN_RADIUS, Planet, point_to_segment_distance

from src.config import AgentConfig
from src.game_context import GameContext
from src.helper import (
    aim_for_planet,
    distance,
    fleet_speed,
    predicted_garrison,
)

# ---------------------------------------------------------------------------
# Pre-built configs for the three canonical agents
# ---------------------------------------------------------------------------

SNIPER_CONFIG = AgentConfig(
    source_sort="none",
    target_mode="nearest",
    accept_comet_src=False,
    cost_mode="none",
    max_moves=None,
)

ROI_CONFIG = AgentConfig(
    source_sort="richest_first",
    target_mode="roi",
    accept_comet_src=False,
    cost_mode="none",
    max_moves=None,
)

NEVER_MISS_DEFAULT_CONFIG = AgentConfig(
    source_sort="none",
    target_mode="random_biased",
    limit_top_k_fraction=0.12,
    bias_draws=1,
    accept_comet_src=True,
    cost_mode="turns_weighted",
    cost_weight=2.0,
    max_moves=10,
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _sort_sources(planets: list[Planet], mode: str) -> list[Planet]:
    if mode == "none":
        return planets
    if mode == "richest_first":
        return sorted(planets, key=lambda p: p.ships, reverse=True)
    raise ValueError(f"Unknown source_sort: {mode!r}")


def _select_target(
    src: Planet,
    ctx: GameContext,
    config: AgentConfig,
    assigned: set[int],
) -> Planet | None:
    if config.target_mode == "roi":
        # best_target handles assigned filtering and ROI scoring internally.
        # It may return a comet; the caller checks ctx.comet_ids afterward.
        return best_target(ctx.planets, src, ctx.player, assigned, ctx.comet_ids)

    # Explicit pool: enemies not yet assigned, comets excluded up front.
    pool = [
        p for p in ctx.targets
        if p.id not in assigned and p.id not in ctx.comet_ids
    ]
    if not pool:
        return None

    # Sort ascending by squared distance (avoids sqrt, monotone with distance).
    pool.sort(key=lambda t: (src.x - t.x) ** 2 + (src.y - t.y) ** 2)

    if config.target_mode == "nearest":
        return pool[0]

    if config.target_mode == "random_biased":
        k = max(1, int(len(pool) * config.limit_top_k_fraction))
        pool = pool[:k]
        n = len(pool)
        # min of ``bias_draws`` uniform samples → closer indices more likely.
        idx = min(random.randint(0, n - 1) for _ in range(max(1, config.bias_draws)))
        return pool[idx]

    raise ValueError(f"Unknown target_mode: {config.target_mode!r}")


def _compute_cost(turns_to_arrive: int, can_capture: bool, config: AgentConfig) -> float:
    if config.cost_mode == "none":
        return 0.0
    if config.cost_mode == "turns_weighted":
        return turns_to_arrive * (config.cost_weight - float(can_capture))
    raise ValueError(f"Unknown cost_mode: {config.cost_mode!r}")


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------
# Comet ROI multiplier — comets are temporary so we prioritise them slightly
COMET_ROI_BONUS: float = 1.5

# Never spend more than this fraction of a planet's garrison on one attack
MAX_COMMIT_FRACTION: float = 0.8
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


# ---------------------------------------------------------------------------
# Shared agent pipeline
# ---------------------------------------------------------------------------

def run_agent(obs, config: AgentConfig) -> list[list]:
    """Execute the shared decision pipeline and return a list of moves.

    Pipeline steps:
        1. Parse ``obs`` into a ``GameContext``.
        2. Order source planets by ``config.source_sort``.
        3. For each source:
           - Select one target (assigned + comet exclusion).
           - Mark target assigned (prevents duplicate attacks).
           - Run ``aim_for_planet`` for angle, ship count, and ETA.
           - Accept when capturable, or source is comet + ``accept_comet_src``.
           - Record ``(move, cost)`` pair.
        4. Sort by cost, cap at ``config.max_moves``, return move triples.

    Args:
        obs: Raw observation dict from the Kaggle Environments environment.
        config: Micro-decision configuration controlling every branch.

    Returns:
        List of ``[from_planet_id, angle_rad, num_ships]`` triples.
    """
    ctx = GameContext.from_obs(obs)
    if not ctx.my_planets or not ctx.targets:
        return []

    sources   = _sort_sources(ctx.my_planets, config.source_sort)
    assigned: set[int] = set()
    candidates: list[tuple[list, float]] = []

    for src in sources:
        target = _select_target(src, ctx, config, assigned)
        if target is None:
            continue

        # Consume the target slot regardless of aim success, so later source
        # planets don't pile onto the same unreachable or comet target.
        assigned.add(target.id)

        if target.id in ctx.comet_ids:
            continue  # roi mode may return comets; skip them here

        result = aim_for_planet(target, src, ctx.angular_velocity)
        if result is None:
            continue

        angle, ships, can_capture, turns_to_arrive = result
        my_is_comet = src.id in ctx.comet_ids

        if not can_capture and not (config.accept_comet_src and my_is_comet):
            continue

        cost = _compute_cost(turns_to_arrive, can_capture, config)
        candidates.append(([src.id, angle, ships], cost))

    candidates.sort(key=lambda x: x[1])
    cap = config.max_moves if config.max_moves is not None else len(candidates)
    return [move for move, _ in candidates[:cap]]


# ---------------------------------------------------------------------------
# Named agent wrappers — backward-compatible 1-arg entry points
# ---------------------------------------------------------------------------

def nearest_planet_sniper(obs) -> list:
    """Sniper agent: attacks the closest enemy from each owned planet."""
    return run_agent(obs, SNIPER_CONFIG)


def roi_agent(obs) -> list:
    """ROI agent: richest-first source ordering, production/cost target scoring."""
    return run_agent(obs, ROI_CONFIG)


def never_miss_agent(obs) -> list:
    """Distance-biased random attacker, fully configurable via ``AgentConfig``."""
    return run_agent(obs, NEVER_MISS_DEFAULT_CONFIG)
