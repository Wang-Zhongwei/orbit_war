"""
Board pool and initial state management for the fully-JAX training loop.

Planet generation uses Python rejection sampling (can't JIT), so we pre-generate
a pool of N boards at startup and cycle through them on episode resets.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics import (
    EnvState,
    init_jax_state,
    jax_step_jit,
    MAX_ACTIONS_PA,
)

_EMPTY_ACTIONS = jnp.full((4, MAX_ACTIONS_PA, 3), -1.0, dtype=jnp.float32)


def _init_one(seed: int, num_agents: int = 2) -> EnvState:
    """Generate one board and apply the kaggle init-tick."""
    state = init_jax_state(seed, num_agents=num_agents)
    state, _, _ = jax_step_jit(state, _EMPTY_ACTIONS)
    return state


def _stack_states(states: list[EnvState]) -> EnvState:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _board_seat_counts(n_boards: int, num_agents: int, mix_4player_frac: float) -> list[int]:
    """Per-board player counts. Uniform `num_agents` unless mixing is requested
    (num_agents == 4 and frac < 1), in which case a `frac` share are 4-player and
    the rest 2-player, evenly interleaved (deterministic, reproducible)."""
    if num_agents != 4 or mix_4player_frac >= 1.0:
        return [num_agents] * n_boards
    assert 0.0 <= mix_4player_frac < 1.0, mix_4player_frac
    # Bresenham-style even spread: board i is 4-player when the running count of
    # 4-player boards ticks over at i. Gives a balanced interleave for any frac.
    counts = []
    for i in range(n_boards):
        is_4p = int((i + 1) * mix_4player_frac) > int(i * mix_4player_frac)
        counts.append(4 if is_4p else 2)
    return counts


def make_board_pool(
    n_boards: int, seed: int, num_agents: int = 2, mix_4player_frac: float = 1.0,
) -> EnvState:
    """Return a stacked JaxState[n_boards, ...] of pre-generated boards.

    num_agents seats are assigned home planets at generation time (2 = diagonal
    1v1, 4 = one seat per home-planet group for free-for-all). When num_agents == 4
    and mix_4player_frac < 1, a fraction of boards are generated as 2-player instead
    (a mixed 1v1/FFA curriculum); each board carries its own num_agents in the
    EnvState so the rollout and physics handle them per-env.
    """
    seats = _board_seat_counts(n_boards, num_agents, mix_4player_frac)
    n4 = sum(1 for s in seats if s == 4)
    print(f"Generating {n_boards} board layouts "
          f"({n4} four-player, {n_boards - n4} two-player) …", flush=True)
    states = [_init_one(seed + i, na) for i, na in enumerate(seats)]
    return _stack_states(states)


def select_boards(board_pool: EnvState, indices: jnp.ndarray, n_boards: int) -> EnvState:
    """Index board_pool with [N] indices → JaxState[N, ...]."""
    safe = indices % n_boards
    return jax.tree_util.tree_map(lambda x: x[safe], board_pool)


def make_initial_carry_state(
    board_pool: EnvState,
    n_boards: int,
    n_envs: int,
    seed: int,
    alternate_sides: bool,
    num_agents: int = 2,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (batched_state, board_indices, ep_rewards, learner_players).

    When alternate_sides, the learner's seat is spread across all seats so training
    sees every seat equally — derived per-env from each board's own num_agents so
    mixed 2p/4p pools never assign the learner to a phantom seat. The num_agents
    argument is unused (kept for API compatibility); the per-board value is used.
    """
    indices = jnp.arange(n_envs, dtype=jnp.int32)
    batched = select_boards(board_pool, indices, n_boards)
    ep_reward = jnp.zeros(n_envs, dtype=jnp.float32)
    if alternate_sides:
        learner_players = (jnp.arange(n_envs, dtype=jnp.int32) % batched.num_agents).astype(jnp.int32)
    else:
        learner_players = jnp.zeros(n_envs, dtype=jnp.int32)
    return batched, indices, ep_reward, learner_players
