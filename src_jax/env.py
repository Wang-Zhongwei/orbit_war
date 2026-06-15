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


def make_board_pool(n_boards: int, seed: int, num_agents: int = 2) -> EnvState:
    """Return a stacked JaxState[n_boards, ...] of pre-generated boards.

    num_agents seats are assigned home planets at generation time (2 = diagonal
    1v1, 4 = one seat per home-planet group for free-for-all).
    """
    print(f"Generating {n_boards} board layouts ({num_agents}-player) …", flush=True)
    states = [_init_one(seed + i, num_agents) for i in range(n_boards)]
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

    When alternate_sides, the learner's seat is spread across all num_agents
    positions so training sees every seat equally (round-robin over envs).
    """
    indices = jnp.arange(n_envs, dtype=jnp.int32)
    batched = select_boards(board_pool, indices, n_boards)
    ep_reward = jnp.zeros(n_envs, dtype=jnp.float32)
    if alternate_sides:
        learner_players = (jnp.arange(n_envs) % num_agents).astype(jnp.int32)
    else:
        learner_players = jnp.zeros(n_envs, dtype=jnp.int32)
    return batched, indices, ep_reward, learner_players
