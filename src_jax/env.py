"""
Board pool and initial state management for the fully-JAX training loop.

Planet generation uses Python rejection sampling (can't JIT), so we pre-generate
a pool of N boards at startup and cycle through them on episode resets.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_physics import (
    JaxState,
    init_jax_state,
    jax_step_jit,
    MAX_BASE_PLANETS,
    MAX_ACTIONS_PA,
)

_EMPTY_ACTIONS = jnp.full((4, MAX_ACTIONS_PA, 3), -1.0, dtype=jnp.float32)


def _init_one(seed: int) -> JaxState:
    """Generate one board and apply the kaggle init-tick."""
    state = init_jax_state(seed, num_agents=2)
    state, _, _ = jax_step_jit(state, _EMPTY_ACTIONS)
    return state


def _stack_states(states: list[JaxState]) -> JaxState:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def make_board_pool(n_boards: int, seed: int) -> JaxState:
    """Return a stacked JaxState[n_boards, ...] of pre-generated boards."""
    print(f"Generating {n_boards} board layouts …", flush=True)
    states = [_init_one(seed + i) for i in range(n_boards)]
    return _stack_states(states)


def select_boards(board_pool: JaxState, indices: jnp.ndarray, n_boards: int) -> JaxState:
    """Index board_pool with [N] indices → JaxState[N, ...]."""
    safe = indices % n_boards
    return jax.tree_util.tree_map(lambda x: x[safe], board_pool)


def make_initial_carry_state(
    board_pool: JaxState,
    n_boards: int,
    n_envs: int,
    seed: int,
    alternate_sides: bool,
) -> tuple[JaxState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (batched_state, board_indices, ep_rewards, learner_players)."""
    indices = jnp.arange(n_envs, dtype=jnp.int32)
    batched = select_boards(board_pool, indices, n_boards)
    ep_reward = jnp.zeros(n_envs, dtype=jnp.float32)
    if alternate_sides:
        learner_players = (jnp.arange(n_envs) % 2).astype(jnp.int32)
    else:
        learner_players = jnp.zeros(n_envs, dtype=jnp.int32)
    return batched, indices, ep_reward, learner_players
