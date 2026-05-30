"""
Pure-JAX stateless greedy heuristic for use as a training opponent.

Mirrors the scoring formula of StatefulAgent (stateful_v2.py) but is
entirely stateless — no cross-turn fleet tracking, coalition attacks, or
defense.  It is vmappable over a batch of EnvState so it fits directly
inside the JAX rollout scan.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics import EnvState, MAX_PLANETS, MAX_ACTIONS_PA, _fleet_speed
from .features import aim_fleets

# Mirror StatefulAgentConfig defaults from stateful_v2.py
_PROD_MULT      = 10.0
_ENEMY_BONUS    = 14.61
_SHIPS_PENALTY  = 0.482
_TIME_PENALTY   = 1.269
_COMET_PENALTY  = 1.52
_MIN_SHIPS      = 13       # int, matches stateful_v2.min_ships_attack
_COMMIT_FRAC    = 0.6      # fraction of garrison to send per fleet


def heuristic_actions(state: EnvState, player: jnp.ndarray) -> jnp.ndarray:
    """Greedy heuristic: score each (src, tgt) pair and attack the best target.

    Pure JAX, no side effects, vmappable over a leading batch axis.
    Returns float32[MAX_ACTIONS_PA, 3]  (planet_id, angle, ships).
    """
    # ── Ownership masks ──────────────────────────────────────────────────────
    src_mask = (
        state.p_alive
        & (state.p_owner == player)
        & (state.p_ships >= _MIN_SHIPS)
    )  # [MAX_P]
    tgt_mask = state.p_alive & (state.p_owner != player)  # [MAX_P]

    # ── Pairwise score matrix [MAX_P, MAX_P] ─────────────────────────────────
    # Speed is estimated from target garrison (matches stateful_v2._score).
    tgt_garrison = state.p_ships[None, :] + 1.0     # [1, MAX_P]
    speed_tgt    = _fleet_speed(tgt_garrison)        # [1, MAX_P]

    dx       = state.p_x[:, None] - state.p_x[None, :]   # [MAX_P, MAX_P]
    dy       = state.p_y[:, None] - state.p_y[None, :]
    dist     = jnp.sqrt(dx * dx + dy * dy + 1e-8)
    time_est = dist / (speed_tgt + 1e-8)                  # [MAX_P, MAX_P]

    prod  = state.p_production[None, :]                   # [1, MAX_P]
    enemy = (state.p_owner[None, :] != -1).astype(jnp.float32)
    comet = state.p_is_comet[None, :].astype(jnp.float32)

    score = (
        _PROD_MULT    * prod
        + _ENEMY_BONUS  * prod * enemy
        - _SHIPS_PENALTY * (tgt_garrison + time_est * prod * enemy)
        - _TIME_PENALTY  * time_est
        - _COMET_PENALTY * comet
    )  # [MAX_P, MAX_P]

    valid_pair = src_mask[:, None] & tgt_mask[None, :]
    score = jnp.where(valid_pair, score, -1e9)

    # ── Per-source: best target slot ─────────────────────────────────────────
    best_tgt_slot = jnp.argmax(score, axis=1).astype(jnp.int32)  # [MAX_P]
    any_valid_tgt = jnp.any(valid_pair, axis=1)                   # [MAX_P]

    # ── Ships to send (fraction of garrison, at least _MIN_SHIPS) ────────────
    ships_send = jnp.maximum(
        jnp.int32(_MIN_SHIPS),
        (_COMMIT_FRAC * state.p_ships).astype(jnp.int32),
    )  # [MAX_P]

    # ── Lead-corrected angles + sun-blocking via reusable aim_fleets ─────────
    src_slots = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    angles, is_blocked = aim_fleets(
        state.p_x, state.p_y, state.p_radius,
        state.p_alive, state.p_is_comet,
        state.angular_velocity,
        src_slots, best_tgt_slot, ships_send,
    )  # each [MAX_P]

    # ── Action validity ───────────────────────────────────────────────────────
    action_ok = src_mask & any_valid_tgt & ~is_blocked  # [MAX_P]

    # ── Select top-MAX_ACTIONS_PA sources by garrison descending ─────────────
    priority   = jnp.where(action_ok, state.p_ships, -1.0)
    sorted_idx = jnp.argsort(-priority)[:MAX_ACTIONS_PA]  # [MAX_ACTIONS_PA]

    sel_ids    = state.p_id[sorted_idx].astype(jnp.float32)
    sel_angles = angles[sorted_idx]
    sel_ships  = ships_send[sorted_idx].astype(jnp.float32)
    sel_ok     = action_ok[sorted_idx]

    rows = jnp.stack([sel_ids, sel_angles, sel_ships], axis=1)  # [MAX_ACTIONS_PA, 3]
    rows = jnp.where(sel_ok[:, None], rows, jnp.full((3,), -1.0))
    return rows
