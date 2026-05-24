"""
src/env.py — thin re-export shim.

All environment logic now lives in src/jax_env/.  This module re-exports
OrbitWarsEnv and StepResult so that existing imports from src.env continue to
work without changes.

Legacy helper functions (extract_observation, extract_status, extract_reward,
terminal_reward, default_make_fn) are kept here for any code that may still
reference them directly.
"""

from __future__ import annotations

from typing import Any

from .jax_env.env import OrbitWarsEnv, StepResult  # noqa: F401


# ---------------------------------------------------------------------------
# Legacy helpers — kept for backward compatibility; no longer used internally
# ---------------------------------------------------------------------------

def default_make_fn() -> Any:
    from kaggle_environments import make
    return make


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def terminal_reward(player_state: Any, opp_state: Any) -> float:
    player_reward = extract_reward(player_state)
    opponent_reward = extract_reward(opp_state)
    if player_reward > 0.0 and opponent_reward > 0.0:
        return 0.0
    return player_reward
