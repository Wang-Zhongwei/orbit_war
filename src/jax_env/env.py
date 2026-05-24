"""
OrbitWarsEnv — JAX-accelerated single-environment wrapper.

Same public interface as the old Python-physics version:
    env = OrbitWarsEnv(cfg, opponent, env_index=i)
    batch = env.reset(seed=42)          # -> TurnBatch
    result = env.step(player_action)    # -> StepResult

Internally the game runs on JIT-compiled JAX arrays; observations are
decoded to the existing dict/GameState format that encode_turn expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from ..config import TrainConfig
from ..features import TurnBatch, encode_turn
from ..agents.rule_based.base import OpponentPolicy
from .jax_physics import (
    JaxState, N_COMET_GROUPS, COMET_PER_GROUP, COMET_SLOT_BASE,
    MAX_PLANETS, MAX_FLEETS,
    init_jax_state, jax_step_jit, encode_joint_actions,
)


@dataclass(slots=True)
class StepResult:
    batch: TurnBatch
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsEnv:
    """Single-game environment backed by JAX-accelerated physics.

    The interface is identical to the original so that ``SubprocVecEnv``
    and the training loop work without any changes.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,  # accepted but unused — API compat
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.env_index = env_index
        self._state: JaxState | None = None
        self.last_obs: dict | None = None
        self.last_opp_obs: dict | None = None
        self.episode_index: int = 0
        self.learner_player: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> TurnBatch:
        """Start a new episode; return the initial observation as a TurnBatch."""
        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0

        self._state = init_jax_state(seed, num_agents=2)

        # Replicate kaggle's init-step: one physics tick before first observation.
        empty = encode_joint_actions([[], []])
        self._state, _, _ = jax_step_jit(self._state, empty)

        self.last_obs = _make_observation(self._state, self.learner_player)
        self.last_opp_obs = _make_observation(self._state, 1 - self.learner_player)
        self.episode_index += 1
        return encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        """Advance the game by one tick and return a StepResult."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        opponent_action = self.opponent(self.last_opp_obs)
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]

        jacts = encode_joint_actions(joint_action)
        self._state, statuses_arr, rewards_arr = jax_step_jit(self._state, jacts)

        # Convert JAX arrays to Python scalars
        statuses = ["DONE" if int(statuses_arr[i]) else "ACTIVE" for i in range(2)]
        rewards: list[float | None] = [
            float(rewards_arr[i]) if statuses[i] == "DONE" else None
            for i in range(2)
        ]

        self.last_obs = _make_observation(self._state, self.learner_player)
        self.last_opp_obs = _make_observation(self._state, 1 - self.learner_player)

        done = statuses[self.learner_player] == "DONE"
        reward = _terminal_reward(rewards, self.learner_player) if done else 0.0
        batch = encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)
        info = {
            "learner_player": self.learner_player,
            "player_status": statuses[self.learner_player],
            "opponent_status": statuses[1 - self.learner_player],
            "reward": reward,
        }
        return StepResult(batch=batch, reward=reward, done=done, info=info)


# ---------------------------------------------------------------------------
# Fast observation builder — avoids full GameState / Python loop conversion
# ---------------------------------------------------------------------------

def _make_observation(state: JaxState, player: int) -> dict:
    """Build a per-player observation dict from a JaxState.

    Pulls JAX arrays to host in one call and constructs the same schema
    that parse_observation / encode_turn expect.
    """
    # Pull entire state to host once (one device→host transfer)
    alive_p   = np.asarray(state.p_alive)
    p_id      = np.asarray(state.p_id)
    p_owner   = np.asarray(state.p_owner)
    p_x       = np.asarray(state.p_x)
    p_y       = np.asarray(state.p_y)
    p_radius  = np.asarray(state.p_radius)
    p_ships   = np.asarray(state.p_ships)
    p_prod    = np.asarray(state.p_production)
    p_init_x  = np.asarray(state.p_init_x)
    p_init_y  = np.asarray(state.p_init_y)
    p_is_comet= np.asarray(state.p_is_comet)

    alive_f   = np.asarray(state.f_alive)
    f_owner   = np.asarray(state.f_owner)
    f_x       = np.asarray(state.f_x)
    f_y       = np.asarray(state.f_y)
    f_angle   = np.asarray(state.f_angle)
    f_ships   = np.asarray(state.f_ships)

    cg_idx    = np.asarray(state.comet_path_idx)
    cg_paths  = np.asarray(state.comet_paths)
    cg_lens   = np.asarray(state.comet_path_lens)

    step_val  = int(state.step)
    av        = float(state.angular_velocity)

    # ---- Planets ----
    idxs = np.where(alive_p)[0]
    planets = [
        [int(p_id[i]), int(p_owner[i]), float(p_x[i]), float(p_y[i]),
         float(p_radius[i]), int(round(float(p_ships[i]))), int(round(float(p_prod[i])))]
        for i in idxs
    ]

    # ---- Initial planets (for encode_turn orbital velocity feature) ----
    initial_planets = [
        [int(p_id[i]), int(p_owner[i]), float(p_init_x[i]), float(p_init_y[i]),
         float(p_radius[i]), int(round(float(p_ships[i]))), int(round(float(p_prod[i])))]
        for i in idxs
    ]

    # ---- Fleets ----
    fidxs = np.where(alive_f)[0]
    fleets = [
        [int(fid), int(f_owner[i]), float(f_x[i]), float(f_y[i]),
         float(f_angle[i]), -1, int(round(float(f_ships[i])))]
        for fid, i in enumerate(fidxs)
    ]

    # ---- Comet groups ----
    comet_planet_ids: list[int] = []
    comets: list[dict] = []
    for gi in range(N_COMET_GROUPS):
        idx = int(cg_idx[gi])
        if idx < -1:        # -2 = never spawned
            continue
        slot_base = COMET_SLOT_BASE + gi * COMET_PER_GROUP
        pids = []
        paths = []
        for sj in range(COMET_PER_GROUP):
            slot = slot_base + sj
            if not alive_p[slot]:
                continue
            pids.append(int(p_id[slot]))
            comet_planet_ids.append(int(p_id[slot]))
            L    = int(cg_lens[gi, sj])
            path = cg_paths[gi, sj, :L].tolist()
            paths.append(path)
        if pids:
            comets.append({
                "planet_ids": pids,
                "paths": paths,
                "path_index": max(idx, 0),   # parse_observation does path[idx+1:]
            })

    return {
        "planets":         planets,
        "fleets":          fleets,
        "player":          player,
        "step":            step_val,
        "angular_velocity": av,
        "initial_planets": initial_planets,
        "comets":          comets,
        "comet_planet_ids": comet_planet_ids,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terminal_reward(rewards: list[float | None], player: int) -> float:
    r     = rewards[player]
    opp_r = rewards[1 - player]
    if r is None:
        return 0.0
    if r > 0.0 and opp_r is not None and opp_r > 0.0:
        return 0.0   # draw → 0
    return float(r)
