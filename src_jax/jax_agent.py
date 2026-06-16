"""
JAX policy → kaggle-environment callable agent.

Usage:
    from src_jax.jax_agent import make_jax_agent

    agent = make_jax_agent("artifacts/bc/bc_pretrain/ckpt_last.pkl")
    env.run([agent, opponent])
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from .bc_data import obs_to_env_state
from .features import MIN_SOURCE_SHIPS, _make_env_encoder, self_feature_dim, candidate_feature_dim
from .feature_versions import make_env_encoder
from .physics import MAX_BASE_PLANETS, MAX_PLANETS, MAX_FLEETS, N_COMET_GROUPS, COMET_PER_GROUP, EnvState
from .policies import build_policy
from .rollout import GLOB_DIM


def _obs_to_dict(obs) -> dict:
    """Accept either a plain dict or a kaggle StructuredObservation object."""
    if isinstance(obs, dict):
        return obs
    try:
        return dict(obs)
    except TypeError:
        pass
    try:
        return vars(obs)
    except TypeError:
        pass
    keys = [
        "player", "step", "planets", "fleets", "initial_planets",
        "angular_velocity", "comet_planet_ids", "comets", "remainingOverageTime",
        "next_fleet_id",
    ]
    return {k: getattr(obs, k) for k in keys if hasattr(obs, k)}


def _safe_target_logits(target_logits: jax.Array) -> jax.Array:
    invalid = ~jnp.isfinite(target_logits).any(axis=-1)
    return jnp.where(invalid[:, None], jnp.zeros_like(target_logits), target_logits)


def build_agent(
    params,
    cfg,
    deterministic: bool = True,
) -> Callable:
    """Return an obs→moves callable from in-memory params."""
    params  = jax.tree_util.tree_map(jnp.asarray, params)
    rng_key = jax.random.PRNGKey(int(getattr(cfg, "seed", 0)))

    K = cfg.env.candidate_count

    module    = build_policy(cfg)
    encode_fn = make_env_encoder(K, cfg.env, getattr(cfg.model, "feat_version", "v7"))

    MAX_SRC = MAX_BASE_PLANETS

    @jax.jit
    def _forward(sf, cf, gf, cm, sv):
        return module.apply(params, sf, cf, gf, cm, sv)

    def agent(obs) -> list:
        nonlocal rng_key

        obs_dict = _obs_to_dict(obs)
        state  = obs_to_env_state(obs_dict)
        player = jnp.array([int(obs_dict.get("player", 0))], dtype=jnp.int32)

        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        enc     = encode_fn(batched, player)

        sf = enc.self_feats[0]               # [MAX_SRC, SELF_DIM]
        cf = enc.cand_feats[0]               # [MAX_SRC, K, CAND_DIM]
        gf = jnp.broadcast_to(enc.global_feat[0][None, :], (MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask[0]                # [MAX_SRC, K]
        sv = enc.src_valid[0]

        out = _forward(sf, cf, gf, cm, sv)

        target_logits = _safe_target_logits(out.target_logits)
        if deterministic:
            ti = jnp.argmax(target_logits, axis=-1)
        else:
            rng_key, subkey = jax.random.split(rng_key)
            ti = jax.random.categorical(subkey, target_logits)

        row_idx = jnp.arange(MAX_SRC)

        src_valid = enc.src_valid[0]
        src_ids   = enc.src_ids[0]
        src_ships = enc.src_ships[0]
        cand_mask = enc.cand_mask[0]
        angles    = enc.angles[0]            # [MAX_SRC, K] — exact, precomputed by encoder

        ships_send      = jnp.maximum(MIN_SOURCE_SHIPS, src_ships)
        selected_angles = angles[row_idx, ti]

        moves: list = []
        for i in range(MAX_SRC):
            if not bool(src_valid[i]):
                continue
            t = int(ti[i])
            if t == 0:
                continue
            if not bool(cand_mask[i, t]):
                continue
            moves.append([int(src_ids[i]), float(selected_angles[i]), int(ships_send[i])])

        return moves

    # Warm up both JITs (encode_fn and _forward) so first in-game call doesn't
    # blow the agentTimeout. encode_fn changed in the pool-K refactor and now
    # takes longer to trace; both need to compile before entering the env.
    from .physics import MAX_PATH_LEN
    _dummy_state = EnvState(
        p_id=jnp.zeros(MAX_PLANETS, jnp.int32),
        p_owner=jnp.full(MAX_PLANETS, -1, jnp.int32),
        p_x=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_y=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_radius=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_ships=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_production=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_alive=jnp.zeros(MAX_PLANETS, bool),
        p_init_x=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_init_y=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_is_comet=jnp.zeros(MAX_PLANETS, bool),
        f_owner=jnp.zeros(MAX_FLEETS, jnp.int32),
        f_x=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_y=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_angle=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_ships=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_alive=jnp.zeros(MAX_FLEETS, bool),
        comet_paths=jnp.zeros((N_COMET_GROUPS, 4, MAX_PATH_LEN, 2), jnp.float32),
        comet_path_lens=jnp.zeros((N_COMET_GROUPS, 4), jnp.int32),
        comet_ships=jnp.zeros(N_COMET_GROUPS, jnp.float32),
        comet_path_idx=jnp.full(N_COMET_GROUPS, -2, jnp.int32),
        step=jnp.int32(0),
        angular_velocity=jnp.float32(0.0),
        num_agents=jnp.int32(2),
        comet_id_base=jnp.int32(0),
    )
    _dummy_player = jnp.array([0], dtype=jnp.int32)
    _dummy_batched = jax.tree_util.tree_map(lambda x: x[None], _dummy_state)
    _dummy_enc = encode_fn(_dummy_batched, _dummy_player)
    _dummy_sf = _dummy_enc.self_feats[0]
    _dummy_cf = _dummy_enc.cand_feats[0]
    _dummy_gf = jnp.broadcast_to(_dummy_enc.global_feat[0][None, :], (MAX_SRC, GLOB_DIM))
    _dummy_cm = _dummy_enc.cand_mask[0]
    _dummy_sv = _dummy_enc.src_valid[0]
    _forward(_dummy_sf, _dummy_cf, _dummy_gf, _dummy_cm, _dummy_sv)

    return agent


def make_jax_agent(
    checkpoint_path: str | Path,
    deterministic: bool = True,
) -> Callable:
    """Return an obs->moves callable that runs a policy checkpoint.

    This is the public checkpoint-loading wrapper used by notebooks and arena
    scripts. The lower-level ``build_agent`` is useful when params/config are
    already loaded in memory.
    """
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    return build_agent(params, ckpt["config"], deterministic=deterministic)
