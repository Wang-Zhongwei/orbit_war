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
from .features import _make_env_encoder
from .physics import MAX_BASE_PLANETS, MAX_FLEETS
from .policy import build_policy
from .rollout import CAND_DIM, GLOB_DIM, SELF_DIM, FLEET_DIM


def _obs_to_dict(obs) -> dict:
    """Accept either a plain dict or a kaggle StructuredObservation object."""
    if isinstance(obs, dict):
        return obs
    # kaggle returns a struct-like; convert via __dict__ or vars
    try:
        return dict(obs)
    except TypeError:
        pass
    try:
        return vars(obs)
    except TypeError:
        pass
    # Last resort: attribute access for known keys
    keys = [
        "player", "step", "planets", "fleets", "initial_planets",
        "angular_velocity", "comet_planet_ids", "comets", "remainingOverageTime",
        "next_fleet_id",
    ]
    return {k: getattr(obs, k) for k in keys if hasattr(obs, k)}


def make_jax_agent(
    checkpoint_path: str | Path,
    deterministic: bool = True,
) -> Callable:
    """Return an obs→moves callable that runs the JAX policy from a checkpoint.

    The checkpoint must be in the format saved by bc_train.py or train.py
    (a pickle with keys 'params' and 'config').

    Parameters
    ----------
    checkpoint_path : path to the .pkl checkpoint file
    deterministic   : if True (default) take argmax; otherwise sample
    """
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    cfg    = ckpt["config"]
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])

    K = cfg.env.candidate_count
    B = cfg.env.ship_bucket_count

    module     = build_policy(cfg)
    encode_fn  = _make_env_encoder(K, cfg.env)

    # JIT-compile the forward pass once.  Input shapes must stay fixed:
    # [MAX_BASE_PLANETS, *] because we process exactly one env at a time.
    MAX_SRC = MAX_BASE_PLANETS

    @jax.jit
    def _forward(sf, cf, gf, cm, ff, fm):
        return module.apply(params, sf, cf, gf, cm, ff, fm)

    def agent(obs) -> list:
        obs_dict = _obs_to_dict(obs)
        state  = obs_to_env_state(obs_dict)
        player = jnp.array([int(obs_dict.get("player", 0))], dtype=jnp.int32)

        # Batch-dim 1 for the encoder (vmapped over envs)
        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        enc     = encode_fn(batched, player)

        # Flatten MAX_SRC decisions for the policy
        sf = enc.self_feats.reshape(MAX_SRC, SELF_DIM)
        cf = enc.cand_feats.reshape(MAX_SRC, K, CAND_DIM)
        gf = jnp.broadcast_to(enc.global_feat[0][None, :], (MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask.reshape(MAX_SRC, K)
        ff = jnp.broadcast_to(enc.fleet_feats[0][None, :, :], (MAX_SRC, MAX_FLEETS, FLEET_DIM))
        fm = jnp.broadcast_to(enc.fleet_mask[0][None, :],     (MAX_SRC, MAX_FLEETS))

        out = _forward(sf, cf, gf, cm, ff, fm)

        # Decode: argmax for deterministic play
        ti = jnp.argmax(out.target_logits, axis=-1)          # [MAX_SRC]
        row_idx  = jnp.arange(MAX_SRC)
        shp_logits = out.ships_logits[row_idx, ti]            # [MAX_SRC, B]
        sb = jnp.argmax(shp_logits, axis=-1)                  # [MAX_SRC]

        # Convert to moves
        src_valid = enc.src_valid[0]   # [MAX_SRC]
        src_ids   = enc.src_ids[0]     # [MAX_SRC]
        src_ships = enc.src_ships[0]   # [MAX_SRC] int32
        angles    = enc.angles[0]      # [MAX_SRC, K]
        cand_mask = enc.cand_mask[0]   # [MAX_SRC, K]

        moves: list = []
        for i in range(MAX_SRC):
            if not bool(src_valid[i]):
                continue
            t = int(ti[i])
            if t == 0:           # no-op chosen
                continue
            if not bool(cand_mask[i, t]):
                continue
            S     = int(src_ships[i])
            ships = max(1, int((int(sb[i]) + 1) * S // B))
            angle = float(angles[i, t])
            moves.append([int(src_ids[i]), angle, ships])

        return moves

    return agent
