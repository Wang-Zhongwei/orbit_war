"""Accuracy probe for ORBIT_SIM_TICK_STRIDE — how wrong are the features at stride>1?

The IN=45 sim feeds the inbound_* ship features and the discrete capture predictions
would_be_mine_* / will_be_mine. Striding lumps combat within each window and accrues
production in chunks, so those features shift. This measures the shift on REAL mid-game
states (env stepped forward with the heuristic so fleets are actually in flight — at
step 0 inbound is all-zero and the error looks deceptively tiny).

For each stride S it compares the encoder output vs the exact stride=1 reference on:
  * will_be_mine        [cand plane 25] — flip rate over valid candidates (the key signal)
  * would_be_mine_final [cand plane 24] — flip rate
  * my_inbound_ships 45 [cand plane 15] — mean abs err + correlation
  * my_inbound 5/10/20  [cand planes 12/13/14]

Run as a single process: it re-encodes at each stride in-process by rebuilding the
encoder under different SIM_TICK_STRIDE (patched on the module before each build).

Usage: python -m tools.stride_accuracy --config configs/ppo_low_kl.yaml --warm 120
"""
from __future__ import annotations

import argparse
import functools
import os

os.environ.setdefault("WANDB_MODE", "offline")

import jax
import jax.numpy as jnp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_low_kl.yaml")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--warm", type=int, default=120, help="ticks to step before encoding (build up fleets)")
    ap.add_argument("--schedules", type=str, nargs="+",
                    default=["anchored", "1-45:2", "1-45:3"],
                    help="tick-schedule specs to compare against exact")
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}", flush=True)

    import src_jax.features as F
    from src_jax.config import load_train_config
    from src_jax.physics import jax_step, N_AGENTS, MAX_ACTIONS_PA
    from src_jax.heuristic import heuristic_actions
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg = load_train_config(args.config)
    K = cfg.env.candidate_count
    N = args.envs
    episode_steps = cfg.env.episode_steps

    bp = make_board_pool(2048, cfg.seed)
    state, _, _, _ = make_initial_carry_state(bp, 2048, N, cfg.seed, cfg.alternate_player_sides)

    step = jax.jit(jax.vmap(functools.partial(jax_step, episode_steps=episode_steps)))
    vheur = jax.vmap(heuristic_actions)
    p0 = jnp.zeros((N,), jnp.int32)
    p1 = jnp.ones((N,), jnp.int32)
    dummy = jnp.full((N, N_AGENTS - 2, MAX_ACTIONS_PA, 3), -1.0)

    print(f"Stepping {N} envs forward {args.warm} ticks (heuristic self-play) …", flush=True)
    for _ in range(args.warm):
        a0 = vheur(state, p0)
        a1 = vheur(state, p1)
        joint = jnp.concatenate([a0[:, None], a1[:, None], dummy], axis=1)
        state, _, _ = step(state, joint)
    n_fleets = int(jnp.sum(state.f_alive))
    print(f"  alive fleets across {N} envs: {n_fleets}  (avg {n_fleets/N:.1f}/env)", flush=True)

    players = p0

    def encode_spec(spec):
        ticks = F.schedule_from_spec(spec)
        F.set_tick_schedule(ticks)
        enc = jax.jit(F._make_env_encoder(K, cfg.env))
        return ticks, jax.block_until_ready(enc(state, players))

    _, ref = encode_spec("exact")
    cm = ref.cand_mask  # [N, MAX_SRC, K] bool — valid candidate cells
    valid = cm & ref.src_valid[:, :, None]
    nvalid = jnp.sum(valid).astype(jnp.float32)

    # cand plane indices (see features.py candidate feature list)
    P_INB5, P_INB10, P_INB20, P_INB45 = 12, 13, 14, 15
    P_WBM_FINAL, P_WILL_BE_MINE = 24, 25

    def flip_rate(a, b):
        return float(jnp.sum((jnp.abs(a - b) > 0.5) & valid).astype(jnp.float32) / nvalid)

    def cont_err(a, b):
        d = jnp.where(valid, jnp.abs(a - b), 0.0)
        mae = float(jnp.sum(d) / nvalid)
        denom = float(jnp.sum(jnp.where(valid, jnp.abs(a), 0.0)) / nvalid) + 1e-9
        return mae, mae / denom

    cf = ref.cand_feats
    print(f"\nReference = stride 1 (exact). Valid candidate cells: {int(nvalid)}", flush=True)
    print(f"Baseline rates @ stride1: will_be_mine={float(jnp.sum((cf[...,P_WILL_BE_MINE]>0.5)&valid)/nvalid):.3f} "
          f"would_be_mine_final={float(jnp.sum((cf[...,P_WBM_FINAL]>0.5)&valid)/nvalid):.3f}", flush=True)

    for spec in args.schedules:
        ticks, e = encode_spec(spec)
        cfs = e.cand_feats
        fr_will = flip_rate(cfs[..., P_WILL_BE_MINE], cf[..., P_WILL_BE_MINE])
        fr_wbm  = flip_rate(cfs[..., P_WBM_FINAL],   cf[..., P_WBM_FINAL])
        mae45, rel45 = cont_err(cfs[..., P_INB45], cf[..., P_INB45])
        mae20, rel20 = cont_err(cfs[..., P_INB20], cf[..., P_INB20])
        mae5,  rel5  = cont_err(cfs[..., P_INB5],  cf[..., P_INB5])
        print(f"\n=== schedule '{spec}' ({len(ticks)} ticks vs 45) vs exact ===", flush=True)
        print(f"  will_be_mine        flip rate: {fr_will*100:6.2f}%  of valid candidates", flush=True)
        print(f"  would_be_mine_final flip rate: {fr_wbm*100:6.2f}%", flush=True)
        print(f"  my_inbound_45  MAE {mae45:.4f}  (rel {rel45*100:.1f}%)", flush=True)
        print(f"  my_inbound_20  MAE {mae20:.4f}  (rel {rel20*100:.1f}%)", flush=True)
        print(f"  my_inbound_5   MAE {mae5:.4f}  (rel {rel5*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
