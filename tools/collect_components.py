"""Component probe — localize WHERE the ~830 ms/step collect cost goes.

The command-buffer A/B (job 48874) showed collect is NOT launch-bound: capturing
the while-loops into CUDA graphs changed nothing (53.2s vs 53.1s). So the time is
genuine sequential GPU compute. This probe times each piece of the scan body in
isolation at the real shapes (num_envs=1024) so we can see which dominates:

  * encode_batch  — runs TWICE per real step (learner + opponent, same board)
  * policy fwd    — runs TWICE per real step (learner params + opponent params)
  * jax_step      — physics, runs ONCE per real step

Real step ≈ 2*encode + 2*fwd + physics + glue. If encode dominates, fusing the two
encodes (one 2N batch) or making the encoder cheaper is the lever; if physics
dominates, the blocking/fleet sim is; etc.

Pair with `nvidia-smi dmon` (see jobs/templates/collect_components.slurm): if GPU
util is pinned ~100% during the loops, it is compute-bound (command buffers / unroll
cannot help — must reduce work). If util is low, it is stall-bound.

Usage:  python -m tools.collect_components --config configs/ppo_low_kl.yaml
"""
from __future__ import annotations

import argparse
import functools
import os
import statistics
import time

os.environ.setdefault("WANDB_MODE", "offline")

import jax
import jax.numpy as jnp


def _bench(name, fn, warmup=2, iters=6):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print(f"  {name:<28} median {med*1000:8.1f} ms  (min {min(ts)*1000:.1f})", flush=True)
    return med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_low_kl.yaml")
    args = ap.parse_args()

    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    print(f"JAX backend: {jax.default_backend()}  devices={jax.devices()}", flush=True)

    from src_jax.config import load_train_config
    from src_jax.policies import build_policy
    from src_jax.ppo import build_train_state
    from src_jax.features import _make_env_encoder, global_feature_dim
    from src_jax.physics import jax_step, MAX_BASE_PLANETS, MAX_ACTIONS_PA, N_AGENTS
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg = load_train_config(args.config)
    if getattr(cfg.ppo, "matmul_precision", ""):
        jax.config.update("jax_default_matmul_precision", cfg.ppo.matmul_precision)
        print(f"matmul precision -> {cfg.ppo.matmul_precision}", flush=True)

    K        = cfg.env.candidate_count
    N        = cfg.ppo.num_envs
    MAX_SRC  = MAX_BASE_PLANETS
    GLOB_DIM = global_feature_dim()
    episode_steps = cfg.env.episode_steps
    N_BOARDS = 2048

    print(f"Config: K={K} H={cfg.model.hidden_size} num_envs={N} "
          f"rollout={cfg.ppo.rollout_steps} MAX_SRC={MAX_SRC} "
          f"matmul={getattr(cfg.ppo,'matmul_precision','')}", flush=True)

    policy = build_policy(cfg)
    train_state = build_train_state(cfg, seed=cfg.seed)
    params = train_state.params
    board_pool = make_board_pool(N_BOARDS, cfg.seed)
    batched_state, _bidx, _ep, learner_players = make_initial_carry_state(
        board_pool, N_BOARDS, N, cfg.seed, cfg.alternate_player_sides
    )
    opp_players = 1 - learner_players
    from src_jax.feature_versions import make_env_encoder
    encode_batch = make_env_encoder(K, cfg.env, getattr(cfg.model, "feat_version", "v7"))

    # ── component fns (mirror rollout.py scan_fn) ─────────────────────────────
    enc_one = jax.jit(encode_batch)

    def _fwd(p, enc):
        gf = jnp.broadcast_to(enc.global_feat[:, None, :], (N, MAX_SRC, GLOB_DIM))
        out = jax.vmap(
            lambda a, b, c, d, e: policy.apply(p, a, b, c, d, e)
        )(enc.self_feats, enc.cand_feats, gf, enc.cand_mask, enc.src_valid)
        return out
    fwd_one = jax.jit(_fwd)

    batched_step = jax.jit(jax.vmap(functools.partial(jax_step, episode_steps=episode_steps)))
    joint_noop = jnp.full((N, N_AGENTS, MAX_ACTIONS_PA, 3), -1.0, jnp.float32)

    # materialize a real encoding once for the forward probe
    l_enc = jax.block_until_ready(enc_one(batched_state, learner_players))

    print(f"\nComponent timings at num_envs={N} (real step ≈ 2*encode + 2*fwd + physics):", flush=True)
    t_enc  = _bench("encode_batch (x2/step)", lambda: enc_one(batched_state, learner_players))
    t_fwd  = _bench("policy forward (x2/step)", lambda: fwd_one(params, l_enc))
    t_phys = _bench("jax_step physics (x1/step)", lambda: batched_step(batched_state, joint_noop))

    est = 2 * t_enc + 2 * t_fwd + t_phys
    print(f"\n  => est per-step (2enc+2fwd+phys): {est*1000:8.1f} ms"
          f"   [encode {2*t_enc*1000:.0f} | fwd {2*t_fwd*1000:.0f} | phys {t_phys*1000:.0f}]", flush=True)
    print(f"  (measured full collect ~= 830 ms/step; remainder = action-build/reward/reset glue)", flush=True)
    print(f"\n  SHARES: encode {200*t_enc/est:.0f}%  fwd {200*t_fwd/est:.0f}%  phys {100*t_phys/est:.0f}%", flush=True)
    print(f"  encode-fuse ceiling (drop 1 of 2 encodes): -{t_enc*1000:.0f} ms/step "
          f"(~-{100*t_enc/est:.0f}% of step)", flush=True)


if __name__ == "__main__":
    main()
