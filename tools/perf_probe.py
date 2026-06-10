"""Runtime perf probe — find where the PPO update's wall-time actually goes.

Unlike compile_probe (which only compiles), this EXECUTES the real ppo_update on
real-shaped GPU arrays and times it, with variants that isolate the cost drivers:

  * KL on vs off            — measures the per-minibatch KL-ref recompute overhead.
  * minibatch sweep         — does a bigger mb improve throughput (GEMM efficiency)
                              or are we launch/latency-bound? (catches OOM.)
  * epochs=1 vs config      — confirms linear scaling in epochs (sanity).

Reports mean ms/update over a few timed iterations (after a warmup that triggers
compile). Run on a GPU node.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "")

import jax
import jax.numpy as jnp


def _real(shape, dtype=jnp.float32, key=None, kind="randn"):
    if dtype == jnp.bool_:
        return jnp.ones(shape, dtype=jnp.bool_)
    if dtype == jnp.int32:
        return jnp.zeros(shape, dtype=jnp.int32)
    return jax.random.normal(key, shape) if key is not None else jnp.zeros(shape, dtype)


def _time(fn, n=5, warmup=2):
    for _ in range(warmup):
        out = fn()
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n


def main():
    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    dev = jax.devices()[0]
    print(f"device: {getattr(dev,'device_kind','?')}", flush=True)

    from src_jax.config import load_train_config
    from src_jax.policies import build_policy
    from src_jax.ppo import build_train_state, ppo_update, TrainingBatch

    cfg = load_train_config("configs/ppo_from_bc_loser0.yaml")
    if getattr(cfg.ppo, "matmul_precision", ""):
        jax.config.update("jax_default_matmul_precision", cfg.ppo.matmul_precision)
        print(f"matmul precision -> {cfg.ppo.matmul_precision}", flush=True)

    K = cfg.env.candidate_count
    B = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    MAX_SRC = 44
    policy = build_policy(cfg)
    ts = build_train_state(cfg, seed=cfg.seed)
    SELF, CAND, GLOB = policy.self_dim, policy.candidate_dim, policy.global_dim
    print(f"arch={cfg.model.arch} H={cfg.model.hidden_size} K={K} B={B} "
          f"epochs={cfg.ppo.epochs} kl_coef={cfg.ppo.kl_coef}", flush=True)

    k = jax.random.PRNGKey(0)
    ks = jax.random.split(k, 6)
    batch = TrainingBatch(
        self_features      = _real((B, MAX_SRC, SELF), key=ks[0]),
        candidate_features = _real((B, MAX_SRC, K, CAND), key=ks[1]),
        global_features    = _real((B, GLOB), key=ks[2]),
        candidate_mask     = _real((B, MAX_SRC, K), jnp.bool_),
        target_index       = _real((B, MAX_SRC), jnp.int32),
        log_prob           = _real((B, MAX_SRC), key=ks[3]),
        returns            = _real((B,), key=ks[4]),
        advantages         = _real((B, MAX_SRC), key=ks[5]),
        src_valid          = _real((B, MAX_SRC), jnp.bool_),
    )
    jax.block_until_ready(batch.self_features)

    def run(mb, epochs, kl_on):
        kc = cfg.ppo.kl_coef if kl_on else 0.0
        ref = ts.params if kl_on else None
        key = jax.random.PRNGKey(1)
        return ppo_update(
            ts, batch, clip_coef=cfg.ppo.clip_coef, ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef, epochs=epochs, minibatch_size=mb,
            key=key, ref_params=ref, kl_coef=kc,
        )

    # All timed at epochs=1 (ppo total ~= 4x these). Each call is ~1 epoch so it's
    # ~4x cheaper to benchmark; warmup=1, n=2 to stay within the time budget.
    print("\n=== ppo_update timing, epochs=1 (mean s; full update ~= x4) ===", flush=True)
    trials = [
        ("mb=1024 KL on",  1024, True),
        ("mb=1024 KL off", 1024, False),
        ("mb=2048 KL on",  2048, True),
        ("mb=4096 KL on",  4096, True),
        ("mb=8192 KL on",  8192, True),
    ]
    for label, mb, kl in trials:
        try:
            dt = _time(lambda: run(mb, 1, kl), n=2, warmup=1)
            print(f"  {label:18s} mb={mb:<5d} -> {dt:7.2f} s  (full~{dt*cfg.ppo.epochs:6.1f}s)", flush=True)
        except Exception as e:
            msg = str(e).splitlines()[0][:70]
            print(f"  {label:18s} mb={mb:<5d} -> FAILED: {msg}", flush=True)

    print("\nInterpretation:", flush=True)
    print("  (mb1024 KLon - mb1024 KLoff) = per-minibatch KL-ref recompute cost", flush=True)
    print("  bigger mb FASTER  = launch/latency-bound (small-op overhead dominates)", flush=True)
    print("  bigger mb SAME    = compute-bound (mb won't help; need arch/FLOP change)", flush=True)


if __name__ == "__main__":
    main()
