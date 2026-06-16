"""Compile-time probe — localize *why* the first JAX compile is slow.

Compiles each jitted program AHEAD-OF-TIME (``.lower().compile()``) without ever
executing it, so we measure pure compile cost — separately for:

  * ``.lower()``   — Python tracing + initial HLO construction (platform-independent)
  * ``.compile()`` — XLA optimization passes + backend codegen
                     (on GPU this is where LLVM + per-kernel ``ptxas`` runs)

Run it on TWO backends to localize the bottleneck:

  * ``JAX_PLATFORMS=cpu``  → no GPU needed. Times the platform-independent work.
                             If a function is slow HERE, the graph itself is huge
                             (the fix is shrinking the HLO, not more cores/ptxas).
  * default (gpu)          → adds XLA GPU codegen + ptxas. If a function is fast on
                             CPU but slow here, the bottleneck is ptxas/codegen
                             (fix: --xla_gpu_force_compilation_parallelism, or a
                             single oversized kernel that can't be parallelized).

Inputs are passed as abstract ``ShapeDtypeStruct`` shapes where large, so nothing
big is allocated and no GPU memory is touched — this is compile-only.

Usage (see jobs/templates/compile_probe.slurm):
    JAX_LOG_COMPILES=1 python -m tools.compile_probe --config configs/ppo_from_bc_loser0.yaml
    JAX_PLATFORMS=cpu  python -m tools.compile_probe --config configs/ppo_from_bc_loser0.yaml
"""
from __future__ import annotations

import argparse
import os
import time

# Per-stage timing below is the primary signal; set JAX_LOG_COMPILES=1 in the
# environment if you also want JAX's per-compilation log lines (very noisy).
os.environ.setdefault("JAX_LOG_COMPILES", "0")

import jax
import jax.numpy as jnp


def _sds(shape, dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape, dtype)


def _timed(label, fn):
    """Run fn(), print elapsed, return result."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"    {label:<24} {dt:8.2f}s", flush=True)
    return out, dt


def _probe(name, jitted, args, kwargs=None):
    """AOT lower + compile a jitted fn; report stage timings + HLO size."""
    kwargs = kwargs or {}
    print(f"\n=== {name} ===", flush=True)
    lowered, t_lower = _timed("lower (trace+HLO)", lambda: jitted.lower(*args, **kwargs))
    # HLO op count is a proxy for graph size (drives every XLA pass).
    try:
        hlo = lowered.as_text()
        n_lines = hlo.count("\n")
        n_ops = hlo.count(" = ")
        print(f"    HLO: ~{n_ops} ops, {n_lines} lines", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"    HLO size unavailable: {e}", flush=True)
    compiled, t_compile = _timed("compile (XLA+codegen)", lambda: lowered.compile())
    print(f"    --> TOTAL {t_lower + t_compile:8.2f}s  ({name})", flush=True)
    return compiled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_from_bc_loser0.yaml")
    ap.add_argument("--skip-collect", action="store_true",
                    help="skip the (heaviest) rollout-collect probe")
    ap.add_argument("--only", choices=["forward", "ppo", "collect"], default=None,
                    help="run only one probe (forward is lightest)")
    args = ap.parse_args()
    only = args.only

    print(f"JAX backend: {jax.default_backend()}  devices={jax.devices()}", flush=True)
    print(f"matmul precision: {jax.config.jax_default_matmul_precision}", flush=True)

    from src_jax.config import load_train_config
    from src_jax.policies import build_policy, init_params
    from src_jax.ppo import build_train_state, _ppo_update_jit, TrainingBatch
    from src_jax.rollout import make_rollout_fn, RolloutCarry
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg = load_train_config(args.config)
    if getattr(cfg.ppo, "matmul_precision", ""):
        jax.config.update("jax_default_matmul_precision", cfg.ppo.matmul_precision)
        print(f"matmul precision override -> {cfg.ppo.matmul_precision}", flush=True)

    K       = cfg.env.candidate_count
    N_ENVS  = cfg.ppo.num_envs
    T       = cfg.ppo.rollout_steps
    B       = N_ENVS * T
    N_BOARDS = 2048

    print(f"\nConfig: arch={cfg.model.arch} H={cfg.model.hidden_size} K={K} "
          f"MAX_SRC=44 num_envs={N_ENVS} rollout={T} B={B} "
          f"minibatch={cfg.ppo.minibatch_size} epochs={cfg.ppo.epochs} "
          f"kl_coef={cfg.ppo.kl_coef}", flush=True)

    # ── Build the (small) real objects needed to construct the jitted fns ──────
    t0 = time.perf_counter()
    policy = build_policy(cfg)
    train_state = build_train_state(cfg, seed=cfg.seed)
    SELF_DIM = policy.self_dim
    CAND_DIM = policy.candidate_dim
    GLOB_DIM = policy.global_dim
    MAX_SRC  = 44  # from physics.MAX_BASE_PLANETS
    print(f"build_policy + build_train_state: {time.perf_counter()-t0:.2f}s  "
          f"(self={SELF_DIM} cand={CAND_DIM} glob={GLOB_DIM})", flush=True)

    # ── Probe 1: single-env policy forward (vmapped over MAX_SRC) ──────────────
    # This is the transformer compile in isolation — the building block that the
    # rollout scan and the ppo update both stamp out.
    @jax.jit
    def _one_env_fwd(params, sf, cf, gf, cm, sv):
        # policy.apply already consumes the [MAX_SRC, ...] per-env layout
        # (MAX_SRC is its internal token-batch dim); no extra vmap needed.
        return policy.apply(params, sf, cf, gf, cm, sv)

    if only in (None, "forward"):
        _probe(
            "policy_forward (vmap MAX_SRC, 1 env)",
            _one_env_fwd,
            (train_state.params,
             _sds((MAX_SRC, SELF_DIM)),
             _sds((MAX_SRC, K, CAND_DIM)),
             _sds((MAX_SRC, GLOB_DIM)),
             _sds((MAX_SRC, K), jnp.bool_),
             _sds((MAX_SRC,), jnp.bool_)),
        )

    # ── Probe 2: ppo update (jitted scan over epochs × minibatches) ────────────
    key = jax.random.PRNGKey(0)
    use_kl = cfg.ppo.kl_coef > 0.0
    if only in (None, "ppo"):
        batch = TrainingBatch(
            self_features      = _sds((B, MAX_SRC, SELF_DIM)),
            candidate_features = _sds((B, MAX_SRC, K, CAND_DIM)),
            global_features    = _sds((B, GLOB_DIM)),
            candidate_mask     = _sds((B, MAX_SRC, K), jnp.bool_),
            target_index       = _sds((B, MAX_SRC), jnp.int32),
            log_prob           = _sds((B, MAX_SRC)),
            returns            = _sds((B,)),
            advantages         = _sds((B, MAX_SRC)),
            src_valid          = _sds((B, MAX_SRC), jnp.bool_),
            ref_target_logits  = _sds((B, MAX_SRC, K)),
        )
        _probe(
            f"ppo_update (epochs={cfg.ppo.epochs} mb={cfg.ppo.minibatch_size} kl={use_kl})",
            _ppo_update_jit,
            (train_state, batch),
            dict(clip_coef=cfg.ppo.clip_coef, ent_coef=cfg.ppo.ent_coef,
                 vf_coef=cfg.ppo.vf_coef, epochs=cfg.ppo.epochs,
                 minibatch_size=cfg.ppo.minibatch_size, key=key,
                 kl_coef=cfg.ppo.kl_coef, use_kl=use_kl),
        )

    # ── Probe 3: full rollout collect (the prime suspect) ──────────────────────
    if only in (None, "collect") and not args.skip_collect:
        t0 = time.perf_counter()
        board_pool = make_board_pool(N_BOARDS, cfg.seed)
        collect = make_rollout_fn(
            board_pool=board_pool, n_boards=N_BOARDS, cfg=cfg, module=policy,
            rollout_steps=T, n_opponents=2,
            heuristic_opp_mask=(False, True), augment=cfg.ppo.augment,
        )
        batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
            board_pool, N_BOARDS, N_ENVS, cfg.seed, cfg.alternate_player_sides
        )
        carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)
        stacked_opp = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), train_state.params, train_state.params
        )
        print(f"\nbuild rollout inputs: {time.perf_counter()-t0:.2f}s", flush=True)
        _probe(
            "collect (rollout scan, full)",
            collect,
            (train_state.params, stacked_opp, carry),
        )

    print("\nDONE. Compare CPU vs GPU runs: a fn slow on BOTH = HLO/graph size; "
          "fast on CPU + slow on GPU = ptxas/codegen.", flush=True)


if __name__ == "__main__":
    main()
