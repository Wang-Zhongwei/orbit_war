"""Runtime probe — measure the EXECUTION wall-clock of the rollout collect step.

Unlike tools/compile_probe.py (which AOT-compiles only, measuring compile cost),
this script actually RUNS collect() repeatedly with block_until_ready and reports
per-call milliseconds. Use it to A/B a launch-overhead lever (e.g. XLA command
buffers for WHILE) by running it twice with different XLA_FLAGS and comparing the
median collect time.

The collect step is launch/serialization bound, so its wall-clock is structural
(kernel + while-loop iteration count) and independent of weight VALUES — random
init params reproduce the real run's launch profile. We mirror the real training
build: n_opponents=1, all-neural opponent, augment from the config.

Usage (see jobs/templates/collect_bench.slurm). A/B example:
    # baseline
    python -m tools.collect_bench --config configs/ppo_low_kl.yaml
    # with command buffers (WHILE captured into CUDA graphs)
    XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE" \
        python -m tools.collect_bench --config configs/ppo_low_kl.yaml
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("WANDB_MODE", "offline")

import jax
import jax.numpy as jnp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_low_kl.yaml")
    ap.add_argument("--warmup", type=int, default=2, help="collect calls before timing (triggers compile)")
    ap.add_argument("--iters", type=int, default=8, help="timed collect calls")
    args = ap.parse_args()

    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    print(f"JAX backend: {jax.default_backend()}  devices={jax.devices()}", flush=True)

    from src_jax.config import load_train_config
    from src_jax.policies import build_policy
    from src_jax.ppo import build_train_state
    from src_jax.rollout import make_rollout_fn, RolloutCarry
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg = load_train_config(args.config)
    if getattr(cfg.ppo, "matmul_precision", ""):
        jax.config.update("jax_default_matmul_precision", cfg.ppo.matmul_precision)
        print(f"matmul precision -> {cfg.ppo.matmul_precision}", flush=True)

    N_ENVS  = cfg.ppo.num_envs
    T       = cfg.ppo.rollout_steps
    N_BOARDS = 2048
    key = jax.random.PRNGKey(cfg.seed)

    print(f"Config: K={cfg.env.candidate_count} H={cfg.model.hidden_size} "
          f"num_envs={N_ENVS} rollout={T} augment={cfg.ppo.augment} "
          f"matmul={getattr(cfg.ppo, 'matmul_precision', '')}", flush=True)

    policy = build_policy(cfg)
    train_state = build_train_state(cfg, seed=cfg.seed)
    board_pool = make_board_pool(N_BOARDS, cfg.seed)

    # Mirror the real run: one neural opponent (n_opponents=1, no heuristic).
    collect = make_rollout_fn(
        board_pool=board_pool, n_boards=N_BOARDS, cfg=cfg, module=policy,
        rollout_steps=T, n_opponents=1, heuristic_opp_mask=(False,),
        augment=cfg.ppo.augment,
    )
    batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
        board_pool, N_BOARDS, N_ENVS, cfg.seed, cfg.alternate_player_sides
    )
    carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)
    stacked_opp = jax.tree_util.tree_map(lambda x: x[None], train_state.params)  # [1, ...]

    print(f"\nWarmup ({args.warmup} calls, first triggers compile) …", flush=True)
    t0 = time.perf_counter()
    for i in range(args.warmup):
        carry, steps = collect(train_state.params, stacked_opp, carry)
        jax.block_until_ready(steps)
        print(f"  warmup {i}: {time.perf_counter()-t0:.2f}s cumulative", flush=True)
        t0 = time.perf_counter()

    print(f"\nTiming {args.iters} collect calls …", flush=True)
    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        carry, steps = collect(train_state.params, stacked_opp, carry)
        jax.block_until_ready(steps)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {i}: {dt*1000:8.1f} ms", flush=True)

    med = statistics.median(times)
    print(f"\n=== collect: median {med*1000:.1f} ms  "
          f"(mean {statistics.mean(times)*1000:.1f}, min {min(times)*1000:.1f}) "
          f"over {len(times)} calls; {med*1000/T:.2f} ms/step ===", flush=True)


if __name__ == "__main__":
    main()
