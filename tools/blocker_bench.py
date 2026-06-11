"""Benchmark: cost of adding rotating-planet ("moving") blocking to the feature
encoder's _find_angle_single.

The encoder normally treats only *static* planets as flight-path blockers
(positions frozen at launch). _find_angle_single grew an optional
`moving_blockers` path that also walks the fleet forward tick-by-tick and tests
it against every rotating planet's time-varying position
(O(MAX_TICKS_OUT * MAX_PLANETS) per [src, tgt] pair).

This times the *full* batched encoder (jit+vmap over N envs) both ways on the
same states and reports the per-encode and per-env-tick slowdown, so we can
decide whether the accuracy is worth the throughput hit.

Run on a GPU node (the encoder is what runs every rollout tick on GPU).
"""
from __future__ import annotations

import dataclasses
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from src_jax.config import EnvConfig
from src_jax.env import make_board_pool, select_boards
from src_jax.features import _make_env_encoder, MAX_TICKS_OUT
from src_jax.physics import MAX_PLANETS, MAX_BASE_PLANETS, MAX_FLEETS
import jax.numpy as jnp


def _time(fn, n=30, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / n


def main():
    N_ENVS  = int(os.environ.get("N_ENVS", "256"))
    N_BOARDS = int(os.environ.get("N_BOARDS", "32"))
    K_cfg   = int(os.environ.get("K", "12"))

    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    dev = jax.devices()[0]
    print(f"device: {getattr(dev, 'device_kind', '?')}", flush=True)
    print(f"N_ENVS={N_ENVS} N_BOARDS={N_BOARDS} K={K_cfg} "
          f"MAX_PLANETS={MAX_PLANETS} MAX_SRC={MAX_BASE_PLANETS} "
          f"MAX_FLEETS={MAX_FLEETS} MAX_TICKS_OUT={MAX_TICKS_OUT}", flush=True)

    # ── Build a batch of real initial states ─────────────────────────────────
    pool = make_board_pool(N_BOARDS, seed=0)
    idx = jnp.arange(N_ENVS, dtype=jnp.int32)
    state = select_boards(pool, idx, N_BOARDS)
    players = jnp.zeros(N_ENVS, dtype=jnp.int32)
    jax.block_until_ready(state.p_x)

    base_cfg = EnvConfig(candidate_count=K_cfg)

    cfg_static = dataclasses.replace(base_cfg)
    cfg_static.moving_blockers = False
    cfg_moving = dataclasses.replace(base_cfg)
    cfg_moving.moving_blockers = True

    enc_static = _make_env_encoder(K_cfg, cfg_static)
    enc_moving = _make_env_encoder(K_cfg, cfg_moving)

    f_static = lambda: enc_static(state, players)
    f_moving = lambda: enc_moving(state, players)

    # ── Correctness sanity: moving blocking is a strict superset of static ───
    es = jax.block_until_ready(f_static())
    em = jax.block_until_ready(f_moving())
    blk_s = (~es.cand_mask).sum()
    blk_m = (~em.cand_mask).sum()
    superset = bool(jnp.all(es.cand_mask | ~em.cand_mask))  # moving only ADDS blocks
    print(f"\nsanity: blocked-cands  static={int(blk_s)}  moving={int(blk_m)}  "
          f"(moving>=static superset={superset})", flush=True)

    # ── Time both ────────────────────────────────────────────────────────────
    t_static = _time(f_static)
    t_moving = _time(f_moving)

    per_env_us_s = t_static / N_ENVS * 1e6
    per_env_us_m = t_moving / N_ENVS * 1e6
    print("\n=== encoder wall-time (mean over timed iters) ===", flush=True)
    print(f"  static-only blocking : {t_static*1e3:8.3f} ms/call  "
          f"({per_env_us_s:7.2f} us / env-tick)", flush=True)
    print(f"  + moving  blocking   : {t_moving*1e3:8.3f} ms/call  "
          f"({per_env_us_m:7.2f} us / env-tick)", flush=True)
    print(f"\n  slowdown  : {t_moving/t_static:5.2f}x   "
          f"(+{(t_moving-t_static)*1e3:.3f} ms/call, "
          f"+{per_env_us_m-per_env_us_s:.2f} us/env-tick)", flush=True)


if __name__ == "__main__":
    main()
