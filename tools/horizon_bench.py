"""Benchmark: encoder cost of the flight-horizon widening
MAX_TICKS_OUT 30->60 (rotating-intercept scan) and MAX_TICKS_IN 30->45
(inbound / would-be-mine outcome sim).

Both constants live at module scope in src_jax.features and are read at trace
time (jnp.arange / range), so monkeypatching them before _make_env_encoder()
fully bakes the new horizon into the compiled encoder. We time the full batched
encoder (jit+vmap over N envs) at each (OUT, IN) setting and report the per-tick
delta. Moving-blocker (rotating planet + comet) detection is always on, so the
numbers include its swept scan at each horizon.

Run on a GPU node.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import src_jax.features as features
from src_jax.config import EnvConfig
from src_jax.env import make_board_pool, select_boards
from src_jax.physics import MAX_PLANETS, MAX_BASE_PLANETS, MAX_FLEETS


def _time(fn, n=30, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / n


def _bench(state, players, K, n_envs, out_ticks, in_ticks):
    # Monkeypatch the module-level horizons, then build a fresh encoder so the
    # new values are baked into the trace.
    features.MAX_TICKS_OUT = out_ticks
    features.MAX_TICKS_IN = in_ticks
    cfg = EnvConfig(candidate_count=K)
    enc = features._make_env_encoder(K, cfg)
    f = lambda: enc(state, players)
    jax.block_until_ready(f())
    dt = _time(f)
    return dt, dt / n_envs * 1e6


def main():
    N_ENVS = int(os.environ.get("N_ENVS", "1024"))
    N_BOARDS = int(os.environ.get("N_BOARDS", "32"))
    K = int(os.environ.get("K", "12"))

    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    print(f"device: {getattr(jax.devices()[0], 'device_kind', '?')}", flush=True)
    print(f"N_ENVS={N_ENVS} N_BOARDS={N_BOARDS} K={K} "
          f"MAX_PLANETS={MAX_PLANETS} MAX_SRC={MAX_BASE_PLANETS} "
          f"MAX_FLEETS={MAX_FLEETS}", flush=True)

    pool = make_board_pool(N_BOARDS, seed=0)
    idx = jnp.arange(N_ENVS, dtype=jnp.int32)
    state = select_boards(pool, idx, N_BOARDS)
    players = jnp.zeros(N_ENVS, dtype=jnp.int32)
    jax.block_until_ready(state.p_x)

    settings = [
        ("old  (OUT=30, IN=30)", 30, 30),
        ("new  (OUT=60, IN=45)", 60, 45),
        # isolating each axis:
        ("OUT-only (OUT=60, IN=30)", 60, 30),
        ("IN-only  (OUT=30, IN=45)", 30, 45),
    ]
    print("\n=== encoder wall-time ===", flush=True)
    results = {}
    for label, ot, it in settings:
        dt, per = _bench(state, players, K, N_ENVS, ot, it)
        results[(ot, it)] = (dt, per)
        print(f"  {label:26s} : {dt*1e3:8.3f} ms/call  ({per:7.2f} us/env-tick)",
              flush=True)

    old_dt, old_per = results[(30, 30)]
    new_dt, new_per = results[(60, 45)]
    print(f"\n  old -> new : {new_dt/old_dt:5.2f}x  "
          f"(+{(new_dt-old_dt)*1e3:.3f} ms/call, +{new_per-old_per:.2f} us/env-tick)",
          flush=True)


if __name__ == "__main__":
    main()
