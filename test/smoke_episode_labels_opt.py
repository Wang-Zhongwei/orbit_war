"""Smoke test: vectorized _episode_target_slots labels match the batch simulator.

Verifies the vectorized label extraction produces sensible labels on real episodes,
and runs the full episode_to_samples + assemble_for_k end-to-end to catch regressions.

Run on a compute node (padded encoder is too heavy for the login node):
    python -m test.smoke_episode_labels_opt --episodes_dir episodes/2026-05-01 --n 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from src_jax.config import TrainConfig
from src_jax.features import _make_env_encoder
import src_jax.bc_data as bd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-05-01")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    print(f"jax devices: {jax.devices()}")
    cfg = TrainConfig()
    K = cfg.env.candidate_count
    encode_fn = _make_env_encoder(bd.POOL_K, cfg.env)

    eps = []
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        if isinstance(ep, dict) and "steps" in ep:
            eps.append(ep)
        if len(eps) >= args.n:
            break
    assert eps, f"no episodes in {args.episodes_dir}"
    print(f"loaded {len(eps)} episodes from {args.episodes_dir}")

    # --- label extraction sanity: every PLANET-labelled action lands on a live planet ---
    n_launch_total = n_planet_total = 0
    for ep in eps:
        states, players, actions = bd._collect_states_actions(ep)
        if not states:
            continue
        N = len(states)
        batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *states)
        enc = jax.tree_util.tree_map(np.asarray,
                                     encode_fn(batched, jnp.array(players, jnp.int32)))
        b = jax.tree_util.tree_map(np.asarray, batched)

        labels = bd._episode_target_slots(
            src_ids=enc.src_ids, src_valid=enc.src_valid, src_slots=enc.src_slots,
            p_x=b.p_x, p_y=b.p_y, p_init_x=b.p_init_x, p_init_y=b.p_init_y,
            p_radius=b.p_radius, p_alive=b.p_alive, p_is_comet=b.p_is_comet,
            angular_velocity=b.angular_velocity, step=b.step,
            batch_actions=actions, N=N)

        for i, row in enumerate(labels):
            for _sidx, tslot in row:
                n_launch_total += 1
                if tslot != -1:
                    n_planet_total += 1
                    assert 0 <= tslot < b.p_alive.shape[1], \
                        f"target_slot {tslot} out of range at state {i}"
                    assert bool(b.p_alive[i][tslot]), \
                        f"target_slot {tslot} is dead at state {i}"

    print(f"\nlabeled {n_launch_total} launches, {n_planet_total} resolved to a live planet")
    print("_episode_target_slots: all PLANET labels land on live planets ✓")

    # --- end-to-end: pool format + assemble_for_k ---
    t0 = time.perf_counter()
    samps = [s for ep in eps if (s := bd.episode_to_samples(ep, encode_fn)) is not None]
    dt = time.perf_counter() - t0
    assert samps, "episode_to_samples produced no samples"
    pool = bd.concat_samples(samps)

    ts = pool["target_slot"]
    noop_frac = float(np.mean(ts == -1))
    print(f"\nepisode_to_samples: {len(ts)} samples in {dt:.1f}s, "
          f"noop_frac={noop_frac:.3f}")
    assert pool["pool_feats"].shape[1] == bd.POOL_K
    assert 0.0 < (1.0 - noop_frac) < 1.0, "labels are all no-op or all launch"
    print("pool format: shapes and label distribution OK ✓")

    d = bd.assemble_for_k(pool, K)
    ti = d["target_idx"]
    assert ti.min() >= 0 and ti.max() < K, \
        f"assembled target_idx out of [0,{K}): {ti.min()}..{ti.max()}"
    print(f"assemble_for_k(K={K}): target_idx in [{int(ti.min())},{int(ti.max())}] ✓")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
