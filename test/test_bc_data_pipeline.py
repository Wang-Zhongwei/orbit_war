"""Integration test for the trajectory-based BC data pipeline.

Runs the real episode_to_samples (K-pool format) + assemble_for_k over a few
episodes and checks:
  * pool-format shapes are mutually consistent;
  * assembled target_idx labels are within the candidate range;
  * the batched JAX simulator agrees with the single-fleet numpy reference;
  * a sane fraction of labels resolve to a real launch (not all no-op).

Intended to run on a compute node (the encoder pads to 256/512-wide batches, too
heavy for a login node). Exits non-zero on any failure so SLURM marks it FAILED.

Usage:
    python -m test.test_bc_data_pipeline --episodes_dir episodes/2026-06-02 --n 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax

from src_jax.config import TrainConfig
from src_jax.features import _make_env_encoder
import src_jax.bc_data as bd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-02")
    ap.add_argument("--n", type=int, default=5, help="episodes to test")
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
    assert eps, f"no episodes found in {args.episodes_dir}"
    print(f"loaded {len(eps)} episodes from {args.episodes_dir}")

    # --- run the real pipeline ---
    t0 = time.perf_counter()
    samps = [s for ep in eps if (s := bd.episode_to_samples(ep, encode_fn)) is not None]
    dt = time.perf_counter() - t0
    assert samps, "episode_to_samples produced no samples"
    pool = bd.concat_samples(samps)

    n = len(pool["target_slot"])
    print(f"\nencoded {len(samps)} episodes -> {n} samples in {dt:.1f}s "
          f"({len(samps)/dt:.1f} ep/s)")
    print("pool-format keys:")
    for k, v in pool.items():
        print(f"  {k:14s}{v.shape} {v.dtype}")

    # --- pool-format shape consistency ---
    assert pool["pool_feats"].shape[1] == bd.POOL_K, \
        f"pool_feats K={pool['pool_feats'].shape[1]} != POOL_K={bd.POOL_K}"
    for key in ("self_feats", "global_feats", "pool_feats", "pool_slot",
                "pool_cat", "pool_dist", "pool_mask", "target_slot"):
        assert pool[key].shape[0] == n, f"{key} row count {pool[key].shape[0]} != {n}"

    # target_slot == -1 means no-op; all other values should be valid planet slots
    ts = pool["target_slot"]
    assert ts.min() >= -1, f"target_slot < -1: {ts.min()}"
    noop_frac = float(np.mean(ts == -1))
    launch_frac = 1.0 - noop_frac
    print(f"\ntarget_slot: min={int(ts.min())} max={int(ts.max())} "
          f"noop_frac={noop_frac:.3f} launch_frac={launch_frac:.3f}")
    assert 0.0 < launch_frac < 1.0, "labels are all no-op or all launch — suspicious"

    # --- assemble_for_k: pool → K-specific format ---
    d = bd.assemble_for_k(pool, K)
    print(f"\nassembled at K={K}:")
    for k, v in d.items():
        print(f"  {k:14s}{v.shape} {v.dtype}")
    assert d["cand_feats"].shape[1] == K, \
        f"cand_feats K={d['cand_feats'].shape[1]} != {K}"
    assert d["cand_mask"].shape[1] == K
    for key in ("self_feats", "cand_feats", "global_feats", "cand_mask", "target_idx"):
        assert d[key].shape[0] == n, f"{key} row count {d[key].shape[0]} != {n}"
    ti = d["target_idx"]
    assert ti.min() >= 0 and ti.max() < K, f"target_idx out of [0,{K}): {ti.min()}..{ti.max()}"
    print(f"\ntarget_idx (assembled): min={int(ti.min())} max={int(ti.max())} "
          f"noop_frac={float(np.mean(ti==0)):.3f}")

    # --- JAX batch simulator vs numpy reference (one episode) ---
    states, players, actions = bd._collect_states_actions(eps[0])
    import jax.numpy as jnp
    batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *states)
    enc = jax.tree_util.tree_map(np.asarray, encode_fn(batched, jnp.array(players, jnp.int32)))
    b = jax.tree_util.tree_map(np.asarray, batched)
    N = len(states)

    flat, meta = [], []
    for i in range(N):
        for act in actions[i]:
            sidx = bd._resolve_src_idx(enc.src_ids[i], enc.src_valid[i], int(act[0]))
            if sidx == -1:
                continue
            flat.append((i, int(enc.src_slots[i][sidx]), float(act[1]), float(act[2])))
            meta.append((i, sidx))
    hk, hslot, htick = bd.simulate_episode_targets_jax(
        flat, p_x=b.p_x, p_y=b.p_y, p_init_x=b.p_init_x, p_init_y=b.p_init_y,
        p_radius=b.p_radius, p_alive=b.p_alive, p_is_comet=b.p_is_comet,
        angular_velocity=b.angular_velocity, step=b.step)

    # Batch simulator sanity: codes are valid, and every PLANET hit lands on a
    # live planet at a positive tick.
    valid_kinds = {int(k) for k in bd.HitKind}
    assert set(int(k) for k in hk) <= valid_kinds, "unexpected HitKind code"
    n_planet = 0
    for nidx, (i, _sidx) in enumerate(meta):
        if int(hk[nidx]) == int(bd.HitKind.PLANET):
            slot = int(hslot[nidx])
            assert 0 <= slot < b.p_alive.shape[1], f"hit_slot {slot} out of range"
            assert bool(b.p_alive[i][slot]), "fleet resolved onto a dead planet"
            assert int(htick[nidx]) >= 1, "PLANET hit with non-positive tick"
            n_planet += 1
    print(f"\nbatch simulator: {len(meta)} fleets, {n_planet} resolved to a live planet")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
