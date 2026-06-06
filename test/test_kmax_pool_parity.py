"""Parity test: K-pool shard re-sliced to K == live encoder at K.

The K-pool shard format (bc_data.POOL_K) stores the candidate pool once at K_max and
re-slices it offline to any K<=K_max via assemble_for_k(). This is only valid if the
candidate selection is set-monotone in K (the K-pool is a superset of the smaller-K
pool). This test verifies that end-to-end on real episodes: for each training row it
compares assemble_for_k(pool@POOL_K, K) against the legacy fixed-K pipeline built from
a live _make_env_encoder(K) — the exact thing inference uses.

Asserts, per row (same state / source / launch order):
  * target_idx   — byte-identical
  * cand_mask    — byte-identical
  * cand_feats   — allclose (same planet ⇒ same features, modulo float)

Run on a compute node (the padded encoder is too heavy for the login node):
    python -m test.test_kmax_pool_parity --episodes_dir episodes/2026-05-01 --n 8 --K 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax

from src_jax.config import TrainConfig
from src_jax.features import _make_env_encoder
import src_jax.bc_data as bd


def _encode(ep, encode_fn):
    """Run an encoder over one episode's launch states; return (enc_np, batched_np, actions, N)."""
    states, players, actions = bd._collect_states_actions(ep)
    if not states:
        return None
    N = len(states)
    pad_n = bd._pad_size_for(N) - N
    all_states = states + [states[-1]] * pad_n
    all_players = players + [players[-1]] * pad_n
    import jax.numpy as jnp
    batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *all_states)
    enc = encode_fn(batched, jnp.array(all_players, dtype=jnp.int32))
    enc_np = jax.tree_util.tree_map(np.asarray, enc)
    batched_np = jax.tree_util.tree_map(np.asarray, batched)
    return enc_np, batched_np, actions, N


def _reference_fixed_k(enc_np, batched_np, actions, N):
    """Fixed-K sample dict from a live K-encoder via _episode_target_slots + assemble_for_k."""
    K = enc_np.cand_feats.shape[1]
    labels = bd._episode_target_slots(
        src_ids=enc_np.src_ids, src_valid=enc_np.src_valid, src_slots=enc_np.src_slots,
        p_x=batched_np.p_x, p_y=batched_np.p_y,
        p_init_x=batched_np.p_init_x, p_init_y=batched_np.p_init_y,
        p_radius=batched_np.p_radius, p_alive=batched_np.p_alive,
        p_is_comet=batched_np.p_is_comet,
        angular_velocity=batched_np.angular_velocity, step=batched_np.step,
        batch_actions=actions, N=N)
    si, sj, tslot = bd._flatten_labels(labels, enc_np.src_valid, N, noop=-1)
    pool = {
        "self_feats":   enc_np.self_feats[si, sj].astype(np.float32),
        "global_feats": enc_np.global_feat[si].astype(np.float32),
        "pool_feats":   enc_np.cand_feats[si, sj].astype(np.float32),
        "pool_slot":    enc_np.cand_slots_safe[si, sj].astype(np.int32),
        "pool_cat":     enc_np.cand_cat[si, sj].astype(np.int8),
        "pool_dist":    enc_np.cand_dist[si, sj].astype(np.float32),
        "pool_mask":    enc_np.cand_mask[si, sj],
        "target_slot":  tslot.astype(np.int32),
    }
    d = bd.assemble_for_k(pool, K)
    return {
        "cand_feats": d["cand_feats"],
        "cand_mask":  d["cand_mask"],
        "target_idx": d["target_idx"],
    }


def _pool_assembled(enc_np, batched_np, actions, N, K):
    """K-pool sample dict (built at POOL_K) re-sliced to K via assemble_for_k."""
    labels = bd._episode_target_slots(
        src_ids=enc_np.src_ids, src_valid=enc_np.src_valid, src_slots=enc_np.src_slots,
        p_x=batched_np.p_x, p_y=batched_np.p_y,
        p_init_x=batched_np.p_init_x, p_init_y=batched_np.p_init_y,
        p_radius=batched_np.p_radius, p_alive=batched_np.p_alive,
        p_is_comet=batched_np.p_is_comet,
        angular_velocity=batched_np.angular_velocity, step=batched_np.step,
        batch_actions=actions, N=N)
    si, sj, tslot = bd._flatten_labels(labels, enc_np.src_valid, N, noop=-1)
    pool = {
        "self_feats":   enc_np.self_feats[si, sj].astype(np.float32),
        "global_feats": enc_np.global_feat[si].astype(np.float32),
        "pool_feats":   enc_np.cand_feats[si, sj].astype(np.float32),
        "pool_slot":    enc_np.cand_slots_safe[si, sj].astype(np.int32),
        "pool_cat":     enc_np.cand_cat[si, sj].astype(np.int8),
        "pool_dist":    enc_np.cand_dist[si, sj].astype(np.float32),
        "pool_mask":    enc_np.cand_mask[si, sj],
        "target_slot":  tslot.astype(np.int32),
    }
    return bd.assemble_for_k(pool, K)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-05-01")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--K", type=int, default=8, help="training K to re-slice to (<= POOL_K)")
    args = ap.parse_args()

    print(f"jax devices: {jax.devices()}")
    cfg = TrainConfig()
    assert args.K <= bd.POOL_K, f"--K {args.K} must be <= POOL_K {bd.POOL_K}"

    enc_pool = _make_env_encoder(bd.POOL_K, cfg.env)
    enc_k    = _make_env_encoder(args.K, cfg.env)

    eps = []
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        if isinstance(ep, dict) and "steps" in ep:
            eps.append(ep)
        if len(eps) >= args.n:
            break
    assert eps, f"no episodes in {args.episodes_dir}"
    print(f"loaded {len(eps)} episodes; POOL_K={bd.POOL_K}, comparing re-slice K={args.K}\n")

    n_rows = ti_mismatch = mask_mismatch = feat_mismatch = 0
    max_feat_err = 0.0
    for ei, ep in enumerate(eps):
        pool_enc = _encode(ep, enc_pool)
        k_enc = _encode(ep, enc_k)
        if pool_enc is None or k_enc is None:
            continue
        got = _pool_assembled(*pool_enc, args.K)
        ref = _reference_fixed_k(*k_enc)

        assert got["target_idx"].shape == ref["target_idx"].shape, (
            f"row count differs ep{ei}: {got['target_idx'].shape} vs {ref['target_idx'].shape}")
        m = got["target_idx"].shape[0]
        n_rows += m

        ti_bad = int(np.sum(got["target_idx"] != ref["target_idx"]))
        mask_bad = int(np.sum(got["cand_mask"] != ref["cand_mask"]))
        ferr = np.abs(got["cand_feats"] - ref["cand_feats"])
        feat_bad = int(np.sum(ferr > 1e-4))
        max_feat_err = max(max_feat_err, float(ferr.max()) if ferr.size else 0.0)

        ti_mismatch += ti_bad
        mask_mismatch += mask_bad
        feat_mismatch += feat_bad
        flag = "" if (ti_bad or mask_bad or feat_bad) else "ok"
        print(f"ep{ei}: rows={m:5d}  target_idx_diff={ti_bad}  "
              f"mask_diff={mask_bad}  feat_diff(>1e-4)={feat_bad}  {flag}")

    print(f"\nTOTAL rows={n_rows}  target_idx_mismatch={ti_mismatch}  "
          f"cand_mask_mismatch={mask_mismatch}  feat_mismatch={feat_mismatch}  "
          f"max_feat_err={max_feat_err:.2e}")

    ok = (ti_mismatch == 0 and mask_mismatch == 0 and feat_mismatch == 0)
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
