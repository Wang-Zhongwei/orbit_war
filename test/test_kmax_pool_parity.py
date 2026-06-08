"""Parity test: K-pool shard re-sliced to K == the live encoder at K.

The K-pool shard format (bc_data.POOL_K) stores the candidate pool once at K_max and
re-slices it offline to any K<=K_max via assemble_for_k(). This is only valid if that
offline selection reproduces what the *live* encoder (_make_env_encoder at K) does —
the exact candidate set inference uses. This test checks that directly:

  * encode each state with the POOL_K encoder -> build the stored-pool dict,
  * assemble_for_k(pool, K),
  * compare cand_feats / cand_mask against the live K-encoder's own output, per
    (state, valid source).

A tiny residual is expected and tolerated: offline ranks are computed over the stored
K-pool rather than the whole board, so at the K-1 selection boundary a near-equidistant
candidate can be swapped (set-monotonicity edge). We assert the per-row disagreement
rate stays below TOL_ROW_RATE — this still catches gross breakage (e.g. a quota scheme
that disagrees on ~20% of candidate slots) while allowing the boundary noise.

Run on a compute node (the padded encoder is heavy for the login node):
    python -m test.test_kmax_pool_parity --episodes_dir episodes/2026-05-01 --n 4 --K 8
    python -m test.test_kmax_pool_parity --episodes_dir episodes/2026-05-01 --n 4   # K in {8,12,24}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from src_jax.config import TrainConfig
from src_jax.features import _make_env_encoder
import src_jax.bc_data as bd

# Max fraction of (state, source) rows allowed to disagree with the live encoder.
TOL_ROW_RATE = 0.02
# Feature closeness on a matched candidate slot.
FEAT_ATOL = 1e-4


def _encode(ep, encode_fn):
    """Run an encoder over one episode's launch states; return (enc_np, batched_np, N)."""
    states, players, _actions = bd._collect_states_actions(ep)
    if not states:
        return None
    N = len(states)
    pad_n = bd._pad_size_for(N) - N
    all_states = states + [states[-1]] * pad_n
    all_players = players + [players[-1]] * pad_n
    batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *all_states)
    enc = encode_fn(batched, jnp.array(all_players, dtype=jnp.int32))
    return (jax.tree_util.tree_map(np.asarray, enc),
            jax.tree_util.tree_map(np.asarray, batched), N)


def _build_pool(ep_np, bp, ii, jj):
    """Stored-pool dict (as a shard would hold it) for the selected (state, source) rows."""
    return {
        "self_feats":   ep_np.self_feats[ii, jj].astype(np.float32),
        "global_feats": ep_np.global_feat[ii].astype(np.float32),
        "pool_feats":   ep_np.cand_feats[ii, jj].astype(np.float32),
        "pool_slot":    ep_np.cand_slots_safe[ii, jj].astype(np.int32),
        "pool_cat":     ep_np.cand_cat[ii, jj].astype(np.int8),
        "pool_dist":    ep_np.cand_dist[ii, jj].astype(np.float32),
        "pool_mask":    ep_np.cand_mask[ii, jj],
        "step":         bp.step[ii].astype(np.int32),
        "target_slot":  np.full(ii.shape[0], -1, np.int32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-05-01")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--K", type=int, default=None,
                    help="single training K to check (default: 8, 12, POOL_K)")
    args = ap.parse_args()

    print(f"jax devices: {jax.devices()}")
    cfg = TrainConfig()
    Ks = [args.K] if args.K is not None else [8, 12, bd.POOL_K]
    for K in Ks:
        assert K <= bd.POOL_K, f"K {K} must be <= POOL_K {bd.POOL_K}"

    enc_pool = _make_env_encoder(bd.POOL_K, cfg.env)

    eps = []
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        if isinstance(ep, dict) and "steps" in ep:
            eps.append(ep)
        if len(eps) >= args.n:
            break
    assert eps, f"no episodes in {args.episodes_dir}"
    print(f"loaded {len(eps)} episodes; POOL_K={bd.POOL_K}\n")

    pool_encs = [_encode(ep, enc_pool) for ep in eps]

    worst_rate = 0.0
    for K in Ks:
        enc_k = _make_env_encoder(K, cfg.env)
        tot_rows = diff_rows = 0
        max_feat_err = 0.0
        print(f"===== K={K} =====")
        for ei, (ep, pe) in enumerate(zip(eps, pool_encs)):
            ke = _encode(ep, enc_k)
            if pe is None or ke is None:
                continue
            ep_np, bp, N = pe
            ek_np, _, _ = ke
            ii, jj = np.where(np.asarray(ep_np.src_valid)[:N])
            got = bd.assemble_for_k(_build_pool(ep_np, bp, ii, jj), K)
            ref_feats = ek_np.cand_feats[ii, jj].astype(np.float32)   # [M, K, D]
            ref_mask = ek_np.cand_mask[ii, jj]                        # [M, K]

            mask_bad = (got["cand_mask"] != ref_mask).any(axis=1)     # [M]
            valid = ref_mask & got["cand_mask"]
            ferr = np.where(valid[..., None], np.abs(got["cand_feats"] - ref_feats), 0.0)
            feat_bad = (ferr > FEAT_ATOL).any(axis=(1, 2))            # [M]
            row_bad = int((mask_bad | feat_bad).sum())
            M = ii.shape[0]
            tot_rows += M
            diff_rows += row_bad
            max_feat_err = max(max_feat_err, float(ferr.max()) if ferr.size else 0.0)
            print(f"  ep{ei}: rows={M:5d}  diff_rows={row_bad:4d}  "
                  f"({row_bad / max(M,1):.3%})  max_feat_err={float(ferr.max()) if ferr.size else 0.0:.2e}")
        rate = diff_rows / max(tot_rows, 1)
        worst_rate = max(worst_rate, rate)
        print(f"  K={K} TOTAL: rows={tot_rows}  diff_rows={diff_rows}  rate={rate:.3%}  "
              f"max_feat_err={max_feat_err:.2e}\n")

    ok = worst_rate <= TOL_ROW_RATE
    print(f"worst per-row disagreement rate = {worst_rate:.3%}  (tolerance {TOL_ROW_RATE:.1%})")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
