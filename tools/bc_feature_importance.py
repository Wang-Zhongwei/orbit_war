#!/usr/bin/env python3
"""Permutation feature-importance for a trained BC policy.

For each *candidate* feature index, shuffle that one feature-plane across
samples in the held-out shards and measure how much top-1 target accuracy
drops (and cross-entropy rises). A large drop => the policy relies on that
feature; ~0 drop => the feature is (currently) useless to this model.

This does NOT retrain. It loads a BC checkpoint (the pickle written by
save_bc_checkpoint: {"params", "opt_state", "config"}) and reuses the exact
eval forward from bc_train (_bc_eval_batch).

Usage
-----
  python -m tools.bc_feature_importance \
      --ckpt  runs_bc/<run_name>/ckpt_last.pkl \
      --shards <dir-or-glob-of-held-out-.npz> \
      --repeats 3 \
      [--config configs/bc_june.yaml]     # only needed if ckpt has no cfg
      [--features 12-29]                   # subset; default = all candidate feats

Notes
-----
* Permutation (not zeroing) is used: zero is out-of-distribution for several
  of these planes (ETA, garrison, would-be-mine), which would over-state
  importance. Permutation keeps each feature's marginal distribution intact
  but destroys its correlation with the expert action.
* Use a HELD-OUT shard set (not training shards) for an honest read.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from glob import glob
from pathlib import Path

# Allow running as a plain file (`python tools/bc_feature_importance.py`):
# put the repo root (this file's parent's parent) on sys.path so `src_jax` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src_jax.config import TrainConfig
from src_jax.policies import build_policy
from src_jax.ppo import TrainState
from src_jax.bc_train import _bc_eval_batch
from src_jax.bc_data import is_pool_shard, assemble_for_k

# Candidate feature names (must mirror src_jax/features.py cand_feats stack order).
CAND_FEATURE_NAMES = [
    "bias",                  # 0
    "is_neutral",            # 1
    "is_friendly",           # 2
    "is_enemy",              # 3
    "delta_x",               # 4
    "delta_y",               # 5
    "tgt_dist",              # 6
    "eta_ticks",             # 7
    "tgt_garrison",          # 8
    "tgt_production",        # 9
    "tgt_orbit_r",           # 10
    "tgt_is_rotating",       # 11
    "my_inbound_5",          # 12
    "opp_inbound_5",         # 13
    "my_inbound_10",         # 14
    "opp_inbound_10",        # 15
    "my_inbound_15",         # 16
    "opp_inbound_15",        # 17
    "my_inbound_30",         # 18
    "opp_inbound_30",        # 19
    "my_nearby_5",           # 20
    "enemy_nearby_5",        # 21
    "my_nearby_10",          # 22
    "enemy_nearby_10",       # 23
    "my_nearby_15",          # 24
    "enemy_nearby_15",       # 25
    "would_be_mine_5",       # 26
    "would_be_mine_10",      # 27
    "would_be_mine_15",      # 28
    "would_be_mine_final",   # 29
]

# Global feature names (mirror src_jax/features.py global_feat stack order).
# The june_h128 ckpt is 12-dim (no [12] n_total_planets); newer 13-dim ckpts add it.
GLOBAL_FEATURE_NAMES = [
    "game_time_frac",        # 0
    "is_4player",            # 1
    "my_planet_frac",        # 2
    "enemy_planet_frac",     # 3
    "neutral_planet_frac",   # 4
    "my_planet_ships",       # 5
    "enemy_planet_ships",    # 6
    "my_fleet_ships",        # 7
    "enemy_fleet_ships",     # 8
    "my_production",         # 9
    "enemy_production",      # 10
    "angular_velocity",      # 11
    "n_total_planets",       # 12
]


def _resolve_shards(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        return sorted(p.glob("*.npz"))
    return sorted(Path(x) for x in glob(spec))


def _load_state(ckpt_path: Path, cfg_override: TrainConfig | None) -> tuple[TrainState, TrainConfig]:
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = cfg_override if cfg_override is not None else ckpt.get("config")
    if cfg is None:
        raise SystemExit("Checkpoint has no embedded config; pass --config.")
    policy = build_policy(cfg)
    # Eval-only: a no-op optimizer satisfies TrainState.create (we never step).
    state = TrainState.create(apply_fn=policy.apply, params=ckpt["params"],
                              tx=optax.set_to_zero())
    return state, cfg


def _load_eval_arrays(shard_paths: list[Path], cfg: TrainConfig):
    """Concatenate held-out shards into one set of arrays (assembling pool shards)."""
    sfs, cfs, gfs, cms, tis = [], [], [], [], []
    for p in shard_paths:
        data = np.load(p)
        if is_pool_shard(data):
            data = assemble_for_k(data, cfg.env.candidate_count)
        sfs.append(np.asarray(data["self_feats"]))
        cfs.append(np.asarray(data["cand_feats"]))
        gfs.append(np.asarray(data["global_feats"]))
        cms.append(np.asarray(data["cand_mask"]))
        tis.append(np.asarray(data["target_idx"]))
    return (np.concatenate(sfs), np.concatenate(cfs), np.concatenate(gfs),
            np.concatenate(cms), np.concatenate(tis))


def _eval(state, sf, cf, gf, cm, ti, batch_size=8192) -> dict:
    N = ti.shape[0]
    tot_c = 0.0
    tot_ce = 0.0
    sf = jnp.asarray(sf); cf = jnp.asarray(cf); gf = jnp.asarray(gf)
    cm = jnp.asarray(cm); ti = jnp.asarray(ti)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        c, ce = _bc_eval_batch(state, sf[s:e], cf[s:e], gf[s:e], cm[s:e], ti[s:e])
        tot_c += float(c); tot_ce += float(ce)
    return {"acc": tot_c / N, "ce": tot_ce / N}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--shards", required=True, help="Held-out .npz dir or glob.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Override config (only if ckpt has none).")
    ap.add_argument("--space", choices=["cand", "global"], default="cand",
                    help="Which feature space to permute: candidate (per-target) "
                         "or global (per-state). Default: cand.")
    ap.add_argument("--features", default=None,
                    help="Index range/list to test, e.g. '12-29' or '7,8,18'. "
                         "Default: all features in the chosen space.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Permutation repeats (averaged) for a stable estimate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()

    cfg_override = None
    if args.config is not None:
        from src_jax.config import load_train_config
        cfg_override = load_train_config(args.config)

    state, cfg = _load_state(args.ckpt, cfg_override)
    shard_paths = _resolve_shards(args.shards)
    if not shard_paths:
        raise SystemExit(f"No shards matched: {args.shards}")
    sf, cf, gf, cm, ti = _load_eval_arrays(shard_paths, cfg)
    N, K, F = cf.shape
    print(f"Loaded {N:,} samples from {len(shard_paths)} shard(s); "
          f"cand_feats={cf.shape} global_feats={gf.shape}")

    # Target array + feature-name table for the chosen space.
    if args.space == "global":
        arr = gf
        names = GLOBAL_FEATURE_NAMES
        n_feat = gf.shape[1]
    else:
        arr = cf
        names = CAND_FEATURE_NAMES
        n_feat = F

    # Which feature indices to test.
    if args.features:
        idxs: list[int] = []
        for part in args.features.split(","):
            if "-" in part:
                a, b = part.split("-")
                idxs.extend(range(int(a), int(b) + 1))
            else:
                idxs.append(int(part))
    else:
        idxs = list(range(n_feat))

    base = _eval(state, sf, cf, gf, cm, ti, args.batch_size)
    print(f"\nSpace={args.space}  Baseline: acc={base['acc']:.4f}  ce={base['ce']:.4f}\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for f in idxs:
        d_acc_runs, d_ce_runs = [], []
        for _ in range(args.repeats):
            perm = rng.permutation(N)
            arr_p = arr.copy()
            if args.space == "global":
                arr_p[:, f] = arr[perm, f]            # shuffle global feature across states
                m = _eval(state, sf, cf, arr_p, cm, ti, args.batch_size)
            else:
                arr_p[:, :, f] = arr[perm, :, f]      # shuffle candidate plane across samples
                m = _eval(state, sf, arr_p, gf, cm, ti, args.batch_size)
            d_acc_runs.append(base["acc"] - m["acc"])  # drop in accuracy
            d_ce_runs.append(m["ce"] - base["ce"])     # rise in CE
        d_acc = float(np.mean(d_acc_runs)); d_acc_sd = float(np.std(d_acc_runs))
        d_ce = float(np.mean(d_ce_runs))
        name = names[f] if f < len(names) else f"feat_{f}"
        rows.append((f, name, d_acc, d_acc_sd, d_ce))
        print(f"  [{f:2d}] {name:<20s} Δacc=-{d_acc:+.4f} (±{d_acc_sd:.4f})  ΔCE=+{d_ce:+.4f}")

    print("\n=== Ranked by accuracy drop (most important first) ===")
    for f, name, d_acc, d_acc_sd, d_ce in sorted(rows, key=lambda r: -r[2]):
        bar = "#" * int(max(0, d_acc) * 500)
        print(f"  [{f:2d}] {name:<20s} {d_acc:+.4f}  {bar}")


if __name__ == "__main__":
    main()
