"""Verify the PPO eval can load the JAX opponents (sub_v6/v7/v8/v81) against the
pinned v7arch src_jax, while the trainee runs on the live (26-dim) tree.

Reproduces the exact path that failed in job 48757: an eval worker builds the
trainee agent from the LIVE src_jax first, then imports src.evaluation (which
must load the 30-dim JAX submissions without colliding on the src_jax module
slot) and plays real matches. Exits non-zero on any crash or missing opponent.

Run:  .venv/bin/python tools/test_eval_opponents.py [ckpt.pkl]
"""
from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Import the LIVE src_jax FIRST, mirroring eval_mp._init_worker building the
# trainee before src.evaluation is ever touched. This is what made the live
# 26-dim tree win the sys.modules["src_jax"] slot and shadow the submissions.
import src_jax.features as _live_feat
from src_jax import eval_mp

CKPT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/bc/bc_train_v9/ckpt_000001_shard1500.pkl"
OPPONENTS = ["sub_v52", "sub_v6", "sub_v7", "sub_v8", "sub_v81"]


def main() -> int:
    print(f"live src_jax: OUT={_live_feat.MAX_TICKS_OUT} IN={_live_feat.MAX_TICKS_IN}", flush=True)
    print(f"checkpoint  : {CKPT}", flush=True)

    with open(CKPT, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    # TrainConfig is a (non-frozen) dataclass, but use replace to be safe.
    try:
        cfg = dataclasses.replace(cfg, eval_opponents=list(OPPONENTS), eval_rounds=2)
    except Exception:
        cfg.eval_opponents = list(OPPONENTS)
        cfg.eval_rounds = 2

    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])

    print(f"running run_eval_mp vs {OPPONENTS} (2 rounds, 4 workers) …", flush=True)
    res = eval_mp.run_eval_mp(params, cfg, base_seed=1000, workers=4)
    print(f"win rates: {res}", flush=True)

    missing = [o for o in OPPONENTS if o not in res]
    if missing:
        print(f"FAIL: opponents missing from results (skipped/crashed): {missing}", flush=True)
        return 1
    print("OK: trainee (live 26-dim) played every opponent — incl. the 30-dim "
          "JAX subs — with no ScopeParamShapeError.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
