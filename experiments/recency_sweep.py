"""
Recency-weighting screening sweep (small/fast model).

Hypothesis: in a relative-rating (ELO/TrueSkill) population, the same avg_score on a
later date corresponds to a stronger player. So down-weighting older episodes by date
should make a BC policy match recent strong play better.

Test design (treats 2026-05-15 as "today"):
    train on  episodes/2026-05-01 .. 2026-05-14
    hold out  episodes/2026-05-15   (the recent target distribution)
    sweep     recency_half_life in {off, 14, 7, 3} days  (off = current behavior)
    metric    unweighted target-action accuracy / CE on the held-out 05-15 shards

A small model (hidden=128, K=8) is used for speed: it screens the *direction* and
*rough* optimum of the half-life. Re-confirm the winner at full scale (hidden=512, K=12).

Run (GPU node):
    uv run python -m experiments.recency_sweep
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from src_jax.accel import setup_accelerator
from src_jax.bc_data import find_shards
from src_jax.bc_train import evaluate_target_acc, train
from src_jax.config import TrainConfig

# --- experiment knobs --------------------------------------------------------
REF_DATE = "2026-05-15"                       # "today"
TRAIN_DATES = [f"2026-05-{d:02d}" for d in range(1, 15)]   # 05-01 .. 05-14
VAL_DATE = "2026-05-15"
HALF_LIVES = [0.0, 14.0, 7.0, 3.0]            # 0.0 = recency off (baseline)

HIDDEN_SIZE = 128
# K=12 (not 8): candidate_count barely affects speed (only widens the [N,K,D] cand
# axis), and assemble_for_k currently over-allocates quotas at K=8 (_k_quota clamp
# bug). 12 also matches full-training K, so the sweep transfers cleanly.
CANDIDATE_COUNT = 12
EPOCHS = 2
BATCH_SIZE = 2048
LR = 3e-4
AUGMENT = False                                # off for speed; same for every run
MAX_SHARDS_PER_DATE = 4                        # bound compute; every date stays represented
SEED = 42
OUT_ROOT = "artifacts/bc/recency_sweep"
# -----------------------------------------------------------------------------


def _make_cfg(half_life: float) -> TrainConfig:
    cfg = TrainConfig()
    cfg.seed = SEED
    cfg.model.hidden_size = HIDDEN_SIZE
    cfg.env.candidate_count = CANDIDATE_COUNT
    bc = cfg.bc
    bc.use_shard = True
    bc.epochs = EPOCHS
    bc.batch_size = BATCH_SIZE
    bc.lr = LR
    bc.augment = AUGMENT
    bc.max_shards_per_date = MAX_SHARDS_PER_DATE
    bc.recency_half_life = half_life
    bc.recency_ref_date = REF_DATE
    bc.out_dir = OUT_ROOT
    bc.run_name = f"hl_{'off' if half_life <= 0 else int(half_life)}"
    return cfg


def main() -> None:
    setup_accelerator()

    train_dirs = [Path("episodes") / d for d in TRAIN_DATES]
    val_shards = find_shards([Path("episodes") / VAL_DATE])
    print(f"Train dirs: {TRAIN_DATES[0]}..{TRAIN_DATES[-1]} ({len(train_dirs)} days)")
    print(f"Val (held-out): {VAL_DATE}  ({len(val_shards)} shards)")
    print(f"Model: hidden={HIDDEN_SIZE} K={CANDIDATE_COUNT} epochs={EPOCHS} augment={AUGMENT}")

    results = []
    for hl in HALF_LIVES:
        tag = "off" if hl <= 0 else f"{int(hl)}d"
        print("\n" + "=" * 70)
        print(f"### recency_half_life = {tag}")
        print("=" * 70, flush=True)
        cfg = _make_cfg(hl)
        t0 = time.time()
        state = train(train_dirs, cfg, use_wandb=False, recursive=False)
        # Evaluate at the run's K (held-out 05-15)
        metrics = evaluate_target_acc(state, val_shards, cfg)
        dt = time.time() - t0
        row = {"half_life": tag, "val_acc": metrics["acc"], "val_ce": metrics["ce"],
               "val_n": metrics["n"], "train_s": round(dt, 1)}
        results.append(row)
        print(f"\n[result] half_life={tag}  val_acc={metrics['acc']:.4f}  "
              f"val_ce={metrics['ce']:.4f}  n={metrics['n']}  ({dt:.0f}s)", flush=True)

    # Summary table
    base = next((r for r in results if r["half_life"] == "off"), None)
    print("\n" + "=" * 70)
    print("RECENCY SWEEP SUMMARY  (held-out 05-15 target-action accuracy)")
    print("=" * 70)
    print(f"{'half_life':>10} {'val_acc':>9} {'Δacc vs off':>12} {'val_ce':>9} {'train_s':>9}")
    for r in results:
        dacc = "" if base is None else f"{r['val_acc'] - base['val_acc']:+.4f}"
        print(f"{r['half_life']:>10} {r['val_acc']:>9.4f} {dacc:>12} {r['val_ce']:>9.4f} {r['train_s']:>9.1f}")

    out_csv = Path(OUT_ROOT) / "recency_sweep_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["half_life", "val_acc", "val_ce", "val_n", "train_s"])
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
