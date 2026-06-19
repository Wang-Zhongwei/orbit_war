"""
Behavior-cloning pretraining: imitate top players by minimizing cross-entropy.

Usage:
    # Formal training — config file + optional CLI overrides (CLI wins):
    python -m src_jax.bc_train --config configs/templates/bc_default.yaml \\
                                --episode_dirs episodes/2026-05-24 \\
                                --top_k_episodes 200 --out_dir ./artifacts/bc/run1

    # Smoke test — dataclass defaults + CLI overrides, no config file needed:
    python -m src_jax.bc_train --episode_dirs episodes/2026-05-24 \\
                                --top_k_episodes 5 --epochs 1

    # Recursive mode — train from every episode subdirectory under a parent:
    python -m src_jax.bc_train --episode_dirs episodes --recursive

Runtime flags (accepted in both modes):
    --use_shard / --no_use_shard   Use pre-built shards (default: stream from zip)
    --no_wandb               Disable W&B logging

Loss:
    CE(target_logits, expert_target)
    No ships head, no value loss, no entropy bonus — pure supervised imitation.

The saved checkpoint is compatible with train.py --resume so you can warm-start PPO.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import pickle
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .augment import DEFAULT_AUG_IDS, expand_with_augmentations
from .bc_data import (
    _episode_avg_score,
    _iter_episode_jsons,
    _load_avg_score_map,
    assemble_for_k,
    find_shards,
    is_pool_shard,
    is_shard_dir_name,
    shard_dir_name,
    shard_episode_counts,
)
from .config import TrainConfig, load_train_config
from .features import _make_env_encoder
from .policies import build_policy, init_params
from .ppo import TrainState


# ---------------------------------------------------------------------------
# Multiprocessing worker: CPU-JAX encoding, returns picklable numpy dict
# ---------------------------------------------------------------------------

# Per-worker globals set once by _worker_init
_g_encode_fn = None


def _worker_init(env_cfg) -> None:
    """Initialize CPU-only JAX and the feature encoder in each worker process."""
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    global _g_encode_fn
    from src_jax.features import _make_env_encoder as _mee  # absolute import for spawn
    _g_encode_fn = _mee(env_cfg.candidate_count, env_cfg)


def _encode_episode(task) -> "dict | None":
    """Worker task: encode (episode, avg_score) → numpy dict (all players; reward + skill tagged)."""
    from src_jax.bc_data import episode_to_samples as _eps  # absolute

    episode, avg_score = task
    if not isinstance(episode, dict) or "steps" not in episode:
        return None

    return _eps(episode, _g_encode_fn, avg_score=avg_score)


# ---------------------------------------------------------------------------
# BC loss — single JIT-compiled minibatch update
# ---------------------------------------------------------------------------

@jax.jit
def _bc_step(
    state: TrainState,
    # S = source rows: a flat BC batch of (state, source) decisions, each with its
    # own global. (Same leading axis the policy calls S.)
    sf: jax.Array,   # [S, self_dim]
    cf: jax.Array,   # [S, K, cand_dim]
    gf: jax.Array,   # [S, global_dim]
    cm: jax.Array,   # [S, K]
    ti: jax.Array,   # [S]  int32  target index
    w: jax.Array,    # [S]  float32 per-sample loss weight (win/loss)
) -> tuple[TrainState, jax.Array, jax.Array]:
    def loss_fn(params):
        sv  = jnp.ones(sf.shape[0], dtype=jnp.bool_)
        out = state.apply_fn(params, sf, cf, gf, cm, sv)
        S   = ti.shape[0]

        tgt_lp   = jax.nn.log_softmax(out.target_logits, axis=-1)
        ce       = -tgt_lp[jnp.arange(S), ti]
        tgt_loss = (w * ce).sum() / (w.sum() + 1e-8)   # weighted CE; zero-weight rows ignored
        tgt_acc  = (jnp.argmax(out.target_logits, axis=-1) == ti).astype(jnp.float32).mean()

        return tgt_loss, tgt_acc

    (loss, tgt_acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, tgt_acc


@jax.jit
def _bc_eval_batch(
    state: TrainState,
    sf: jax.Array, cf: jax.Array, gf: jax.Array, cm: jax.Array, ti: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Eval-only forward: returns (sum_correct, sum_ce) over the batch (unweighted)."""
    sv  = jnp.ones(sf.shape[0], dtype=jnp.bool_)
    out = state.apply_fn(state.params, sf, cf, gf, cm, sv)
    S   = ti.shape[0]
    tgt_lp  = jax.nn.log_softmax(out.target_logits, axis=-1)
    ce_sum  = -tgt_lp[jnp.arange(S), ti].sum()
    correct = (jnp.argmax(out.target_logits, axis=-1) == ti).sum()
    return correct.astype(jnp.float32), ce_sum


def evaluate_target_acc(state, shard_paths, cfg, *, batch_size: int = 8192) -> dict:
    """Unweighted held-out target-action accuracy / CE over the given shards.

    No augmentation, no gradient — measures how well the policy matches the experts
    in `shard_paths` (the held-out target distribution). Returns {n, acc, ce}.
    """
    tot_n = 0
    tot_correct = 0.0
    tot_ce = 0.0
    for p in shard_paths:
        data = np.load(p)
        if is_pool_shard(data):
            data = assemble_for_k(data, cfg.env.candidate_count)
        N = int(data["target_idx"].shape[0])
        sf = jnp.asarray(data["self_feats"])
        cf = jnp.asarray(data["cand_feats"])
        gf = jnp.asarray(data["global_feats"])
        cm = jnp.asarray(data["cand_mask"])
        ti = jnp.asarray(data["target_idx"])
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            c, ce = _bc_eval_batch(state, sf[start:end], cf[start:end],
                                   gf[start:end], cm[start:end], ti[start:end])
            tot_correct += float(c)
            tot_ce += float(ce)
            tot_n += end - start
    acc = tot_correct / max(tot_n, 1)
    ce = tot_ce / max(tot_n, 1)
    return {"n": tot_n, "acc": acc, "ce": ce}


@dataclass
class _SkillNorm:
    """Maps a per-game ``avg_score`` to a bounded skill weight.

    weight = clip(exp((avg_score - median) / T), 1/R, R). T is chosen by
    _build_skill_norm so a reference-percentile (p95) game maps to exactly R.
    Because the BC loss is a weighted mean (sum(w·ce)/sum(w)), only *relative*
    weights matter, so centering at the median is cosmetic and the absolute
    scale is irrelevant — R and the clip band are what shape the gradients.
    """
    median: float
    T: float
    R: float

    def weights(self, avg_score) -> np.ndarray:
        s = np.asarray(avg_score, np.float64)
        # Clip the exponent to the band first — equivalent to clipping exp() to
        # [1/R, R] but avoids overflow warnings on extreme/NaN scores.
        z = np.clip((s - self.median) / self.T, -math.log(self.R), math.log(self.R))
        w = np.exp(z)
        # Games with no manifest match (NaN) get a neutral weight of 1.
        return np.where(np.isfinite(s), w, 1.0).astype(np.float32)


def _build_skill_norm(episodes_dirs, R, ref_pct: float = 95.0) -> "_SkillNorm | None":
    """Compute the skill-weight normalization from the manifest avg_score spread.

    T is set so a ref_pct (p95) game maps to weight R: T = (p95 - median)/ln(R).
    Returns None (skill weighting off) if R <= 1, no manifests are found, or the
    spread is degenerate.
    """
    if R is None or R <= 1.0:
        return None
    vals: list[float] = []
    for d in episodes_dirs:
        vals.extend(_load_avg_score_map(d).values())
    if not vals:
        print(f"  skill_clip_ratio={R} set but no manifest.csv found — skill weighting OFF", flush=True)
        return None
    arr = np.asarray(vals, np.float64)
    median = float(np.median(arr))
    p = float(np.percentile(arr, ref_pct))
    T = (p - median) / math.log(R)
    if not (T > 0.0):
        print(f"  skill weighting OFF: degenerate spread (median={median:.1f}, p{ref_pct:.0f}={p:.1f})",
              flush=True)
        return None
    print(f"  skill weighting ON: R={R}  median={median:.1f}  p{ref_pct:.0f}={p:.1f}  "
          f"T={T:.1f}  (p{ref_pct:.0f} game → weight {R}, clipped to [1/{R}, {R}])", flush=True)
    return _SkillNorm(median=median, T=T, R=float(R))


def _shard_date(path: Path) -> "date | None":
    """Parse the YYYY-MM-DD date from a shard path episodes/<date>/shards/shard_*.npz."""
    # shard file -> shards dir -> date dir
    name = path.parent.parent.name
    try:
        return datetime.strptime(name, "%Y-%m-%d").date()
    except ValueError:
        return None


def _build_recency(shard_paths, half_life: float, ref_date_str: str):
    """Return (ref_date, per-path multiplier dict) for recency weighting, or (None, None) if off.

    multiplier = 0.5 ** (days_old / half_life), days_old = (ref_date - shard_date).days,
    clamped at >= 0 so future-dated shards (days_old < 0) are not up-weighted past 1.0.
    """
    if half_life is None or half_life <= 0.0:
        return None, None
    dates = {p: _shard_date(p) for p in shard_paths}
    known = [d for d in dates.values() if d is not None]
    if not known:
        print(f"  recency_half_life={half_life} set but no dated shard paths — recency OFF", flush=True)
        return None, None
    if ref_date_str:
        ref = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    else:
        ref = max(known)
    mult = {}
    for p, d in dates.items():
        if d is None:
            mult[p] = 1.0  # undated shard: neutral
            continue
        days_old = max((ref - d).days, 0)
        mult[p] = float(0.5 ** (days_old / half_life))
    lo, hi = min(mult.values()), max(mult.values())
    print(f"  recency weighting ON: half_life={half_life}d  ref={ref}  "
          f"mult range [{lo:.3f}, {hi:.3f}] over {len(known)} dated shards", flush=True)
    return ref, mult


def _make_optimizer(bc, total_steps: int | None = None) -> optax.GradientTransformation:
    """Build the BC optimizer from cfg.bc stability knobs.

    - grad_clip > 0     -> global-norm clip before the optimizer
    - warmup_steps > 0  -> linear LR warmup 0->lr over warmup_steps
    - lr_schedule cosine-> after warmup, cosine-decay lr->lr_final over total_steps
                           (falls back to warmup+constant if total_steps is unknown)
    - lr_schedule const -> hold lr after warmup
    - weight_decay > 0  -> AdamW (decoupled decay); else plain Adam
    """
    warm = bc.warmup_steps if (bc.warmup_steps and bc.warmup_steps > 0) else 0
    if getattr(bc, "lr_schedule", "constant") == "cosine" and total_steps and total_steps > warm:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=bc.lr,
            warmup_steps=warm,
            decay_steps=total_steps,   # total horizon; cosine spans (total_steps - warm)
            end_value=bc.lr_final,
        )
    else:
        lr = optax.linear_schedule(0.0, bc.lr, warm) if warm else bc.lr
    core = (optax.adamw(lr, weight_decay=bc.weight_decay)
            if bc.weight_decay and bc.weight_decay > 0.0 else optax.adam(lr))
    stages = []
    if bc.grad_clip and bc.grad_clip > 0.0:
        stages.append(optax.clip_by_global_norm(bc.grad_clip))
    stages.append(core)
    return optax.chain(*stages)


def _estimate_total_steps(shard_paths, bc, aug_ids, K: int, epochs: int) -> int:
    """Estimate total optimizer steps for the cosine horizon.

    Peeks the first shard (post loser-drop + augmentation expansion) for a
    representative steps/shard, then scales by shard count × epochs. Shards are
    near-uniform in size, so this is within a few % — plenty for a cosine horizon.
    """
    data = np.load(shard_paths[0])
    if is_pool_shard(data):
        data = assemble_for_k(data, K)
    data = _drop_losers(data, bc)
    n = int(data["target_idx"].shape[0])
    n_aug = n * (len(aug_ids) if bc.augment else 1)
    steps_per_shard = max(1, n_aug // bc.batch_size)
    return steps_per_shard * len(shard_paths) * max(1, epochs)


def _drop_losers(data, bc) -> dict:
    """When loser_weight == 0, drop loser rows entirely (speed: they'd be zero-weighted anyway).

    The weighted-mean loss already ignores zero-weight rows, so this changes nothing
    numerically — it just stops wasting batch slots on samples that contribute nothing.
    Returns a plain dict (works on both NpzFile and dict inputs).
    """
    if bc.loser_weight != 0.0 or "reward" not in data:
        return data
    keep = np.asarray(data["reward"], np.float32) > 0.0
    if keep.all():
        return data
    n = keep.shape[0]
    return {k: (np.asarray(data[k])[keep] if np.asarray(data[k]).shape[:1] == (n,) else np.asarray(data[k]))
            for k in data.keys()}


def _sample_weights(data, bc, skill: "_SkillNorm | None" = None) -> np.ndarray:
    """Per-sample loss weight = win/loss weight × skill weight.

    Win/loss: winners get 1.0, losers get ``bc.loser_weight``. Legacy shards
    without a ``reward`` field fall back to uniform 1.0. Skill: if ``skill`` is set
    and the shard carries ``avg_score``, multiply by the bounded skill weight;
    shards encoded before that field simply skip it.
    """
    n = int(data["target_idx"].shape[0])
    if "reward" not in data:
        base = np.ones(n, np.float32)
    else:
        rwd = np.asarray(data["reward"], np.float32)
        base = np.where(rwd > 0.0, 1.0, float(bc.loser_weight)).astype(np.float32)
    if skill is not None and "avg_score" in data:
        base = base * skill.weights(data["avg_score"])
    return base.astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_bc_checkpoint(
    save_dir: Path,
    run_name: str,
    epoch: int,
    state: TrainState,
    cfg: TrainConfig,
    *,
    tag: str | None = None,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "update":    epoch,
        "params":    state.params,
        "opt_state": state.opt_state,
        "config":    cfg,
    }
    stem = tag if tag is not None else f"{epoch:06d}"
    # Save in the same format as train.py so PPO --resume works directly.
    for path in [run_dir / "ckpt_last.pkl", run_dir / f"ckpt_{stem}.pkl"]:
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
    print(f"  Checkpoint saved → {run_dir}/ckpt_{stem}.pkl")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes_dirs: list[Path],
    cfg: TrainConfig,
    *,
    use_wandb: bool = True,
    recursive: bool = False,
) -> TrainState:
    """Shard-based BC training loop. All hyperparameters are read from cfg.bc."""
    bc = cfg.bc
    aug_ids = DEFAULT_AUG_IDS
    episodes_dirs = _resolve_episode_dirs(
        episodes_dirs,
        include_shard_dirs=True,
        recursive=recursive,
        shard_tag=bc.shard_tag,
    )
    shard_paths = find_shards(episodes_dirs, bc.shard_tag)
    if bc.max_shards_per_date > 0:
        kept: list[Path] = []
        per_date: dict[str, int] = {}
        for p in shard_paths:
            day = p.parent.parent.name
            if per_date.get(day, 0) < bc.max_shards_per_date:
                kept.append(p)
                per_date[day] = per_date.get(day, 0) + 1
        print(f"  max_shards_per_date={bc.max_shards_per_date}: {len(shard_paths)} → {len(kept)} shards", flush=True)
        shard_paths = kept
    if bc.top_k_episodes is not None:
        counts = shard_episode_counts(shard_paths)
        cumulative = 0
        n_shards = len(shard_paths)
        for i, c in enumerate(counts):
            cumulative += c
            if cumulative >= bc.top_k_episodes:
                n_shards = i + 1
                break
        shard_paths = shard_paths[:n_shards]
    print(f"Found {len(shard_paths)} shards across {len(episodes_dirs)} date dir(s)"
          + (f" | top_k_episodes={bc.top_k_episodes}" if bc.top_k_episodes else "")
          + (f" | augment={aug_ids}" if bc.augment else ""))

    skill = _build_skill_norm(episodes_dirs, bc.skill_clip_ratio)
    _, recency_mult = _build_recency(shard_paths, bc.recency_half_life, bc.recency_ref_date)
    if bc.loser_weight == 0.0:
        print("  loser_weight=0 → loser rows dropped at load (winner-only training)", flush=True)

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    total_steps = None
    if bc.lr_schedule == "cosine":
        total_steps = bc.decay_steps if bc.decay_steps > 0 else _estimate_total_steps(
            shard_paths, bc, aug_ids, cfg.env.candidate_count, bc.epochs)
        print(f"  cosine LR: warmup={bc.warmup_steps} → peak={bc.lr} → "
              f"end={bc.lr_final} over ~{total_steps} steps"
              f"{' (auto)' if bc.decay_steps <= 0 else ''}", flush=True)
    tx = _make_optimizer(bc, total_steps)
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    if use_wandb:
        import wandb
        run = wandb.init(
            project="orbit-war",
            name=bc.run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "n_shards": len(shard_paths),
                "epochs": bc.epochs,
                "batch_size": bc.batch_size,
                "lr": bc.lr,
                "weight_decay": bc.weight_decay,
                "warmup_steps": bc.warmup_steps,
                "lr_schedule": bc.lr_schedule,
                "lr_final": bc.lr_final,
                "decay_steps": bc.decay_steps,
                "grad_clip": bc.grad_clip,
                "augment": bc.augment,
                "aug_ids": aug_ids if bc.augment else [],
                "candidate_count": cfg.env.candidate_count,
                "top_k_episodes": bc.top_k_episodes,
                "hidden_size": cfg.model.hidden_size,
                "loser_weight": bc.loser_weight,
                "skill_clip_ratio": bc.skill_clip_ratio,
                "recency_half_life": bc.recency_half_life,
                "recency_ref_date": bc.recency_ref_date,
            },
        )
        # Sidecar: let the (separate) eval job resume this exact W&B run.
        run_dir = Path(bc.out_dir) / bc.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "wandb_run_id.txt").write_text(run.id)

    print(f"Starting BC training: {bc.epochs} epochs, batch={bc.batch_size}", flush=True)
    print(f"Checkpoints → {Path(bc.out_dir) / bc.run_name}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    global_step = 0

    for epoch in range(1, bc.epochs + 1):
        shard_order = rng.permutation(len(shard_paths)).tolist()
        losses, tgt_accs = [], []
        t0 = time.time()
        n_steps_epoch = 0

        print(f"\nepoch {epoch}/{bc.epochs}  ({len(shard_paths)} shards)", flush=True)

        for si, shard_i in enumerate(shard_order):
            data = np.load(shard_paths[shard_i])
            # K-pool shards store a K_max candidate pool; re-slice to this run's K.
            # Legacy fixed-K shards already carry cand_feats/target_idx directly.
            if is_pool_shard(data):
                data = assemble_for_k(data, cfg.env.candidate_count)
            data = _drop_losers(data, bc)
            N    = data["target_idx"].shape[0]
            perm = rng.permutation(N)

            w_all = _sample_weights(data, bc, skill)
            if recency_mult is not None:
                w_all = w_all * recency_mult.get(shard_paths[shard_i], 1.0)
            if skill is not None and si == 0 and epoch == 1:
                ess = float(w_all.sum() ** 2 / (np.square(w_all).sum() + 1e-12))
                print(f"  skill weights: mean={w_all.mean():.3f} min={w_all.min():.3f} "
                      f"max={w_all.max():.3f}  ESS/N={ess / max(len(w_all), 1):.2f}", flush=True)

            sf = jnp.asarray(data["self_feats"][perm])
            cf = jnp.asarray(data["cand_feats"][perm])
            gf = jnp.asarray(data["global_feats"][perm])
            cm = jnp.asarray(data["cand_mask"][perm])
            ti = jnp.asarray(data["target_idx"][perm])
            w  = jnp.asarray(w_all[perm])

            # D4 augmentation: expand shard N→N*len(aug_ids) then re-shuffle.
            if bc.augment:
                sf, cf, gf, cm, ti = expand_with_augmentations(
                    sf, cf, gf, cm, ti, aug_ids
                )
                w = jnp.concatenate([w] * len(aug_ids), axis=0)  # tile to match expansion
                N = sf.shape[0]
                perm2 = jnp.asarray(rng.permutation(N))
                sf, cf, gf = sf[perm2], cf[perm2], gf[perm2]
                cm, ti, w = cm[perm2], ti[perm2], w[perm2]

            shard_losses: list[jax.Array] = []
            shard_tgt_a: list[jax.Array] = []
            for start in range(0, N - bc.batch_size + 1, bc.batch_size):
                end = start + bc.batch_size
                state, loss, tgt_a = _bc_step(
                    state,
                    sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                    ti[start:end], w[start:end],
                )
                shard_losses.append(loss)
                shard_tgt_a.append(tgt_a)
                global_step += 1
                n_steps_epoch += 1

            # Single GPU→CPU sync per shard instead of per step
            losses.append(float(jnp.mean(jnp.stack(shard_losses))))
            tgt_accs.append(float(jnp.mean(jnp.stack(shard_tgt_a))))

            elapsed = time.time() - t0
            steps_per_s = n_steps_epoch / elapsed if elapsed > 0 else 0
            print(
                f"  shard {si+1:3d}/{len(shard_paths)}"
                f"  loss={losses[-1]:.4f}"
                f"  tgt_acc={tgt_accs[-1]:.3f}"
                f"  steps={n_steps_epoch}"
                f"  {elapsed:.0f}s ({steps_per_s:.1f} steps/s)",
                flush=True,
            )

            if use_wandb:
                import wandb
                wandb.log(
                    {
                        "bc/loss": losses[-1],
                        "bc/tgt_acc": tgt_accs[-1],
                        "bc/epoch": epoch,
                        "bc/shard": si + 1,
                        "bc/steps_per_s": steps_per_s,
                    },
                    step=global_step,
                )

            if (si + 1) % cfg.bc.checkpoint_every == 0:
                save_bc_checkpoint(
                    Path(bc.out_dir), bc.run_name, epoch, state, cfg,
                    tag=f"{epoch:06d}_shard{si+1:04d}",
                )

        mean_loss    = float(np.mean(losses))
        mean_tgt_acc = float(np.mean(tgt_accs))
        elapsed      = time.time() - t0
        steps_per_s  = n_steps_epoch / elapsed if elapsed > 0 else 0
        print(
            f"epoch={epoch:3d}  loss={mean_loss:.4f}"
            f"  tgt_acc={mean_tgt_acc:.3f}"
            f"  steps={n_steps_epoch}  {elapsed:.1f}s ({steps_per_s:.1f} steps/s)"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "bc/epoch_loss": mean_loss,
                    "bc/epoch_tgt_acc": mean_tgt_acc,
                    "bc/epoch": epoch,
                },
                step=global_step,
            )

        save_bc_checkpoint(Path(bc.out_dir), bc.run_name, epoch, state, cfg)

    if use_wandb:
        import wandb
        wandb.finish()

    return state


def _resolve_episode_dirs(
    episodes_dirs: list[Path],
    *,
    include_shard_dirs: bool = False,
    recursive: bool = False,
    shard_tag: str = "",
) -> list[Path]:
    """Auto-expand dirs that contain no episode files directly but have dated subdirs.

    A subdir qualifies if it directly holds a ``*.zip``/``*.json`` episode file.
    With ``include_shard_dirs=True`` (shard-based training), a subdir also qualifies
    if it has an encoded ``shards``/``shards_<tag>`` directory (per shard_tag) — so
    already-encoded dates whose raw zip has since been deleted are still picked up.
    """
    def _qualifies(p: Path) -> bool:
        if any(p.glob("*.zip")) or any(p.glob("*.json")):
            return True
        return include_shard_dirs and (p / shard_dir_name(shard_tag)).is_dir()

    resolved: list[Path] = []
    for d in episodes_dirs:
        d = Path(d)
        if not d.is_dir():
            resolved.append(d)
            continue
        if recursive:
            subdirs = [
                child for child in sorted(d.rglob("*"))
                if child.is_dir() and not is_shard_dir_name(child.name) and _qualifies(child)
            ]
            if subdirs:
                print(f"  recursive-discovered under {d}: {[str(s) for s in subdirs]}", flush=True)
                resolved.extend(subdirs)
            else:
                resolved.append(d)   # keep as-is; will raise a clear error later
            continue
        if _qualifies(d):
            resolved.append(d)
        else:
            subdirs = sorted(
                child for child in d.iterdir()
                if child.is_dir() and _qualifies(child)
            )
            if subdirs:
                print(f"  auto-discovered: {[s.name for s in subdirs]}", flush=True)
                resolved.extend(subdirs)
            else:
                resolved.append(d)   # keep as-is; will raise a clear error later
    return resolved


# ---------------------------------------------------------------------------
# Streaming training loop (on-the-fly encode from zip, no shards)
# ---------------------------------------------------------------------------

def train_stream(
    episodes_dirs: list[Path],
    cfg: TrainConfig,
    *,
    use_wandb: bool = True,
    recursive: bool = False,
) -> TrainState:
    """Streaming BC training loop with parallel CPU encoding.

    Worker processes run CPU-JAX to encode episodes → numpy dicts while the
    main process trains on the GPU, overlapping the two via a sliding window.
    """
    bc = cfg.bc
    aug_ids = DEFAULT_AUG_IDS

    episodes_dirs = _resolve_episode_dirs(episodes_dirs, recursive=recursive)

    skill = _build_skill_norm(episodes_dirs, bc.skill_clip_ratio)
    if bc.loser_weight == 0.0:
        print("  loser_weight=0 → loser rows dropped before training (winner-only)", flush=True)
    # Per-directory episode_id -> avg_score maps for the skill join (read in the
    # main process; threaded to workers alongside each episode).
    dir_avg_maps = {d: _load_avg_score_map(d) for d in episodes_dirs}

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    # Episode mode can't cheaply pre-count steps; cosine needs an explicit decay_steps,
    # else _make_optimizer falls back to warmup+constant.
    tx = _make_optimizer(bc, bc.decay_steps if bc.decay_steps > 0 else None)
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    num_workers = bc.encode_workers if bc.encode_workers > 0 else max(1, multiprocessing.cpu_count() - 1)
    # inflight cap: enough futures to keep workers busy during one GPU drain
    _INFLIGHT = num_workers * 4

    if use_wandb:
        import wandb
        run = wandb.init(
            project="orbit-war",
            name=bc.run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "epochs": bc.epochs,
                "batch_size": bc.batch_size,
                "lr": bc.lr,
                "weight_decay": bc.weight_decay,
                "warmup_steps": bc.warmup_steps,
                "lr_schedule": bc.lr_schedule,
                "lr_final": bc.lr_final,
                "decay_steps": bc.decay_steps,
                "grad_clip": bc.grad_clip,
                "use_shard": False,
                "stream_buffer": bc.stream_buffer,
                "top_k_episodes": bc.top_k_episodes,
                "augment": bc.augment,
                "aug_ids": aug_ids if bc.augment else [],
                "candidate_count": cfg.env.candidate_count,
                "hidden_size": cfg.model.hidden_size,
                "encode_workers": num_workers,
                "loser_weight": bc.loser_weight,
                "skill_clip_ratio": bc.skill_clip_ratio,
            },
        )
        # Sidecar: let the (separate) eval job resume this exact W&B run.
        run_dir = Path(bc.out_dir) / bc.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "wandb_run_id.txt").write_text(run.id)

    rng = np.random.default_rng(cfg.seed)

    def _drain_buffer(buf: list[dict]) -> None:
        nonlocal state, global_step, n_samples
        # buf items are pool-format numpy dicts from workers; concatenate then assemble
        keys = buf[0].keys()
        data = {k: np.concatenate([s[k] for s in buf], axis=0) for k in keys}
        data = assemble_for_k(data, cfg.env.candidate_count)
        data = _drop_losers(data, bc)
        N    = int(data["target_idx"].shape[0])
        perm = jnp.array(rng.permutation(N))
        sf = jnp.asarray(data["self_feats"][perm])
        cf = jnp.asarray(data["cand_feats"][perm])
        gf = jnp.asarray(data["global_feats"][perm])
        cm = jnp.asarray(data["cand_mask"][perm])
        ti = jnp.asarray(data["target_idx"][perm])
        w  = jnp.asarray(_sample_weights(data, bc, skill)[perm])

        if bc.augment:
            sf, cf, gf, cm, ti = expand_with_augmentations(
                sf, cf, gf, cm, ti, aug_ids
            )
            w = jnp.concatenate([w] * len(aug_ids), axis=0)  # tile to match expansion
            N = int(sf.shape[0])
            perm2 = jnp.array(rng.permutation(N))
            sf, cf, gf = sf[perm2], cf[perm2], gf[perm2]
            cm, ti, w = cm[perm2], ti[perm2], w[perm2]

        buf_losses: list[jax.Array] = []
        buf_tgt_a: list[jax.Array] = []
        for start in range(0, N - bc.batch_size + 1, bc.batch_size):
            end = start + bc.batch_size
            state, loss, tgt_a = _bc_step(
                state,
                sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                ti[start:end], w[start:end],
            )
            buf_losses.append(loss)
            buf_tgt_a.append(tgt_a)
            global_step += 1

        losses.append(float(jnp.mean(jnp.stack(buf_losses))))
        tgt_accs.append(float(jnp.mean(jnp.stack(buf_tgt_a))))
        n_samples += N

    print(f"Starting BC training: {bc.epochs} epochs, batch={bc.batch_size}, "
          f"encode_workers={num_workers}", flush=True)
    print(f"Checkpoints → {Path(bc.out_dir) / bc.run_name}", flush=True)

    # Workers must not try to initialize the GPU backend — they do CPU encoding.
    # Set JAX_PLATFORMS=cpu in the environment *before* spawning so the workers
    # inherit it at spawn time (the `spawn` start method imports bc_train.py which
    # triggers `import jax` before the initializer callback runs).
    _saved_jax_platforms = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    mp_ctx = multiprocessing.get_context("spawn")
    try:
      with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(cfg.env,),
        mp_context=mp_ctx,
      ) as pool:

        def _iter_all():
            for d in episodes_dirs:
                avg_map = dir_avg_maps.get(d, {})
                for _name, ep in _iter_episode_jsons(d):
                    yield ep, _episode_avg_score(ep, avg_map)

        for epoch in range(1, bc.epochs + 1):
            losses, tgt_accs = [], []
            t0 = time.time()
            n_episodes = n_samples = global_step = 0
            buffer: list[dict] = []
            limit_msg = "" if bc.top_k_episodes is None else f" (top_k_episodes={bc.top_k_episodes})"
            print(
                f"\nepoch {epoch}/{bc.epochs}: reading from {[str(d) for d in episodes_dirs]}"
                f"{limit_msg} ...",
                flush=True,
            )

            pending: deque[Future] = deque()
            ep_iter = _iter_all()

            # Pre-fill sliding window
            for _ in range(_INFLIGHT):
                task = next(ep_iter, None)
                if task is None:
                    break
                pending.append(pool.submit(_encode_episode, task))

            while pending:
                f = pending.popleft()

                # Refill window before blocking on result (keeps workers busy)
                task = next(ep_iter, None)
                if task is not None and (bc.top_k_episodes is None or n_episodes < bc.top_k_episodes):
                    pending.append(pool.submit(_encode_episode, task))

                samples = f.result()
                if samples is None:
                    continue

                buffer.append(samples)
                n_episodes += 1

                if n_episodes % 20 == 0:
                    print(f"  loading ep={n_episodes}  buffered={len(buffer)} ...", flush=True)

                if len(buffer) >= bc.stream_buffer:
                    if global_step == 0:
                        print("  (first batch — JAX JIT compile, may take ~2 min) ...", flush=True)
                    _drain_buffer(buffer)
                    buffer = []
                    elapsed = time.time() - t0
                    steps_per_s = global_step / elapsed if elapsed > 0 else 0
                    print(
                        f"  ep={n_episodes}  samples={n_samples}  steps={global_step}"
                        f"  loss={float(np.mean(losses)):.4f}"
                        f"  tgt_acc={float(np.mean(tgt_accs)):.3f}"
                        f"  {elapsed:.0f}s ({steps_per_s:.1f} steps/s)",
                        flush=True,
                    )

                if bc.top_k_episodes is not None and n_episodes >= bc.top_k_episodes:
                    for pf in pending:
                        pf.cancel()
                    pending.clear()
                    break

            if buffer:
                _drain_buffer(buffer)

            if n_episodes == 0:
                dirs_str = ", ".join(d.name for d in episodes_dirs)
                raise RuntimeError(
                    f"Stream mode processed 0 episodes from: {dirs_str}\n"
                    "Check that the directories contain .zip or .json episode replay files.\n"
                    "Example: --episode_dirs episodes/2026-05-24"
                )

            mean_loss    = float(np.mean(losses)) if losses else float("nan")
            mean_tgt_acc = float(np.mean(tgt_accs)) if tgt_accs else float("nan")
            elapsed      = time.time() - t0
            steps_per_s  = global_step / elapsed if elapsed > 0 else 0
            print(
                f"epoch={epoch:3d}  loss={mean_loss:.4f}"
                f"  tgt_acc={mean_tgt_acc:.3f}"
                f"  ep={n_episodes}  samples={n_samples}"
                f"  steps={global_step}  {elapsed:.1f}s ({steps_per_s:.1f} steps/s)"
            )

            if use_wandb:
                import wandb
                wandb.log(
                    {"bc/loss": mean_loss, "bc/tgt_acc": mean_tgt_acc},
                    step=global_step,
                )

            save_bc_checkpoint(Path(bc.out_dir), bc.run_name, epoch, state, cfg)

    finally:
        if _saved_jax_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = _saved_jax_platforms

    if use_wandb:
        import wandb
        wandb.finish()

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode_dirs", "--episodes_dirs", "--episodes_dir",
                        dest="episodes_dirs", nargs="+", default=["episodes"],
                        help="Date directories (e.g. episodes/2026-05-24). "
                             "With --recursive, parent dirs whose episode subdirs are trained.")
    parser.add_argument("--recursive", action="store_true",
                        help="Treat --episode_dirs as parent dirs and train from every subdirectory "
                             "that contains episode files. In --use_shard mode, subdirectories with "
                             "pre-built shards also qualify.")
    parser.add_argument("--config", default=None,
                        help="TrainConfig YAML (formal mode). Omit to use dataclass defaults "
                             "with CLI overrides (smoke-test mode).")
    # Smoke-test overrides — applied to cfg.bc when --config is not given
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="AdamW decoupled weight decay; >0 switches Adam->AdamW (0=plain Adam)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Linear LR warmup 0->lr over N optimizer steps, then hold/decay (0=off)")
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default=None,
                        help="Post-warmup LR schedule: constant (hold) or cosine (decay to lr_final)")
    parser.add_argument("--lr_final", type=float, default=None,
                        help="Cosine end LR (absolute); only used with --lr_schedule cosine")
    parser.add_argument("--decay_steps", type=int, default=None,
                        help="Cosine step horizon; 0=auto-estimate from shard count (shard mode)")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Global-norm gradient clip before the optimizer (<=0 disables)")
    parser.add_argument("--loser_weight", type=float, default=None,
                        help="Loss weight for losing-player samples (1.0=equal, 0.0=winner-only, 0.5=default mix)")
    parser.add_argument("--skill_clip_ratio", type=float, default=None,
                        help="R: skill-weight clip band [1/R, R] from manifest avg_score; T auto-set so a "
                             "p95 game → weight R. 1.0=off. Requires shards re-encoded with avg_score.")
    parser.add_argument("--recency_half_life", type=float, default=None,
                        help="Recency decay half-life in days (shard mode): a shard's samples are "
                             "weighted by 0.5**(days_old/half_life). <=0 or omitted = off.")
    parser.add_argument("--recency_ref_date", type=str, default=None,
                        help="Reference date YYYY-MM-DD recency decays back from (default: latest shard date)")
    parser.add_argument("--checkpoint_every", type=int, default=None,
                        help="Save a checkpoint every N shards within an epoch (shard mode).")
    parser.add_argument("--stream_buffer", type=int, default=None)
    parser.add_argument("--top_k_episodes", type=int, default=None,
                        help="Process only the first N valid episodes (zip-streaming mode only)")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="Override model.hidden_size (e.g. for a capacity sweep)")
    parser.add_argument("--candidate_count", type=int, default=None,
                        help="Override env.candidate_count K (re-sliced from the POOL_K=24 shard pool; "
                             "must be <= POOL_K). For a candidate-count sweep.")
    parser.add_argument("--shard_tag", default=None,
                        help="Read shards from .../shards_<tag>/ instead of .../shards/ "
                             "(must match the tag used at encode time). Overrides cfg.bc.shard_tag.")
    parser.add_argument("--augment", default=None, action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    # Runtime flags — accepted in both modes
    parser.add_argument("--use_shard", default=None, action="store_true",
                        help="Use pre-built shards instead of streaming from zip")
    parser.add_argument("--no_use_shard", dest="use_shard", action="store_false",
                        help="Stream on the fly from zip (default)")
    parser.add_argument("--encode_workers", type=int, default=None,
                        help="Parallel CPU encoder processes (0 = auto = cpu_count-1)")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    from .accel import setup_accelerator
    setup_accelerator()  # CPU / GPU / TPU — pins matmul precision for backend parity

    args = parse_args()

    cfg = load_train_config(args.config) if args.config is not None else TrainConfig()

    # CLI flags always override config (last writer wins: defaults → YAML → CLI)
    bc = cfg.bc
    if args.out_dir        is not None: bc.out_dir        = args.out_dir
    if args.epochs         is not None: bc.epochs         = args.epochs
    if args.batch_size     is not None: bc.batch_size     = args.batch_size
    if args.lr             is not None: bc.lr             = args.lr
    if args.weight_decay   is not None: bc.weight_decay   = args.weight_decay
    if args.warmup_steps   is not None: bc.warmup_steps   = args.warmup_steps
    if args.lr_schedule    is not None: bc.lr_schedule    = args.lr_schedule
    if args.lr_final       is not None: bc.lr_final       = args.lr_final
    if args.decay_steps    is not None: bc.decay_steps    = args.decay_steps
    if args.grad_clip      is not None: bc.grad_clip      = args.grad_clip
    if args.loser_weight   is not None: bc.loser_weight   = args.loser_weight
    if args.skill_clip_ratio is not None: bc.skill_clip_ratio = args.skill_clip_ratio
    if args.recency_half_life is not None: bc.recency_half_life = args.recency_half_life
    if args.recency_ref_date is not None: bc.recency_ref_date = args.recency_ref_date
    if args.checkpoint_every is not None: bc.checkpoint_every = args.checkpoint_every
    if args.stream_buffer  is not None: bc.stream_buffer  = args.stream_buffer
    if args.top_k_episodes is not None: bc.top_k_episodes = args.top_k_episodes
    if args.run_name       is not None: bc.run_name       = args.run_name
    if args.hidden_size    is not None: cfg.model.hidden_size = args.hidden_size
    if args.candidate_count is not None: cfg.env.candidate_count = args.candidate_count
    if args.shard_tag      is not None: bc.shard_tag      = args.shard_tag
    if args.augment        is not None: bc.augment        = args.augment
    if args.use_shard      is not None: bc.use_shard      = args.use_shard
    if args.encode_workers is not None: bc.encode_workers = args.encode_workers
    use_wandb = not args.no_wandb

    if cfg.bc.top_k_episodes is not None and cfg.bc.top_k_episodes <= 0:
        raise ValueError("--top_k_episodes must be a positive integer when provided")

    episodes_dirs = [Path(d) for d in args.episodes_dirs]

    if cfg.bc.use_shard:
        train(episodes_dirs, cfg, use_wandb=use_wandb, recursive=args.recursive)
    else:
        train_stream(episodes_dirs, cfg, use_wandb=use_wandb, recursive=args.recursive)


if __name__ == "__main__":
    main()
