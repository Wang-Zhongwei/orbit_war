"""
Behavior-cloning pretraining: imitate top players by minimizing cross-entropy.

Usage:
    # Formal training — config file + optional CLI overrides (CLI wins):
    python -m src_jax.bc_train --config src_jax/configs/bc_cfg.yaml \\
                                --episode_dirs episodes/2026-05-24 \\
                                --top_k_episodes 200 --out_dir ./artifacts/bc/run1

    # Smoke test — dataclass defaults + CLI overrides, no config file needed:
    python -m src_jax.bc_train --episode_dirs episodes/2026-05-24 \\
                                --top_k_episodes 5 --epochs 1

Runtime flags (accepted in both modes):
    --use_shard / --no_use_shard   Use pre-built shards (default: stream from zip)
    --no_wandb               Disable W&B logging

Loss:
    CE(target_logits, expert_target) + CE(ships_logits[expert_target], expert_bucket)
    No value loss, no entropy bonus — pure supervised imitation.

The saved checkpoint is compatible with train.py --resume so you can warm-start PPO.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .augment import DEFAULT_AUG_IDS, expand_with_augmentations
from .bc_data import (
    _filter_winner,
    _iter_episode_jsons,
    episode_to_samples_jax,
    find_shards,
    shard_episode_counts,
)
from .config import TrainConfig, load_train_config
from .features import _make_env_encoder
from .policy import build_policy, init_params
from .ppo import TrainState


# ---------------------------------------------------------------------------
# BC loss — single JIT-compiled minibatch update
# ---------------------------------------------------------------------------

@jax.jit
def _bc_step(
    state: TrainState,
    sf: jax.Array,   # [N, self_dim]
    cf: jax.Array,   # [N, K, cand_dim]
    gf: jax.Array,   # [N, global_dim]
    cm: jax.Array,   # [N, K]
    ti: jax.Array,   # [N]  int32  target index
    sb: jax.Array,   # [N]  int32  ship bucket
) -> tuple[TrainState, jax.Array, jax.Array, jax.Array]:
    def loss_fn(params):
        out = state.apply_fn(params, sf, cf, gf, cm)
        N   = ti.shape[0]

        # Target cross-entropy
        tgt_lp   = jax.nn.log_softmax(out.target_logits, axis=-1)
        tgt_loss = -tgt_lp[jnp.arange(N), ti].mean()

        # Ships cross-entropy — only over the chosen target's ship logits
        shp_logits = out.ships_logits[jnp.arange(N), ti]   # [N, B]
        shp_lp     = jax.nn.log_softmax(shp_logits, axis=-1)
        shp_loss   = -shp_lp[jnp.arange(N), sb].mean()

        # Top-1 accuracy helpers (returned as aux, not differentiated)
        tgt_acc = (jnp.argmax(out.target_logits, axis=-1) == ti).astype(jnp.float32).mean()
        shp_acc = (jnp.argmax(shp_logits,        axis=-1) == sb).astype(jnp.float32).mean()

        total = tgt_loss + shp_loss
        return total, (tgt_loss, shp_loss, tgt_acc, shp_acc)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux[0], aux[1], aux[2], aux[3]  # state, total, tgt_loss, shp_loss, tgt_acc, shp_acc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_bc_checkpoint(
    save_dir: Path,
    run_name: str,
    epoch: int,
    state: TrainState,
    cfg: TrainConfig,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "update":    epoch,
        "params":    state.params,
        "opt_state": state.opt_state,
        "config":    cfg,
    }
    # Save in the same format as train.py so PPO --resume works directly.
    for path in [run_dir / "ckpt_last.pkl", run_dir / f"ckpt_{epoch:06d}.pkl"]:
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
    print(f"  Checkpoint saved → {run_dir}/ckpt_{epoch:06d}.pkl")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes_dirs: list[Path],
    cfg: TrainConfig,
    *,
    use_wandb: bool = True,
) -> TrainState:
    """Shard-based BC training loop. All hyperparameters are read from cfg.bc."""
    bc = cfg.bc
    aug_ids = DEFAULT_AUG_IDS
    shard_paths = find_shards(episodes_dirs)
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

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(bc.lr),
    )
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    if use_wandb:
        import wandb
        wandb.init(
            project="orbit-war",
            name=bc.run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "n_shards": len(shard_paths),
                "epochs": bc.epochs,
                "batch_size": bc.batch_size,
                "lr": bc.lr,
                "augment": bc.augment,
                "aug_ids": aug_ids if bc.augment else [],
                "candidate_count": cfg.env.candidate_count,
                "ship_bucket_count": cfg.env.ship_bucket_count,
                "top_k_episodes": bc.top_k_episodes,
                "hidden_size": cfg.model.hidden_size,
            },
        )

    print(f"Starting BC training: {bc.epochs} epochs, batch={bc.batch_size}", flush=True)
    print(f"Checkpoints → {Path(bc.out_dir) / bc.run_name}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    global_step = 0

    for epoch in range(1, bc.epochs + 1):
        shard_order = rng.permutation(len(shard_paths)).tolist()
        losses, tgt_losses, shp_losses = [], [], []
        tgt_accs, shp_accs = [], []
        t0 = time.time()
        n_steps_epoch = 0

        print(f"\nepoch {epoch}/{bc.epochs}  ({len(shard_paths)} shards)", flush=True)

        for si, shard_i in enumerate(shard_order):
            data = np.load(shard_paths[shard_i])
            N    = data["target_idx"].shape[0]
            perm = rng.permutation(N)

            sf = jnp.asarray(data["self_feats"][perm])
            cf = jnp.asarray(data["cand_feats"][perm])
            gf = jnp.asarray(data["global_feats"][perm])
            cm = jnp.asarray(data["cand_mask"][perm])
            ti = jnp.asarray(data["target_idx"][perm])
            sb = jnp.asarray(data["ships_bucket"][perm])

            # D4 augmentation: expand shard N→N*len(aug_ids) then re-shuffle.
            if bc.augment:
                sf, cf, gf, cm, ti, sb = expand_with_augmentations(
                    sf, cf, gf, cm, ti, sb, aug_ids
                )
                N = sf.shape[0]
                perm2 = jnp.asarray(rng.permutation(N))
                sf, cf, gf = sf[perm2], cf[perm2], gf[perm2]
                cm, ti, sb = cm[perm2], ti[perm2], sb[perm2]

            shard_losses: list[jax.Array] = []
            shard_tgt_l: list[jax.Array] = []
            shard_shp_l: list[jax.Array] = []
            shard_tgt_a: list[jax.Array] = []
            shard_shp_a: list[jax.Array] = []
            for start in range(0, N - bc.batch_size + 1, bc.batch_size):
                end = start + bc.batch_size
                state, loss, tgt_l, shp_l, tgt_a, shp_a = _bc_step(
                    state,
                    sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                    ti[start:end], sb[start:end],
                )
                shard_losses.append(loss)
                shard_tgt_l.append(tgt_l)
                shard_shp_l.append(shp_l)
                shard_tgt_a.append(tgt_a)
                shard_shp_a.append(shp_a)
                global_step += 1
                n_steps_epoch += 1

            # Single GPU→CPU sync per shard instead of per step
            losses.append(float(jnp.mean(jnp.stack(shard_losses))))
            tgt_losses.append(float(jnp.mean(jnp.stack(shard_tgt_l))))
            shp_losses.append(float(jnp.mean(jnp.stack(shard_shp_l))))
            tgt_accs.append(float(jnp.mean(jnp.stack(shard_tgt_a))))
            shp_accs.append(float(jnp.mean(jnp.stack(shard_shp_a))))

            elapsed = time.time() - t0
            steps_per_s = n_steps_epoch / elapsed if elapsed > 0 else 0
            print(
                f"  shard {si+1:3d}/{len(shard_paths)}"
                f"  loss={losses[-1]:.4f}"
                f"  tgt_acc={tgt_accs[-1]:.3f}"
                f"  shp_acc={shp_accs[-1]:.3f}"
                f"  steps={n_steps_epoch}"
                f"  {elapsed:.0f}s ({steps_per_s:.1f} steps/s)",
                flush=True,
            )

        mean_loss = float(np.mean(losses))
        mean_tgt  = float(np.mean(tgt_losses))
        mean_shp  = float(np.mean(shp_losses))
        mean_tgt_acc = float(np.mean(tgt_accs))
        mean_shp_acc = float(np.mean(shp_accs))
        elapsed   = time.time() - t0
        steps_per_s = n_steps_epoch / elapsed if elapsed > 0 else 0
        print(
            f"epoch={epoch:3d}  loss={mean_loss:.4f}"
            f"  tgt={mean_tgt:.4f}  shp={mean_shp:.4f}"
            f"  tgt_acc={mean_tgt_acc:.3f}  shp_acc={mean_shp_acc:.3f}"
            f"  steps={n_steps_epoch}  {elapsed:.1f}s ({steps_per_s:.1f} steps/s)"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "bc/loss": mean_loss, "bc/tgt_loss": mean_tgt, "bc/ships_loss": mean_shp,
                    "bc/tgt_acc": mean_tgt_acc, "bc/shp_acc": mean_shp_acc,
                },
                step=global_step,
            )

        save_bc_checkpoint(Path(bc.out_dir), bc.run_name, epoch, state, cfg)

    if use_wandb:
        import wandb
        wandb.finish()

    return state


def _resolve_episode_dirs(episodes_dirs: list[Path]) -> list[Path]:
    """Auto-expand dirs that contain no episode files directly but have dated subdirs."""
    resolved: list[Path] = []
    for d in episodes_dirs:
        d = Path(d)
        if not d.is_dir():
            resolved.append(d)
            continue
        has_direct = any(d.glob("*.zip")) or any(d.glob("*.json"))
        if has_direct:
            resolved.append(d)
        else:
            subdirs = sorted(
                child for child in d.iterdir()
                if child.is_dir()
                and (any(child.glob("*.zip")) or any(child.glob("*.json")))
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
) -> TrainState:
    """Streaming BC training loop (on-the-fly encoding from zip). All hyperparameters are read from cfg.bc."""
    bc = cfg.bc
    aug_ids = DEFAULT_AUG_IDS

    episodes_dirs = _resolve_episode_dirs(episodes_dirs)
    encode_fn = _make_env_encoder(cfg.env.candidate_count, cfg.env)

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(bc.lr),
    )
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    if use_wandb:
        import wandb
        wandb.init(
            project="orbit-war",
            name=bc.run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "epochs": bc.epochs,
                "batch_size": bc.batch_size,
                "lr": bc.lr,
                "use_shard": False,
                "stream_buffer": bc.stream_buffer,
                "top_k_episodes": bc.top_k_episodes,
                "augment": bc.augment,
                "aug_ids": aug_ids if bc.augment else [],
                "candidate_count": cfg.env.candidate_count,
                "ship_bucket_count": cfg.env.ship_bucket_count,
                "hidden_size": cfg.model.hidden_size,
            },
        )

    rng = np.random.default_rng(cfg.seed)

    def _drain_buffer(buf: list[dict]) -> None:
        nonlocal state, global_step, n_samples
        keys = buf[0].keys()
        data = {k: jnp.concatenate([s[k] for s in buf], axis=0) for k in keys}
        N    = int(data["target_idx"].shape[0])
        perm = jnp.array(rng.permutation(N))
        sf = data["self_feats"][perm]
        cf = data["cand_feats"][perm]
        gf = data["global_feats"][perm]
        cm = data["cand_mask"][perm]
        ti = data["target_idx"][perm]
        sb = data["ships_bucket"][perm]

        # D4 augmentation: expand buffer N→N*len(aug_ids) then re-shuffle.
        if bc.augment:
            sf, cf, gf, cm, ti, sb = expand_with_augmentations(
                sf, cf, gf, cm, ti, sb, aug_ids
            )
            N = int(sf.shape[0])
            perm2 = jnp.array(rng.permutation(N))
            sf, cf, gf = sf[perm2], cf[perm2], gf[perm2]
            cm, ti, sb = cm[perm2], ti[perm2], sb[perm2]

        buf_losses: list[jax.Array] = []
        buf_tgt_l: list[jax.Array] = []
        buf_shp_l: list[jax.Array] = []
        buf_tgt_a: list[jax.Array] = []
        buf_shp_a: list[jax.Array] = []
        for start in range(0, N - bc.batch_size + 1, bc.batch_size):
            end = start + bc.batch_size
            state, loss, tgt_l, shp_l, tgt_a, shp_a = _bc_step(
                state,
                sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                ti[start:end], sb[start:end],
            )
            buf_losses.append(loss)
            buf_tgt_l.append(tgt_l)
            buf_shp_l.append(shp_l)
            buf_tgt_a.append(tgt_a)
            buf_shp_a.append(shp_a)
            global_step += 1

        # Single GPU→CPU sync per buffer drain instead of per step
        losses.append(float(jnp.mean(jnp.stack(buf_losses))))
        tgt_losses.append(float(jnp.mean(jnp.stack(buf_tgt_l))))
        shp_losses.append(float(jnp.mean(jnp.stack(buf_shp_l))))
        tgt_accs.append(float(jnp.mean(jnp.stack(buf_tgt_a))))
        shp_accs.append(float(jnp.mean(jnp.stack(buf_shp_a))))
        n_samples += N

    for epoch in range(1, bc.epochs + 1):
        losses, tgt_losses, shp_losses = [], [], []
        tgt_accs, shp_accs = [], []
        t0 = time.time()
        n_episodes = n_samples = global_step = 0
        buffer: list[dict] = []
        limit_msg = "" if bc.top_k_episodes is None else f" (top_k_episodes={bc.top_k_episodes})"
        print(
            f"\nepoch {epoch}/{bc.epochs}: reading from {[str(d) for d in episodes_dirs]}"
            f"{limit_msg} ...",
            flush=True,
        )

        for episodes_dir in episodes_dirs:
            reached_episode_limit = False
            for _name, episode in _iter_episode_jsons(episodes_dir):
                if not isinstance(episode, dict) or "steps" not in episode:
                    continue
                if bc.winner_only:
                    rewards  = episode.get("rewards", [])
                    statuses = episode.get("statuses", [])
                    valid_rewards = [r for r in rewards if r is not None]
                    if not valid_rewards:
                        continue
                    best = max(valid_rewards)
                    winning_player = next(
                        (i for i, (r, s) in enumerate(zip(rewards, statuses))
                         if r == best and s == "DONE"),
                        None,
                    )
                    if winning_player is None:
                        continue
                    episode = _filter_winner(episode, winning_player)

                samples = episode_to_samples_jax(episode, encode_fn, cfg.env.ship_bucket_count)
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
                        f"  shp_acc={float(np.mean(shp_accs)):.3f}"
                        f"  {elapsed:.0f}s ({steps_per_s:.1f} steps/s)",
                        flush=True,
                    )

                if bc.top_k_episodes is not None and n_episodes >= bc.top_k_episodes:
                    reached_episode_limit = True
                    break

            if reached_episode_limit:
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

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_tgt  = float(np.mean(tgt_losses)) if tgt_losses else float("nan")
        mean_shp  = float(np.mean(shp_losses)) if shp_losses else float("nan")
        mean_tgt_acc = float(np.mean(tgt_accs)) if tgt_accs else float("nan")
        mean_shp_acc = float(np.mean(shp_accs)) if shp_accs else float("nan")
        elapsed = time.time() - t0
        steps_per_s = global_step / elapsed if elapsed > 0 else 0
        print(
            f"epoch={epoch:3d}  loss={mean_loss:.4f}"
            f"  tgt={mean_tgt:.4f}  shp={mean_shp:.4f}"
            f"  tgt_acc={mean_tgt_acc:.3f}  shp_acc={mean_shp_acc:.3f}"
            f"  ep={n_episodes}  samples={n_samples}"
            f"  steps={global_step}  {elapsed:.1f}s ({steps_per_s:.1f} steps/s)"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "bc/loss": mean_loss, "bc/tgt_loss": mean_tgt, "bc/ships_loss": mean_shp,
                    "bc/tgt_acc": mean_tgt_acc, "bc/shp_acc": mean_shp_acc,
                },
                step=global_step,
            )

        save_bc_checkpoint(Path(bc.out_dir), bc.run_name, epoch, state, cfg)

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
                        help="Date directories (e.g. episodes/2026-05-24)")
    parser.add_argument("--config", default=None,
                        help="TrainConfig YAML (formal mode). Omit to use dataclass defaults "
                             "with CLI overrides (smoke-test mode).")
    # Smoke-test overrides — applied to cfg.bc when --config is not given
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--winner_only", default=None, action="store_true")
    parser.add_argument("--stream_buffer", type=int, default=None)
    parser.add_argument("--top_k_episodes", type=int, default=None,
                        help="Process only the first N valid episodes (zip-streaming mode only)")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--augment", default=None, action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    # Runtime flags — accepted in both modes
    parser.add_argument("--use_shard", default=None, action="store_true",
                        help="Use pre-built shards instead of streaming from zip")
    parser.add_argument("--no_use_shard", dest="use_shard", action="store_false",
                        help="Stream on the fly from zip (default)")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_train_config(args.config) if args.config is not None else TrainConfig()

    # CLI flags always override config (last writer wins: defaults → YAML → CLI)
    bc = cfg.bc
    if args.out_dir        is not None: bc.out_dir        = args.out_dir
    if args.epochs         is not None: bc.epochs         = args.epochs
    if args.batch_size     is not None: bc.batch_size     = args.batch_size
    if args.lr             is not None: bc.lr             = args.lr
    if args.winner_only    is not None: bc.winner_only    = args.winner_only
    if args.stream_buffer  is not None: bc.stream_buffer  = args.stream_buffer
    if args.top_k_episodes is not None: bc.top_k_episodes = args.top_k_episodes
    if args.run_name       is not None: bc.run_name       = args.run_name
    if args.augment        is not None: bc.augment        = args.augment
    if args.use_shard      is not None: bc.use_shard      = args.use_shard
    use_wandb = not args.no_wandb

    if cfg.bc.top_k_episodes is not None and cfg.bc.top_k_episodes <= 0:
        raise ValueError("--top_k_episodes must be a positive integer when provided")

    episodes_dirs = [Path(d) for d in args.episodes_dirs]

    if cfg.bc.use_shard:
        train(episodes_dirs, cfg, use_wandb=use_wandb)
    else:
        train_stream(episodes_dirs, cfg, use_wandb=use_wandb)


if __name__ == "__main__":
    main()
