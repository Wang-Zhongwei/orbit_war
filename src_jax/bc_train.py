"""
Behavior-cloning pretraining: imitate top players by minimizing cross-entropy.

Usage:
    python -m src_jax.bc_train [options]

    --episode_dirs DIR [DIR ...]
                         Date directories containing shards/ (default: episodes/)
    --out_dir DIR        Where to save checkpoints (default: artifacts/bc)
    --config PATH        TrainConfig YAML (default: src_jax/configs/default_cfg.yaml)
    --epochs N           Passes over the dataset (default: 10)
    --batch_size N       Samples per gradient step (default: 2048)
    --lr LR              Learning rate (default: 3e-4)
    --winner_only        Learn only from the winning agent (default: False)
    --stream             Encode episodes on the fly from zip; skips shards entirely.
                         Best for 1-epoch runs over large datasets (no disk space wasted,
                         no pre-encode step). Re-encoding cost per epoch is the trade-off.
    --stream_buffer N    Episodes to buffer before training in stream mode (default: 200)
    --run_name NAME      W&B / checkpoint run name (default: bc_pretrain)
    --no_wandb           Disable W&B logging

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

from .bc_data import (
    _filter_winner,
    _iter_episode_jsons,
    concat_samples,
    episode_to_samples,
    episode_to_samples_jax,
    find_shards,
)
from .config import TrainConfig, default_train_config_path, load_train_config
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
    return state, loss, aux[0], aux[1]   # state, total, tgt_loss, shp_loss


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
    out_dir: Path,
    cfg: TrainConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    run_name: str,
    use_wandb: bool,
) -> TrainState:
    shard_paths = find_shards(episodes_dirs)
    print(f"Found {len(shard_paths)} shards across {len(episodes_dirs)} date dir(s)")

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr),
    )
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    if use_wandb:
        import wandb
        wandb.init(
            project="orbit-war",
            name=run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "n_shards": len(shard_paths),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "candidate_count": cfg.env.candidate_count,
                "ship_bucket_count": cfg.env.ship_bucket_count,
                "hidden_size": cfg.model.hidden_size,
            },
        )

    print(f"Starting BC training: {epochs} epochs, batch={batch_size}")
    print(f"Checkpoints → {out_dir / run_name}")

    rng = np.random.default_rng(cfg.seed)
    global_step = 0

    for epoch in range(1, epochs + 1):
        shard_order = rng.permutation(len(shard_paths)).tolist()
        losses, tgt_losses, shp_losses = [], [], []
        t0 = time.time()

        for shard_i in shard_order:
            data = np.load(shard_paths[shard_i])
            N    = data["target_idx"].shape[0]
            perm = rng.permutation(N)

            sf = jnp.asarray(data["self_feats"][perm])
            cf = jnp.asarray(data["cand_feats"][perm])
            gf = jnp.asarray(data["global_feats"][perm])
            cm = jnp.asarray(data["cand_mask"][perm])
            ti = jnp.asarray(data["target_idx"][perm])
            sb = jnp.asarray(data["ships_bucket"][perm])

            for start in range(0, N - batch_size + 1, batch_size):
                end = start + batch_size
                state, loss, tgt_l, shp_l = _bc_step(
                    state,
                    sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                    ti[start:end], sb[start:end],
                )
                losses.append(float(loss))
                tgt_losses.append(float(tgt_l))
                shp_losses.append(float(shp_l))
                global_step += 1

        mean_loss = float(np.mean(losses))
        mean_tgt  = float(np.mean(tgt_losses))
        mean_shp  = float(np.mean(shp_losses))
        elapsed   = time.time() - t0
        print(
            f"epoch={epoch:3d}  loss={mean_loss:.4f}"
            f"  tgt={mean_tgt:.4f}  shp={mean_shp:.4f}"
            f"  {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {"bc/loss": mean_loss, "bc/tgt_loss": mean_tgt, "bc/ships_loss": mean_shp},
                step=global_step,
            )

        save_bc_checkpoint(out_dir, run_name, epoch, state, cfg)

    if use_wandb:
        import wandb
        wandb.finish()

    return state


# ---------------------------------------------------------------------------
# Streaming training loop (on-the-fly encode from zip, no shards)
# ---------------------------------------------------------------------------

def train_stream(
    episodes_dirs: list[Path],
    out_dir: Path,
    cfg: TrainConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    run_name: str,
    use_wandb: bool,
    winner_only: bool,
    stream_buffer: int,
) -> TrainState:
    encode_fn = _make_env_encoder(cfg.env.candidate_count, cfg.env)

    policy = build_policy(cfg)
    params = init_params(policy, cfg.seed)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr),
    )
    state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    if use_wandb:
        import wandb
        wandb.init(
            project="orbit-war",
            name=run_name,
            config={
                "episodes_dirs": [str(d) for d in episodes_dirs],
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "stream": True,
                "stream_buffer": stream_buffer,
                "candidate_count": cfg.env.candidate_count,
                "ship_bucket_count": cfg.env.ship_bucket_count,
                "hidden_size": cfg.model.hidden_size,
            },
        )

    rng = np.random.default_rng(cfg.seed)

    def _drain_buffer(buf: list[dict]) -> None:
        nonlocal state, global_step, n_samples
        # Concatenate on-device — no host transfer of feature tensors
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
        for start in range(0, N - batch_size + 1, batch_size):
            end = start + batch_size
            state, loss, tgt_l, shp_l = _bc_step(
                state,
                sf[start:end], cf[start:end], gf[start:end], cm[start:end],
                ti[start:end], sb[start:end],
            )
            losses.append(float(loss))
            tgt_losses.append(float(tgt_l))
            shp_losses.append(float(shp_l))
            global_step += 1
        n_samples += N

    for epoch in range(1, epochs + 1):
        losses, tgt_losses, shp_losses = [], [], []
        t0 = time.time()
        n_episodes = n_samples = global_step = 0
        buffer: list[dict] = []

        for episodes_dir in episodes_dirs:
            for _name, episode in _iter_episode_jsons(episodes_dir):
                if not isinstance(episode, dict) or "steps" not in episode:
                    continue
                if winner_only:
                    rewards  = episode.get("rewards", [])
                    statuses = episode.get("statuses", [])
                    winning_player = next(
                        (i for i, (r, s) in enumerate(zip(rewards, statuses))
                         if r == max(rewards) and s == "DONE"),
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

                if len(buffer) >= stream_buffer:
                    _drain_buffer(buffer)
                    buffer = []
                    if n_episodes % (stream_buffer * 5) == 0:
                        print(f"  ep={n_episodes}  samples={n_samples}  steps={global_step}"
                              f"  ({time.time() - t0:.0f}s)")

        if buffer:
            _drain_buffer(buffer)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_tgt  = float(np.mean(tgt_losses)) if tgt_losses else float("nan")
        mean_shp  = float(np.mean(shp_losses)) if shp_losses else float("nan")
        print(
            f"epoch={epoch:3d}  loss={mean_loss:.4f}"
            f"  tgt={mean_tgt:.4f}  shp={mean_shp:.4f}"
            f"  ep={n_episodes}  samples={n_samples}"
            f"  {time.time() - t0:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {"bc/loss": mean_loss, "bc/tgt_loss": mean_tgt, "bc/ships_loss": mean_shp},
                step=global_step,
            )

        save_bc_checkpoint(out_dir, run_name, epoch, state, cfg)

    if use_wandb:
        import wandb
        wandb.finish()

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_bc_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "bc_cfg.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dirs", "--episodes_dirs", "--episodes_dir",
                        dest="episodes_dirs", nargs="+", default=["episodes"],
                        help="Date directories (e.g. episodes/2026-05-24 episodes/2026-05-23)")
    parser.add_argument("--config", default=str(_default_bc_config_path()),
                        help="TrainConfig YAML (default: src_jax/configs/bc_cfg.yaml)")
    # BC fields — all optional; config file values are used when not provided
    parser.add_argument("--out_dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--winner_only", default=None, action="store_true")
    parser.add_argument("--stream", default=None, action="store_true",
                        help="Encode on the fly from zip; no shards needed")
    parser.add_argument("--no_stream", dest="stream", action="store_false",
                        help="Use pre-built shards (overrides config stream=true)")
    parser.add_argument("--stream_buffer", type=int, default=None,
                        help="Episodes to buffer before a training pass in stream mode")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_train_config(args.config)
    bc   = cfg.bc  # defaults from config file

    # CLI args override config; None means "not provided on command line"
    out_dir       = Path(args.out_dir   if args.out_dir       is not None else bc.out_dir)
    epochs        = args.epochs         if args.epochs         is not None else bc.epochs
    batch_size    = args.batch_size     if args.batch_size     is not None else bc.batch_size
    lr            = args.lr             if args.lr             is not None else bc.lr
    winner_only   = args.winner_only    if args.winner_only    is not None else bc.winner_only
    use_stream    = args.stream         if args.stream         is not None else bc.stream
    stream_buffer = args.stream_buffer  if args.stream_buffer  is not None else bc.stream_buffer
    run_name      = args.run_name       if args.run_name       is not None else bc.run_name

    episodes_dirs = [Path(d) for d in args.episodes_dirs]
    common = dict(
        out_dir    = out_dir,
        cfg        = cfg,
        epochs     = epochs,
        batch_size = batch_size,
        lr         = lr,
        run_name   = run_name,
        use_wandb  = not args.no_wandb,
    )

    if use_stream:
        train_stream(
            episodes_dirs,
            **common,
            winner_only   = winner_only,
            stream_buffer = stream_buffer,
        )
    else:
        train(episodes_dirs, **common)


if __name__ == "__main__":
    main()
