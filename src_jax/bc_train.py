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
    --max_samples N      Cap total samples loaded (0 = unlimited, default: 0)
    --winner_only        Learn only from the winning agent (default: False)
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

from .bc_data import find_shards
from .config import TrainConfig, default_train_config_path, load_train_config
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dirs", "--episodes_dirs", "--episodes_dir",
                        dest="episodes_dirs", nargs="+", default=["episodes"],
                        help="Date directories containing shards/ (e.g. episodes/2026-05-24 episodes/2026-05-23)")
    parser.add_argument("--out_dir", default="artifacts/bc",
                        help="Checkpoint output directory")
    parser.add_argument("--config", default=str(default_train_config_path()),
                        help="TrainConfig YAML path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--run_name", default="bc_pretrain")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_train_config(args.config)

    train(
        episodes_dirs = [Path(d) for d in args.episodes_dirs],
        out_dir       = Path(args.out_dir),
        cfg           = cfg,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        run_name      = args.run_name,
        use_wandb     = not args.no_wandb,
    )


if __name__ == "__main__":
    main()
