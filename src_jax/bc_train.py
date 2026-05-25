"""
Behavior-cloning pretraining: imitate top players by minimizing cross-entropy.

Usage:
    python -m src_jax.bc_train [options]

    --episodes_dir DIR   Directory with episode JSON files (default: episodes/)
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

from .bc_data import concat_samples, iter_episodes
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
    ff: jax.Array,   # [N, MAX_FLEETS, fleet_dim]
    fm: jax.Array,   # [N, MAX_FLEETS]
    ti: jax.Array,   # [N]  int32  target index
    sb: jax.Array,   # [N]  int32  ship bucket
) -> tuple[TrainState, jax.Array, jax.Array, jax.Array]:
    def loss_fn(params):
        out = state.apply_fn(params, sf, cf, gf, cm, ff, fm)
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
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    episodes_dir: Path,
    encode_fn,
    cfg: TrainConfig,
    max_samples: int,
    winner_only: bool,
) -> dict[str, np.ndarray]:
    """Load all episodes and concatenate into a single flat dataset."""
    sample_list = []
    total = 0
    n_episodes = 0
    t0 = time.time()

    for samples in iter_episodes(
        episodes_dir,
        encode_fn,
        cfg.env.ship_bucket_count,
        winner_only=winner_only,
    ):
        sample_list.append(samples)
        total += len(samples["target_idx"])
        n_episodes += 1
        if n_episodes % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Loaded {n_episodes} episodes, {total} samples "
                f"({elapsed:.1f}s)"
            )
        if max_samples > 0 and total >= max_samples:
            break

    if not sample_list:
        raise RuntimeError(
            f"No valid episode samples found in {episodes_dir}. "
            "Run download_episodes.py first."
        )

    dataset = concat_samples(sample_list)
    print(
        f"Dataset: {n_episodes} episodes, {total} samples "
        f"({time.time() - t0:.1f}s total)"
    )
    return dataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_samples: int,
    winner_only: bool,
    run_name: str,
    use_wandb: bool,
) -> TrainState:
    K = cfg.env.candidate_count

    # Build encoder (JIT+vmapped)
    encode_fn = _make_env_encoder(K, cfg.env)

    # Build policy + train state with BC-specific optimiser
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
                "episodes_dir": str(episodes_dir),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_samples": max_samples,
                "winner_only": winner_only,
                "candidate_count": K,
                "ship_bucket_count": cfg.env.ship_bucket_count,
                "hidden_size": cfg.model.hidden_size,
            },
        )

    print("Loading episodes …")
    dataset = load_dataset(episodes_dir, encode_fn, cfg, max_samples, winner_only)

    sf  = jnp.asarray(dataset["self_feats"])
    cf  = jnp.asarray(dataset["cand_feats"])
    gf  = jnp.asarray(dataset["global_feats"])
    cm  = jnp.asarray(dataset["cand_mask"])
    ff  = jnp.asarray(dataset["fleet_feats"])
    fm  = jnp.asarray(dataset["fleet_mask"])
    ti  = jnp.asarray(dataset["target_idx"])
    sb  = jnp.asarray(dataset["ships_bucket"])
    N   = sf.shape[0]

    key = jax.random.PRNGKey(cfg.seed)

    print(f"\nStarting BC training: {epochs} epochs × {N} samples, batch={batch_size}")
    print(f"Checkpoints → {out_dir / run_name}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        order  = jax.random.permutation(subkey, N)
        losses, tgt_losses, shp_losses = [], [], []
        t0 = time.time()

        for start in range(0, N, batch_size):
            idx = order[start : start + batch_size]
            state, loss, tgt_l, shp_l = _bc_step(
                state,
                sf[idx], cf[idx], gf[idx], cm[idx],
                ff[idx], fm[idx],
                ti[idx], sb[idx],
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
                {
                    "bc/loss":       mean_loss,
                    "bc/tgt_loss":   mean_tgt,
                    "bc/ships_loss": mean_shp,
                },
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
    parser.add_argument("--episodes_dir", default="episodes",
                        help="Directory containing episode JSON files")
    parser.add_argument("--out_dir", default="artifacts/bc",
                        help="Checkpoint output directory")
    parser.add_argument("--config", default=str(default_train_config_path()),
                        help="TrainConfig YAML path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap dataset size (0 = unlimited)")
    parser.add_argument("--winner_only", action="store_true",
                        help="Learn only from the winning agent")
    parser.add_argument("--run_name", default="bc_pretrain")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_train_config(args.config)

    train(
        episodes_dir = Path(args.episodes_dir),
        out_dir      = Path(args.out_dir),
        cfg          = cfg,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        max_samples  = args.max_samples,
        winner_only  = args.winner_only,
        run_name     = args.run_name,
        use_wandb    = not args.no_wandb,
    )


if __name__ == "__main__":
    main()
