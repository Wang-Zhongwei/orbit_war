---
name: orbit-wars-training-pipeline
description: Run the Orbit Wars JAX training workflow: download Kaggle episode replays with download_episodes.py, pre-encode into shards with src_jax.bc_encode, behavior-clone pretrain with src_jax.bc_train, then start PPO self-play from the BC checkpoint. Use when the user asks to run, resume, monitor, or document the Orbit Wars data/pretraining/self-play pipeline.
---

# Orbit Wars Training Pipeline

## Preconditions

Work from the repository root in the `torch-gpu` conda environment.

Before launching anything long-running:
1. Check existing IDE terminal files for active download, encode, BC, or PPO jobs.
2. Verify the target dataset slug and output directories.
3. Use a new `--run_name` for each full-data BC run to avoid overwriting checkpoints.

```bash
conda run -n torch-gpu kaggle --version
conda run -n torch-gpu python -c "import jax, wandb; print('jax', jax.__version__); print('wandb', wandb.__version__)"
```

## Step 1 — Download

Downloads the full dataset as a single zip into `episodes/<date>/`.

```bash
conda run --no-capture-output -n torch-gpu python -u download_episodes.py \
  --dataset kaggle/orbit-wars-episodes-2026-05-24
# → episodes/2026-05-24/orbit-wars-episodes-2026-05-24.zip
```

The date subdirectory is parsed automatically from the dataset slug. Pass `--out_dir` to override.

Verification gate — confirm zip landed:
```bash
ls -lh episodes/2026-05-24/*.zip
```

## Step 2 — Encode (one-time per date, ~600s)

Reads the zip and writes sharded `.npz` files of pre-encoded features. Skips automatically if shards already exist.

```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.bc_encode \
  --episodes_dir episodes/2026-05-24
# → episodes/2026-05-24/shards/shard_0000.npz ...
```

Verification gate — confirm shards wrote:
```bash
ls episodes/2026-05-24/shards/ | head
```

Use `--force` to re-encode from scratch. Use `--winner_only` to encode only the winning agent's actions.

## Step 3 — Behavior Cloning

Streams shards per epoch; starts immediately. Pass multiple paths to `--episode_dirs` to train across several dates.

```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.bc_train \
  --episode_dirs episodes/2026-05-24 \
  --out_dir artifacts/bc \
  --run_name bc_pretrain_may24_all
# → artifacts/bc/bc_pretrain_may24_all/ckpt_last.pkl
```

Default settings: 10 epochs, batch_size 2048, lr 3e-4, W&B enabled. Pass `--no_wandb` to disable.

Multi-day example:
```bash
--episode_dirs episodes/2026-05-24 episodes/2026-05-25
```

Compatibility aliases: `--episodes_dirs` and `--episodes_dir` are accepted by `src_jax.bc_train`.

## Step 4 — PPO Self-Play From BC Checkpoint

```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.train \
  --resume artifacts/bc/bc_pretrain_may24_all/ckpt_last.pkl
# → artifacts/jax/orbit_wars_jax_ppo/
```

Default config: `src_jax/configs/default_cfg.yaml` (1000 updates, checkpoint every 50, opponent: self).

## Monitoring Notes

- Use `--no-capture-output` with `conda run` or output is buffered until the process exits.
- Encode progress is printed every 50 episodes.
- BC training prints `epoch=N loss=… tgt=… shp=… Xs` after each epoch (no loading wait).
- W&B run is tagged with `n_shards` and `episodes_dirs` for traceability.
