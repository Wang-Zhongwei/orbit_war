---
name: orbit-wars-training-pipeline
description: Run the Orbit Wars JAX training workflow: download Kaggle episode replays with fetch_daily_episodes.py, then either pre-encode into shards with src_jax.bc_encode + behavior-clone with src_jax.bc_train, OR stream directly from zip with src_jax.bc_train --stream (no shards, best for large 1-epoch runs), then start PPO self-play from the BC checkpoint. Use when the user asks to run, resume, monitor, or document the Orbit Wars data/pretraining/self-play pipeline.
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

Downloads each day's dataset as a single zip into `episodes/<date>/archive.zip`.
Already-downloaded days are skipped, so it is safe to re-run for daily backfill.

```bash
# One specific day:
conda run --no-capture-output -n torch-gpu python -u fetch_daily_episodes.py \
  --start 2026-05-24 --end 2026-05-24
# → episodes/2026-05-24/archive.zip  (+ extracted manifest.csv)

# Or backfill a range (omit --end for "through today"):
conda run --no-capture-output -n torch-gpu python -u fetch_daily_episodes.py \
  --start 2026-05-01
```

Each day's dataset is `kaggle/orbit-wars-episodes-<date>`; the date subdirectory
is derived from the date. Pass `--out_dir` to override, `--force` to re-download.

Verification gate — confirm zip landed:
```bash
ls -lh episodes/2026-05-24/*.zip
```

## Step 2 — Behavior Cloning

Two modes depending on how many epochs you need:

### Stream mode (recommended for large datasets, 1 epoch)

Encodes episodes on the fly directly from the zip — no intermediate shard files, no pre-encode step.
Each episode is encoded once, fed to training, and discarded. Best when the dataset is large enough
that 1 epoch provides sufficient coverage.

The default config (`configs/bc_default.yaml`) already sets `stream: true` and `epochs: 1`,
so you can usually just pass `--episode_dirs`:

```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.bc_train \
  --episode_dirs episodes/2026-05-24 \
  --run_name bc_stream_may24
# → artifacts/bc/bc_stream_may24/ckpt_last.pkl
```

To use a custom config:
```bash
--config configs/bc_default.yaml
```

CLI args override config file values when provided. All BC fields are optional:
- `--epochs N` / `--batch_size N` / `--lr F` — training hyperparams
- `--stream` / `--no_stream` — enable or disable on-the-fly encoding
- `--stream_buffer N` — episodes to buffer before a training pass (default: 200). Larger = better shuffle within buffer.
- `--winner_only` — only learn from the winning agent's actions.
- `--out_dir DIR` / `--run_name NAME` — output paths
- `--episode_dirs` accepts multiple dates: `episodes/2026-05-24 episodes/2026-05-25`

Progress is printed every `stream_buffer * 5` episodes and after each epoch.

### Shard mode (for multi-epoch runs over smaller datasets)

Encode once, train many epochs cheaply from pre-built `.npz` shards.

**Encode (one-time per date, ~600s):**
```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.bc_encode \
  --episodes_dir episodes/2026-05-24
# → episodes/2026-05-24/shards/shard_0000.npz ...
```

Use `--force` to re-encode from scratch. Use `--winner_only` to encode only the winning agent's actions.

Verification gate:
```bash
ls episodes/2026-05-24/shards/ | head
```

**Train:**
```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.bc_train \
  --episode_dirs episodes/2026-05-24 \
  --out_dir artifacts/bc \
  --run_name bc_pretrain_may24_all
# → artifacts/bc/bc_pretrain_may24_all/ckpt_last.pkl
```

Default settings: 10 epochs, batch_size 2048, lr 3e-4, W&B enabled. Pass `--no_wandb` to disable.

Compatibility aliases: `--episodes_dirs` and `--episodes_dir` are accepted by `src_jax.bc_train`.

## Step 3 — PPO Self-Play From BC Checkpoint

```bash
conda run --no-capture-output -n torch-gpu python -u -m src_jax.train \
  --resume artifacts/bc/bc_pretrain_may24_all/ckpt_last.pkl
# → artifacts/jax/orbit_wars_jax_ppo/
```

Default config: `configs/ppo_default.yaml` (1000 updates, checkpoint every 50, opponent: self).

## Monitoring Notes

- Use `--no-capture-output` with `conda run` or output is buffered until the process exits.
- Stream mode prints `ep=N samples=N steps=N` every 1000 episodes (with default buffer=200), then a full epoch summary.
- Shard mode prints `epoch=N loss=… tgt=… shp=… Xs` after each epoch.
- W&B run is tagged with `episodes_dirs` and (in shard mode) `n_shards` for traceability.
