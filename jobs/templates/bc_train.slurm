#!/usr/bin/env bash
#SBATCH --job-name=bc_train
#SBATCH --output=./jobs/logs/bc_train-%j.out
#SBATCH --error=./jobs/logs/bc_train-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=condo
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your-email@example.com>

set -euo pipefail

# Do NOT load nvhpc-hpcx-cuda12 here — it overrides the venv-bundled nvidia-*
# CUDA libraries (cublas, cudnn, etc.) and causes JAX to fail with
# "the library was not initialized". The uv venv carries everything needed.
export WANDB_MODE=offline

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
cd "${REPO_ROOT}"

args=()
if [[ -n "${EPISODE_DIRS:-}" ]]; then
  # Space-separated list of episode directories
  read -ra _dirs <<< "${EPISODE_DIRS}"
  args+=(--episode_dirs "${_dirs[@]}")
fi
if [[ -n "${CONFIG:-}" ]]; then
  args+=(--config "${CONFIG}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  args+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  args+=(--run_name "${RUN_NAME}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  args+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${LR:-}" ]]; then
  args+=(--lr "${LR}")
fi
if [[ -n "${TOP_K_EPISODES:-}" ]]; then
  args+=(--top_k_episodes "${TOP_K_EPISODES}")
fi
if [[ "${WINNER_ONLY:-0}" == "1" ]]; then
  args+=(--winner_only)
fi
if [[ "${AUGMENT:-0}" == "1" ]]; then
  args+=(--augment)
fi
if [[ "${USE_SHARD:-0}" == "1" ]]; then
  args+=(--use_shard)
else
  # Streaming mode: reserve one CPU for the main process
  args+=(--no_use_shard)
  args+=(--encode_workers "$(( SLURM_NTASKS - 1 ))")
fi
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  args+=(--no_wandb)
fi

echo "Running from: ${REPO_ROOT}"

echo "Running command: uv run python -m src_jax.bc_train ${args[*]}"
uv run python -m src_jax.bc_train "${args[@]}"
echo "Done."

echo "Syncing W&B offline runs from ${SLURM_SUBMIT_HOST}"
ssh coe-hpc1 "cd ${REPO_ROOT} && .venv/bin/wandb sync wandb/offline-run-*/"
echo "Done."
