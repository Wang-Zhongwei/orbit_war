#!/usr/bin/env bash
#SBATCH --job-name=bc_encode
#SBATCH --output=./jobs/logs/bc_encode-%j.out
#SBATCH --error=./jobs/logs/bc_encode-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your-email@example.com>

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
cd "${REPO_ROOT}"

args=()
if [[ -n "${EPISODES_DIR:-}" ]]; then
  args+=(--episodes_dir "${EPISODES_DIR}")
fi
if [[ "${RECURSIVE:-0}" == "1" ]]; then
  args+=(--recursive)
fi
if [[ -n "${CONFIG:-}" ]]; then
  args+=(--config "${CONFIG}")
fi
if [[ -n "${EPISODES_PER_SHARD:-}" ]]; then
  args+=(--episodes_per_shard "${EPISODES_PER_SHARD}")
fi
if [[ "${WINNER_ONLY:-0}" == "1" ]]; then
  args+=(--winner_only)
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=("--force")
fi

args+=(--workers "${SLURM_NTASKS:-1}")


echo "Running from: ${REPO_ROOT}"

echo "Running command: uv run python -m src_jax.bc_encode ${args[*]}"
uv run python -m src_jax.bc_encode "${args[@]}"
echo "Done.
