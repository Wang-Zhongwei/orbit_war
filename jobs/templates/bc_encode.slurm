#!/usr/bin/env bash
#SBATCH --job-name=bc_encode
#SBATCH --output=./jobs/logs/bc_encode-%j.out
#SBATCH --error=./jobs/logs/bc_encode-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
if [[ -n "${SHARD_TAG:-}" ]]; then
  args+=(--shard_tag "${SHARD_TAG}")
fi
if [[ -n "${EPISODES_PER_SHARD:-}" ]]; then
  args+=(--episodes_per_shard "${EPISODES_PER_SHARD}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=("--force")
fi

args+=(--workers "${SLURM_CPUS_PER_TASK:-1}")

# NB: oversubscription control lives in src_jax/bc_data._worker_init, which pins each
# pool worker to a disjoint *block* of cores. Do NOT cap threads here with
# XLA_FLAGS=--xla_cpu_multi_thread_eigen=false / OMP_NUM_THREADS=1 — that forces XLA to
# compile single-threaded, which on this scanned simulator stalls each worker's one-time
# compile for many minutes and the job makes no progress. Let each worker compile
# multi-threaded across its core block instead.

echo "Running from: ${REPO_ROOT}"

echo "Running command: uv run python -m src_jax.bc_encode ${args[*]}"
uv run python -m src_jax.bc_encode "${args[@]}"
echo "Done."
