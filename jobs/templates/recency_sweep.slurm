#!/usr/bin/env bash
#SBATCH --job-name=recency_sweep
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/recency_sweep-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/recency_sweep-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Do NOT load nvhpc-hpcx-cuda12 here — it overrides the venv-bundled nvidia-*
# CUDA libraries and breaks JAX. The uv venv carries everything needed.

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"

export WANDB_MODE=offline

echo "Running from: ${REPO_ROOT}"
echo "Recency-weighting screening sweep (held-out 2026-05-15)"
nvidia-smi -L || true
.venv/bin/python -m experiments.recency_sweep
echo "Done."
