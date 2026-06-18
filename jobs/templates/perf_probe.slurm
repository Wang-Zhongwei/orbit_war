#!/usr/bin/env bash
#SBATCH --job-name=perf_probe
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/perf_probe-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/perf_probe-%j.err
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
_NCPU="${SLURM_CPUS_PER_TASK:-16}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Host: $(hostname)"; nvidia-smi -L || true
uv run python -m tools.perf_probe
echo "perf_probe done."
