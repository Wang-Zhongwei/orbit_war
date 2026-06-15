#!/usr/bin/env bash
# Smoke-test the N-player PPO rollout change (num_agents = 2 and 4).
# Tiny + CPU-only; pinned to the ppo4p worktree so it exercises the edited
# src_jax. Submit on coe-hpc3 (per global policy):
#   ssh coe-hpc3 'cd /home/018270239/orbit_war && \
#                 sbatch jobs/templates/smoke_4player.slurm'
#SBATCH --job-name=smoke_4player
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/%x-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
_NCPU="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${_NCPU}"

echo "Host: $(hostname) | repo=${REPO_ROOT} | cpus=${_NCPU}"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo

uv run python -m tools.smoke_4player --config configs/ppo_low_kl.yaml

echo "smoke_4player done."
