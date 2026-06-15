#!/usr/bin/env bash
#SBATCH --job-name=stride_acc
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/stride_acc-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/stride_acc-%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
_NCPU="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU}"

cd /home/018270239/orbit_war
CONFIG="${CONFIG:-configs/ppo_low_kl.yaml}"
echo "Host: $(hostname) | config=${CONFIG}"
nvidia-smi -L || true
echo
uv run python -m tools.stride_accuracy --config "${CONFIG}" --envs 256 --warm 120 \
    --schedules anchored 1-45:2 1-45:3
echo
echo "stride_accuracy done."
