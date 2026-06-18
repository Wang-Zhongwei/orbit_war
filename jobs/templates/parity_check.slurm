#!/usr/bin/env bash
#SBATCH --job-name=parity_check
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/parity_check-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/parity_check-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1

set -euo pipefail
export WANDB_MODE=offline
cd /home/018270239/orbit_war
echo "K-pool / live-encoder parity check for assemble_for_k"
.venv/bin/python -u -m test.test_kmax_pool_parity --episodes_dir episodes/2026-05-01 --n 4
echo "Done."
