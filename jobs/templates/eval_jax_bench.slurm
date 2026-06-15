#!/usr/bin/env bash
#SBATCH --job-name=eval_jax_bench
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/eval_jax_bench-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/eval_jax_bench-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT} on $(hostname)"
echo "========================================================================"
uv run python /home/018270239/.claude/jobs/1ef78571/tmp/bench_eval_jax.py
echo "========================================================================"
echo "eval_jax bench finished."
