#!/usr/bin/env bash
#SBATCH --job-name=eval_jax_smoke
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/eval_jax_smoke-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/eval_jax_smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT} on $(hostname)"
echo "========================================================================"
uv run python /home/018270239/.claude/jobs/1ef78571/tmp/smoke_eval_jax.py
echo "========================================================================"
echo "eval_jax smoke finished."
