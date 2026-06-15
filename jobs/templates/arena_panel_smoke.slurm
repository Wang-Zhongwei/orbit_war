#!/usr/bin/env bash
#SBATCH --job-name=arena_panel_smoke
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/arena_panel_smoke-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/arena_panel_smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT} on $(hostname)"

CKPT="artifacts/ppo/ppo_low_kl_0615/ckpt_last.pkl"
WT=".claude/worktrees/v7arch"
ROUNDS="${ROUNDS:-16}"

echo "=================== 1v1 panel (sub_v8 batched, sub_v52 fallback) ==================="
uv run python arena.py panel \
  --eval jax --opponents sub_v8 sub_v52 --rounds "${ROUNDS}" \
  --jax-checkpoint "${CKPT}" --jax-src-root "${WT}" \
  --out results/panel_smoke_1v1.csv

echo "=================== 1-vs-3 FFA panel (sub_v8 batched, sub_v52 fallback) ==================="
uv run python arena.py panel \
  --eval jax --opponents sub_v8 sub_v52 --rounds "${ROUNDS}" --ffa \
  --jax-checkpoint "${CKPT}" --jax-src-root "${WT}" \
  --out results/panel_smoke_ffa.csv

echo "=================== arena panel smoke finished ==================="
