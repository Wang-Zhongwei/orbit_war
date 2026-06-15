#!/usr/bin/env bash
#SBATCH --job-name=panel_4p_eval
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/panel_4p_eval-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/panel_4p_eval-%j.err
#SBATCH --time=00:40:00
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

WT="${REPO_ROOT}"
ROUNDS="${ROUNDS:-64}"

echo "=================== ppo_4player ckpt_000425 vs sub_v8 (1-vs-3 FFA, ${ROUNDS}r) ==================="
uv run python arena.py panel \
  --eval jax --opponents sub_v8 --rounds "${ROUNDS}" --num-players 4 \
  --jax-checkpoint artifacts/ppo/ppo_4player/ckpt_000425.pkl --jax-src-root "${WT}" \
  --out results/panel_4p_ppo4player425_vs_sub_v8.csv

echo "=================== ppo_mixed ckpt_000450 vs sub_v8 (1-vs-3 FFA, ${ROUNDS}r) ==================="
uv run python arena.py panel \
  --eval jax --opponents sub_v8 --rounds "${ROUNDS}" --num-players 4 \
  --jax-checkpoint artifacts/ppo/ppo_mixed/ckpt_000450.pkl --jax-src-root "${WT}" \
  --out results/panel_4p_ppomixed450_vs_sub_v8.csv

echo "=================== panel 4p eval finished ==================="
