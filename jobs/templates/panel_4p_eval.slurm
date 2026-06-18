#!/usr/bin/env bash
#SBATCH --job-name=panel_4p_eval
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/panel_4p_eval-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/panel_4p_eval-%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=gpuqs,condo
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

ROUNDS="${ROUNDS:-64}"
CKPT="${CKPT:-artifacts/ppo/ppo_4player/ckpt_000425.pkl}"
OPPONENTS="${OPPONENTS:-sub_v8}"

CKPT_DIR="$(basename "$(dirname "${CKPT}")")"
CKPT_BASE="$(basename "${CKPT}" .pkl)"
CKPT_NAME="${CKPT_DIR}_${CKPT_BASE}"
OUT_CSV="${OUT_CSV:-results/panel_4p_${CKPT_NAME}_vs_${OPPONENTS}.csv}"

echo "=================== ${CKPT_NAME} vs ${OPPONENTS} (1-vs-3, ${ROUNDS}r) ==================="
uv run python arena.py panel \
  --eval jax --opponents "${OPPONENTS}" --rounds "${ROUNDS}" --num-players 4 \
  --jax-checkpoint "${CKPT}" \
  --out "${OUT_CSV}"

echo "=================== panel 4p eval finished ==================="
