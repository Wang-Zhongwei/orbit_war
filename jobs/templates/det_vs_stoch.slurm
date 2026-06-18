#!/usr/bin/env bash
#SBATCH --job-name=det_vs_stoch
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/det_vs_stoch-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/det_vs_stoch-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Same checkpoint, deterministic vs stochastic sampling. Both agents built from
# the live src_jax (incl. comet evac), seats alternated for fairness.
#
#   sbatch --export=ALL,CKPT=...,ROUNDS=16 jobs/templates/det_vs_stoch.slurm

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"

CKPT="${CKPT:-artifacts/ppo/ppo_mixed_0615/ckpt_000900.pkl}"
ROUNDS="${ROUNDS:-16}"
REPLAY="${REPLAY:-results/h2h/det_vs_stoch_replay.json}"
mkdir -p "$(dirname "${REPLAY}")"

uv run python tools/det_vs_stoch.py \
  --ckpt "${CKPT}" \
  --rounds "${ROUNDS}" \
  --base-seed 1000 \
  --replay "${REPLAY}"

echo "Done. Replay: ${REPLAY}"
