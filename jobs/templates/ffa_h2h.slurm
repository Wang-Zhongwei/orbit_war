#!/usr/bin/env bash
# 4-player FFA head-to-head between two JAX checkpoints (1-vs-3 line-ups with a
# probability split + seat shuffle; reports each agent's win rate).
# CPU inference (matches the eval_mp pattern). Submit on coe-hpc3:
#   ssh coe-hpc3
#   sbatch --export=ALL,CKPT_A=...,CKPT_B=...,ROUNDS=256,WORKERS=16 \
#          /home/018270239/orbit_war/.claude/worktrees/v7arch/jobs/templates/ffa_h2h.slurm
#SBATCH --job-name=ffa_h2h
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/018270239/orbit_war/.claude/worktrees/v7arch/jobs/logs/ffa_h2h-%j.out
#SBATCH --error=/home/018270239/orbit_war/.claude/worktrees/v7arch/jobs/logs/ffa_h2h-%j.err

set -euo pipefail

REPO_ROOT="/home/018270239/orbit_war/.claude/worktrees/v7arch"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Persistent compile cache so each worker's agent JITs aren't recompiled cold
# every run (warmup is N_workers x 2 agents otherwise).
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.jax_cache"

CKPT_A="${CKPT_A:-artifacts/ppo/ppo_4player/ckpt_000500.pkl}"
CKPT_B="${CKPT_B:-artifacts/ppo/ppo_mixed/ckpt_000500.pkl}"
NAME_A="${NAME_A:-ffa}"
NAME_B="${NAME_B:-mixed}"
ROUNDS="${ROUNDS:-256}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PROB_A="${PROB_A:-0.5}"
OUT="${OUT:-results/ffa_h2h_${NAME_A}_vs_${NAME_B}.csv}"

echo "Running from: ${REPO_ROOT} ($(git rev-parse --abbrev-ref HEAD))"
echo "A=${NAME_A}:${CKPT_A}  B=${NAME_B}:${CKPT_B}  rounds=${ROUNDS} workers=${WORKERS}"

uv run python -m tools.ffa_h2h \
  --ckpt-a "${CKPT_A}" --ckpt-b "${CKPT_B}" \
  --name-a "${NAME_A}" --name-b "${NAME_B}" \
  --rounds "${ROUNDS}" --workers "${WORKERS}" \
  --prob-a-singleton "${PROB_A}" \
  --out "${OUT}"

echo "ffa_h2h done. CSV: ${OUT}"
