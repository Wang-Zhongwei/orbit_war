#!/usr/bin/env bash
#SBATCH --job-name=bc_sweep_train
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/bc_sweep_train-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/bc_sweep_train-%j.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Single-axis BC hyperparameter sweep — TRAIN stage (one run, one swept knob).
# All knobs default to configs/bc_entity_policy_12.yaml; override exactly one via env var.
#
# Do NOT module-load nvhpc-hpcx-cuda12 — it overrides the venv-bundled nvidia-*
# CUDA libs and breaks JAX. The uv venv carries everything needed.
#
# Required env var:
#   RUN_NAME           checkpoint/run identifier -> artifacts/bc/<RUN_NAME>/
# Optional env vars (set ONE to vary that axis; unset = config default):
#   HIDDEN_SIZE        model.hidden_size       (config default 256)
#   CANDIDATE_COUNT    env.candidate_count K   (config default 12; <= POOL_K=24)
#   SKILL_CLIP_RATIO   bc.skill_clip_ratio     (config default 1.0)
#   RECENCY_HALF_LIFE  bc.recency_half_life    (config default 14.0)
#   LOSER_WEIGHT       bc.loser_weight         (config default 0.1)
#   SHARD_TAG          bc.shard_tag            (read shards_<tag>/ dirs; default "" = shards/)
#   CONFIG             base config             (default configs/bc_entity_policy_12.yaml)
#   OUT_DIR            checkpoint root         (default artifacts/bc)
#   EPISODE_DIRS       space-separated dirs    (default: all 2026-05-* and 2026-06-* date dirs)

export WANDB_MODE=offline

# Allocate GPU memory on demand rather than JAX's default ~75% up-front grab. The launcher
# runs these serially (LANES=1) since gpuqs oversubscribes GPUs and two of these jobs on one
# H100 OOM; this is just a safety net if anyone raises LANES.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"
echo "Host: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

: "${RUN_NAME:?Set RUN_NAME (e.g. sweep_hidden_512)}"
CONFIG="${CONFIG:-configs/bc_entity_policy_12.yaml}"
OUT_DIR="${OUT_DIR:-artifacts/bc}"

# Default to the entire dataset: every 2026-05-* / 2026-06-* date dir (excludes
# helper symlinks like episodes/_qtest_0604). Pass EPISODE_DIRS to override.
if [[ -z "${EPISODE_DIRS:-}" ]]; then
  shopt -s nullglob
  _dirs=(episodes/2026-0[56]-[0-3][0-9])
  shopt -u nullglob
else
  read -ra _dirs <<< "${EPISODE_DIRS}"
fi
echo "Training on ${#_dirs[@]} date dirs: ${_dirs[*]}"

args=()
args+=(--episode_dirs "${_dirs[@]}")
args+=(--config "${CONFIG}")
args+=(--use_shard)
args+=(--out_dir "${OUT_DIR}")
args+=(--run_name "${RUN_NAME}")

# Vary exactly one knob (whichever env var is set).
[[ -n "${HIDDEN_SIZE:-}"       ]] && args+=(--hidden_size       "${HIDDEN_SIZE}")
[[ -n "${CANDIDATE_COUNT:-}"   ]] && args+=(--candidate_count   "${CANDIDATE_COUNT}")
[[ -n "${SKILL_CLIP_RATIO:-}"  ]] && args+=(--skill_clip_ratio  "${SKILL_CLIP_RATIO}")
[[ -n "${RECENCY_HALF_LIFE:-}" ]] && args+=(--recency_half_life "${RECENCY_HALF_LIFE}")
[[ -n "${LOSER_WEIGHT:-}"      ]] && args+=(--loser_weight      "${LOSER_WEIGHT}")
[[ -n "${SHARD_TAG:-}"         ]] && args+=(--shard_tag         "${SHARD_TAG}")

echo "Running command: uv run python -m src_jax.bc_train ${args[*]}"
uv run python -m src_jax.bc_train "${args[@]}"
echo "Training done. Checkpoints in ${OUT_DIR}/${RUN_NAME}/"

echo "Syncing W&B offline runs from login node..."
ssh coe-hpc1 "cd ${REPO_ROOT} && .venv/bin/wandb sync wandb/offline-run-*/" || true
echo "Done."
