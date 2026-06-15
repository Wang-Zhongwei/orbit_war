#!/usr/bin/env bash
#SBATCH --job-name=ppo_train_4p
#SBATCH --output=/home/018270239/orbit_war/.claude/worktrees/ppo4p/jobs/logs/ppo_train_4p-%j.out
#SBATCH --error=/home/018270239/orbit_war/.claude/worktrees/ppo4p/jobs/logs/ppo_train_4p-%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# 4-player (FFA) PPO training. Pinned to the ppo4p worktree (branch ppo-4player)
# so it runs the 4-player code — the shared ppo_train.slurm hardcodes the v7arch
# worktree and has NO num_agents support, so it would silently run 2-player (or,
# with the stale path, fail at `cd`).
#
# Do NOT module-load nvhpc-hpcx-cuda12 — it overrides the venv-bundled nvidia-*
# CUDA libs and breaks JAX. The uv venv carries everything needed.
#
# Submit on coe-hpc3 (per global policy: submit slurm jobs on coe-hpc3 only):
#   ssh coe-hpc3
#   sbatch /home/018270239/orbit_war/.claude/worktrees/ppo4p/jobs/templates/ppo_train_4player.slurm
#   sbatch --export=ALL,RUN_NAME=...,TOTAL_UPDATES=... <this script>
#
# Optional env vars (all have defaults):
#   CONFIG     PPO config yaml   (default configs/ppo_4player.yaml)
#   RESUME     warm-start ckpt   (default: from config's `resume:`)
#   RUN_NAME / TOTAL_UPDATES / REF_POLICY — override config values
#
# NOTE: eval still runs 1v1 benchmark matches (a sanity signal, not a 4-player
# eval). The "v7arch worktree not found ... loading against the live src_jax"
# UserWarning is expected and benign here: ppo4p shares v7arch's 30-dim arch, so
# the live tree is the matching arch for sub_v6/v7/v8/v81.

export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

_NCPU="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU} --xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS="${_NCPU}"
export OPENBLAS_NUM_THREADS="${_NCPU}"
export MKL_NUM_THREADS="${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war/.claude/worktrees/ppo4p"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Host: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONFIG="${CONFIG:-configs/ppo_4player.yaml}"

args=()
args+=(--config "${CONFIG}")
[[ -n "${RUN_NAME:-}"      ]] && args+=(--run_name      "${RUN_NAME}")
[[ -n "${TOTAL_UPDATES:-}" ]] && args+=(--total_updates "${TOTAL_UPDATES}")
[[ -n "${REF_POLICY:-}"    ]] && args+=(--ref_policy    "${REF_POLICY}")
[[ -n "${RESUME:-}"        ]] && args+=(--resume        "${RESUME}")

echo "Running command: uv run python -m src_jax.train ${args[*]}"
uv run python -m src_jax.train "${args[@]}"
echo "Training done. Checkpoints in artifacts/ppo/<run_name>/"

echo "Syncing W&B offline runs from login node..."
ssh coe-hpc1 "cd ${REPO_ROOT} && .venv/bin/wandb sync wandb/offline-run-*/" || true
echo "Done."
