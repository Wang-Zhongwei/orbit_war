#!/usr/bin/env bash
#SBATCH --job-name=ppo_train
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/ppo_train-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/ppo_train-%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# PPO training warm-started from a BC checkpoint, with KL regularization toward the
# frozen BC policy (kl_coef > 0 in the config). The trainer (src_jax/train.py) loads
# --resume for the warm-start params AND, since kl_coef > 0 and no --ref_policy is
# passed, uses those same params as the frozen KL reference.
#
# Do NOT module-load nvhpc-hpcx-cuda12 — it overrides the venv-bundled nvidia-* CUDA
# libs and breaks JAX. The uv venv carries everything needed.
#
# Submit on coe-hpc3 (per global policy: submit slurm jobs on coe-hpc3 only):
#   ssh coe-hpc3
#   sbatch jobs/templates/ppo_train.slurm                                  # defaults below
#   sbatch --export=ALL,CONFIG=...,RESUME=...,RUN_NAME=... jobs/templates/ppo_train.slurm
#
# Optional env vars (all have defaults):
#   CONFIG     PPO config yaml   (default configs/ppo_from_bc_loser0.yaml)
#   RESUME     BC/PPO checkpoint (default ./artifacts/bc/sweep_loser_0_0/ckpt_000001_shard1700.pkl)
#   RUN_NAME   run/ckpt name     (default from config: ppo_from_bc_loser0)
#   TOTAL_UPDATES  override ppo.total_updates
#   REF_POLICY     explicit KL reference ckpt (default: the RESUME checkpoint, via train.py)
#   JAX_SRC_ROOT   pin the trained arch to a worktree's src_jax (default: live tree).
#                  e.g. JAX_SRC_ROOT=$PWD/.claude/worktrees/v9arch to train the previous
#                  (v9) arch now that the live tree is the v7 arch.
#                  train.py re-execs via tools/train_pinned.py to pin it.

export WANDB_MODE=offline
# Allocate GPU memory on demand rather than JAX's default ~75% up-front grab.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Flush stdout per line. Without this, Python block-buffers stdout (~4-8KB) when
# it's redirected to the slurm .out file, so per-update progress is invisible for
# ~1-2h until the buffer fills — looks like a hang/endless compile but isn't.
export PYTHONUNBUFFERED=1

# --- XLA compile speed -------------------------------------------------------
# XLA's GPU compilation (LLVM + per-kernel ptxas) is CPU-bound and parallelizes
# across cores. With the old cpus-per-task=2 the first compile of the vmapped
# rollout took 40+ min (single-core-bound); a GCP VM with more vCPUs did it in
# minutes. Give XLA the allocated cores explicitly. The compute node also exposes
# all 192 host cores via os.cpu_count(), so cap the math/thread pools to the
# cgroup allocation to avoid oversubscription thrash during compile.
_NCPU="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU} --xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS="${_NCPU}"
export OPENBLAS_NUM_THREADS="${_NCPU}"
export MKL_NUM_THREADS="${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"
echo "Host: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONFIG="${CONFIG:-configs/ppo_from_bc_loser0.yaml}"

args=()
args+=(--config "${CONFIG}")
[[ -n "${RUN_NAME:-}"      ]] && args+=(--run_name      "${RUN_NAME}")
[[ -n "${TOTAL_UPDATES:-}" ]] && args+=(--total_updates "${TOTAL_UPDATES}")
[[ -n "${REF_POLICY:-}"    ]] && args+=(--ref_policy    "${REF_POLICY}")
[[ -n "${RESUME:-}"        ]] && args+=(--resume        "${RESUME}")
[[ -n "${JAX_SRC_ROOT:-}"  ]] && args+=(--jax-src-root  "${JAX_SRC_ROOT}")

echo "Running command: uv run python -m src_jax.train ${args[*]}"
uv run python -m src_jax.train "${args[@]}"
echo "Training done. Checkpoints in artifacts/ppo/<run_name>/"

echo "Syncing W&B offline runs from login node..."
ssh coe-hpc1 "cd ${REPO_ROOT} && .venv/bin/wandb sync wandb/offline-run-*/" || true
echo "Done."
