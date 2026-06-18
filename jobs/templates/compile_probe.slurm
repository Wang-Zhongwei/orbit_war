#!/usr/bin/env bash
#SBATCH --job-name=compile_probe
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/compile_probe-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/compile_probe-%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Localize WHY the first JAX compile is slow. Runs tools/compile_probe.py twice:
#   1) CPU backend  — times platform-independent work (tracing + HLO + XLA passes).
#                     A fn slow HERE => the HLO graph is huge (shrink the graph).
#   2) GPU backend  — adds XLA GPU codegen + ptxas. Fast on CPU but slow HERE
#                     => ptxas/codegen bound (more cores / fewer-bigger kernels).
# Both are AOT compile-only (.lower().compile()); nothing executes, no GPU OOM.
#
# Submit on coe-hpc3 (per global policy):
#   ssh coe-hpc3 'cd /home/018270239/orbit_war && sbatch jobs/templates/compile_probe.slurm'

export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Disable the persistent cache so we measure a real cold compile every time.
export JAX_COMPILATION_CACHE_DIR=""
export JAX_DISABLE_COMPILATION_CACHE=1

_NCPU="${SLURM_CPUS_PER_TASK:-32}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU} --xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS="${_NCPU}"
export OPENBLAS_NUM_THREADS="${_NCPU}"
export MKL_NUM_THREADS="${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
CONFIG="${CONFIG:-configs/ppo_from_bc_loser0.yaml}"

echo "Host: $(hostname) | cpus=${_NCPU} | config=${CONFIG}"
nvidia-smi -L || true
echo

echo "############################################################"
echo "# PASS 1 — CPU backend (platform-independent compile cost)  "
echo "############################################################"
JAX_PLATFORMS=cpu uv run python -m tools.compile_probe --config "${CONFIG}"

echo
echo "############################################################"
echo "# PASS 2 — GPU backend (adds XLA GPU codegen + ptxas)       "
echo "############################################################"
uv run python -m tools.compile_probe --config "${CONFIG}"

echo
echo "compile_probe done."
