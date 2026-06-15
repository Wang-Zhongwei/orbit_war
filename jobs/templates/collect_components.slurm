#!/usr/bin/env bash
#SBATCH --job-name=collect_comp
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/collect_comp-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/collect_comp-%j.err
#SBATCH --time=0:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Localize WHERE the ~830 ms/step collect cost goes (encode x2 / fwd x2 / physics),
# and settle compute-bound vs stall-bound by sampling GPU utilization concurrently.

export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
_NCPU="${SLURM_CPUS_PER_TASK:-8}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU}"
export OMP_NUM_THREADS="${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
CONFIG="${CONFIG:-configs/ppo_low_kl.yaml}"
DMON="${CLAUDE_JOB_DIR:-/tmp}/dmon_${SLURM_JOB_ID:-comp}.log"

echo "Host: $(hostname) | cpus=${_NCPU} | config=${CONFIG}"
nvidia-smi -L || true
echo

# sample GPU SM-utilization + power once/sec in the background for the whole run
nvidia-smi dmon -d 1 -s pu -o T > "${DMON}" 2>/dev/null &
DMON_PID=$!

uv run python -m tools.collect_components --config "${CONFIG}"

kill "${DMON_PID}" 2>/dev/null || true

echo
echo "=== GPU utilization during run (nvidia-smi dmon -s pu) ==="
echo "(sm% column: ~100 => compute-bound, command-buffers/unroll cannot help;"
echo " low/spiky => stall-bound). tail of ${DMON}:"
tail -n 40 "${DMON}" 2>/dev/null || true
echo
echo "sm% distribution (col with header 'sm'):"
awk 'NR>2 && $1!="#" {print $5}' "${DMON}" 2>/dev/null | sort -n | awk '
  {a[NR]=$1; s+=$1}
  END{if(NR>0){print "  n="NR"  min="a[1]"  median="a[int(NR/2)]"  max="a[NR]"  mean="s/NR}}' || true

echo
echo "collect_components done."
