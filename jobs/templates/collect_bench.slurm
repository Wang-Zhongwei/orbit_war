#!/usr/bin/env bash
#SBATCH --job-name=collect_bench
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/collect_bench-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/collect_bench-%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# A/B the runtime cost of the launch-bound collect step: baseline vs XLA command
# buffers capturing the WHILE computation (physics while-loops) into CUDA graphs.
# Both passes EXECUTE collect repeatedly (tools/collect_bench.py) and report median
# ms; compare the two "=== collect: median ... ms ===" lines.
#
# Submit on coe-hpc3 (per global policy):
#   ssh coe-hpc3 'cd /home/018270239/orbit_war && sbatch jobs/templates/collect_bench.slurm'

export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false

_NCPU="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${_NCPU}"

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
CONFIG="${CONFIG:-configs/ppo_low_kl.yaml}"
CMDBUF="${CMDBUF:-FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE}"

echo "Host: $(hostname) | cpus=${_NCPU} | config=${CONFIG}"
nvidia-smi -L || true
echo

echo "############################################################"
echo "# PASS 1 — BASELINE (no command-buffer flag)               "
echo "############################################################"
XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU}" \
    uv run python -m tools.collect_bench --config "${CONFIG}"

echo
echo "############################################################"
echo "# PASS 2 — command buffers incl. WHILE: ${CMDBUF}          "
echo "############################################################"
DUMPDIR="${CLAUDE_JOB_DIR:-/tmp}/xladump_${SLURM_JOB_ID:-bench}"
rm -rf "${DUMPDIR}"; mkdir -p "${DUMPDIR}"
XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${_NCPU} --xla_gpu_enable_command_buffer=${CMDBUF} --xla_dump_to=${DUMPDIR}" \
    uv run python -m tools.collect_bench --config "${CONFIG}"

echo
echo "############################################################"
echo "# CAPTURE CHECK — did the WHILE loops actually get captured?"
echo "############################################################"
# If command buffers are a silent no-op, PASS 2 median won't drop. This grep
# proves capture is real: command-buffer thunks (and While/Conditional nodes
# inside them) show up in the post-optimization HLO / thunk dump.
echo "command_buffer occurrences in XLA dump:"
grep -rIl -i "command.buffer\|commandbuffer" "${DUMPDIR}" 2>/dev/null | head || true
grep -rIi "command.buffer\|commandbuffer" "${DUMPDIR}" 2>/dev/null | wc -l || true
echo "While/Conditional command mentions:"
grep -rIi "WhileCmd\|while_cmd\|command_buffer.*while\|while.*command_buffer" "${DUMPDIR}" 2>/dev/null | wc -l || true

echo
echo "collect_bench A/B done. Compare the two '=== collect: median ... ms ===' lines."
echo "If PASS 2 < PASS 1 -> the WHILE command-buffer flag is a real win."
