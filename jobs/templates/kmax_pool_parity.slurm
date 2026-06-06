#!/usr/bin/env bash
#SBATCH --job-name=kmax_parity
#SBATCH --output=./jobs/logs/kmax_parity-%j.out
#SBATCH --error=./jobs/logs/kmax_parity-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

EPISODES_DIR="${EPISODES_DIR:-episodes/2026-05-01}"
N="${N:-8}"
K="${K:-8}"

export JAX_PLATFORMS=cpu
echo "Parity test: pool@POOL_K vs live encoder K=${K}, episodes=${EPISODES_DIR}, n=${N}"
uv run python -m test.test_kmax_pool_parity --episodes_dir "${EPISODES_DIR}" --n "${N}" --K "${K}"
echo "Done."
