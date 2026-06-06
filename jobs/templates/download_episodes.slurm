#!/usr/bin/env bash
#SBATCH --job-name=download_daily_episodes
#SBATCH --output=./jobs/logs/download_episodes-%j.out
#SBATCH --error=./jobs/logs/download_episodes-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your-email@example.com>

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
cd "${REPO_ROOT}"

args=()
if [[ -n "${START:-}" ]]; then
  args+=(--start "${START}")
fi
if [[ -n "${END:-}" ]]; then
  args+=(--end "${END}")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  args+=(--out_dir "${OUT_DIR}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--force)
fi

echo "Running from: ${REPO_ROOT}"

echo "Running command: uv run python -u fetch_daily_episodes.py ${args[*]}"
uv run python -u fetch_daily_episodes.py "${args[@]}"
echo "Done."
