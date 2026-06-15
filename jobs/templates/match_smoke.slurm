#!/usr/bin/env bash
#SBATCH --job-name=match_smoke
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/match_smoke-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/match_smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Single-match smoke test: play the newly-registered sub_v82 vs sub_v8 and surface
# any error (import/compile/runtime). Both are standalone submission agents that
# embed their own src_jax, loaded against the pinned v7arch tree in src.evaluation.
REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"

A="${A:-sub_v82}"
B="${B:-sub_v8}"
SEED="${SEED:-42}"
# FFA=1 → 1-vs-3 free-for-all: A vs 3 copies of B (a 4-player game). Default = 1v1.
FFA="${FFA:-}"
OUT_CSV="${OUT_CSV:-results/match_${A}_vs_${B}.csv}"
mkdir -p "$(dirname "${OUT_CSV}")"

match_args=(--agents "${A}" "${B}" --seed "${SEED}" --out "${OUT_CSV}")
if [[ -n "${FFA}" && "${FFA}" != "0" ]]; then
  match_args+=(--ffa)
  echo "Match (1-vs-3 FFA): ${A} vs 3x ${B} (seed ${SEED})"
else
  echo "Match: ${A} vs ${B} (seed ${SEED})"
fi
echo "========================================================================"
uv run python arena.py match "${match_args[@]}"

echo "========================================================================"
echo "Match completed without error. CSV: ${OUT_CSV}"
