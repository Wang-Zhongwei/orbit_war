#!/usr/bin/env bash
#SBATCH --job-name=h2h_match
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/h2h_match-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/h2h_match-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Head-to-head 1v1 tournament between two REGISTERED agents (src.evaluation.AGENTS).
# Round-robin over the pair => ROUNDS matches with randomized slot order per round.
#
#   sbatch --export=ALL,A=sub_v86,B=sub_v861,ROUNDS=32 jobs/templates/h2h_match.slurm

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"

A="${A:-sub_v86}"
B="${B:-sub_v861}"
ROUNDS="${ROUNDS:-32}"
RESULTS_DIR="${RESULTS_DIR:-results/h2h}"
mkdir -p "${RESULTS_DIR}"
OUT_CSV="${RESULTS_DIR}/h2h_${A}_vs_${B}.csv"

echo "Match: ${A} vs ${B}  rounds=${ROUNDS}  workers=${SLURM_NTASKS:-1}"
echo "========================================================================"

uv run python arena.py tournament \
  --agents "${A}" "${B}" \
  --rounds "${ROUNDS}" \
  --base-seed 1000 \
  --workers "${SLURM_NTASKS:-1}" \
  --out "${OUT_CSV}"

# Per-agent win rate from the per-slot CSV (outcome: 1=win, 0.5=draw, 0=loss).
uv run python - "${OUT_CSV}" "${A}" "${B}" <<'PY'
import csv, sys
path, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
agg = {a: [], b: []}
with open(path) as f:
    for row in csv.DictReader(f):
        ag = row.get("agent")
        if ag in agg and row.get("outcome"):
            try:
                agg[ag].append(float(row["outcome"]))
            except ValueError:
                pass
print("========================================================================")
for ag, vals in agg.items():
    if vals:
        print(f"[winrate] {ag}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    else:
        print(f"[winrate] {ag}: no rows parsed")
PY

echo "CSV: ${OUT_CSV}"
echo "Done."
