#!/usr/bin/env bash
#SBATCH --job-name=bc_sweep_eval
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/bc_sweep_eval-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/bc_sweep_eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# Single-axis BC hyperparameter sweep — EVAL stage.
# Plays each every-N-shard checkpoint of a finished training run against sub_v52
# (best heuristic agent) and prints the jax win rate, producing a progress curve.
# Mirrors jobs/match_with_sub_v52.slurm (--agents jax sub_v52, 32 rounds, 16 workers).
#
# Required env var:
#   RUN_NAME       run identifier -> reads checkpoints from <OUT_DIR>/<RUN_NAME>/
# Optional env vars:
#   OUT_DIR        checkpoint root         (default artifacts/bc)
#   ROUNDS         seed rounds per ckpt    (default 32)
#   SHARD_STRIDE   eval every N shards     (default 100)
#   MAX_SHARD      highest shard index to look for (default 2000)
#   RESULTS_DIR    where match CSVs go     (default results/sweep/<RUN_NAME>)
#   SRC_ROOT       dir whose src_jax/ matches the ckpt arch (default: REPO_ROOT).
#                  Set to a .claude/worktrees/<wt> path if the ckpt was trained
#                  from a worktree whose arch differs from the working repo.

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"

: "${RUN_NAME:?Set RUN_NAME (must match the training run)}"
OUT_DIR="${OUT_DIR:-artifacts/bc}"
SRC_ROOT="${SRC_ROOT:-${REPO_ROOT}}"
ROUNDS="${ROUNDS:-64}"
SHARD_STRIDE="${SHARD_STRIDE:-100}"
MAX_SHARD="${MAX_SHARD:-2000}"
RESULTS_DIR="${RESULTS_DIR:-results/sweep/${RUN_NAME}}"
CKPT_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

# Collect checkpoints at shard multiples of SHARD_STRIDE, plus the final ckpt_last.
ckpts=()
for (( n=SHARD_STRIDE; n<=MAX_SHARD; n+=SHARD_STRIDE )); do
  printf -v tag "%04d" "${n}"
  for f in "${CKPT_DIR}"/ckpt_*_shard${tag}.pkl; do
    [[ -e "${f}" ]] && ckpts+=("${f}")
  done
done
[[ -e "${CKPT_DIR}/ckpt_last.pkl" ]] && ckpts+=("${CKPT_DIR}/ckpt_last.pkl")

if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No checkpoints found under ${CKPT_DIR} (looked for ckpt_*_shard{stride..}.pkl + ckpt_last.pkl)"
  exit 1
fi
echo "Evaluating ${#ckpts[@]} checkpoints vs sub_v52 (${ROUNDS} rounds, ${SLURM_NTASKS} workers)"
echo "========================================================================"

for ckpt in "${ckpts[@]}"; do
  base="$(basename "${ckpt}" .pkl)"
  out_csv="${RESULTS_DIR}/match_${base}.csv"
  echo ""
  echo ">>> ${base}"
  uv run python arena.py tournament \
    --agents jax sub_v52 \
    --jax-checkpoint "${ckpt}" \
    --jax-src-root "${SRC_ROOT}" \
    --rounds "${ROUNDS}" \
    --base-seed 1000 \
    --workers "${SLURM_NTASKS:-1}" \
    --out "${out_csv}" \
    --quiet
  # Summarize jax win rate from the per-slot CSV (outcome: 1=win, 0.5=draw, 0=loss).
  uv run python - "${out_csv}" "${base}" <<'PY'
import csv, sys
path, tag = sys.argv[1], sys.argv[2]
vals = []
with open(path) as f:
    for row in csv.DictReader(f):
        if row.get("agent") == "jax" and row.get("outcome"):
            try:
                vals.append(float(row["outcome"]))
            except ValueError:
                pass
if vals:
    wr = sum(vals) / len(vals)
    print(f"[winrate] {tag}: jax vs sub_v52 = {wr:.3f}  (n={len(vals)})")
else:
    print(f"[winrate] {tag}: no jax rows parsed from {path}")
PY
done

echo ""
echo "========================================================================"
echo "Sweep-eval win-rate curve for ${RUN_NAME}:"
grep -h "^\[winrate\]" "/home/018270239/orbit_war/jobs/logs/bc_sweep_eval-${SLURM_JOB_ID}.out" || true
echo "CSVs in ${RESULTS_DIR}/"

# --- push the win-rate curve into the training run's W&B run (ONLINE) ---
# sync_eval_to_wandb.py scrapes the [winrate] lines above and resumes the run id
# bc_train wrote to <OUT_DIR>/<RUN_NAME>/wandb_run_id.txt, logging
# eval/winrate_vs_sub_v52 vs eval/shard.
#
# IMPORTANT: this MUST run ONLINE, not offline+`wandb sync`. Appending history to
# an already-finished run via offline sync silently drops the time-series (only the
# summary scalar survives) -> the curve is invisible. Online resume appends it. The
# compute node has no network, so we run the logging step on the login node, which
# does. No sidecar (old runs) -> the script creates a fresh "<RUN_NAME>_eval" run.
echo ""
echo "Logging eval curve to W&B (online, via login node) ..."
RUN_ID=""
SIDECAR="${OUT_DIR}/${RUN_NAME}/wandb_run_id.txt"
[[ -f "${SIDECAR}" ]] && RUN_ID="$(cat "${SIDECAR}")"
ssh coe-hpc1 "cd ${REPO_ROOT} && WANDB_MODE=online OUT_DIR='${OUT_DIR}' \
  .venv/bin/python experiments/sync_eval_to_wandb.py '${SLURM_JOB_ID}' ${RUN_ID}" \
  || echo "WARN: eval->W&B logging failed; run experiments/sync_eval_to_wandb.py manually"

echo "Done."
