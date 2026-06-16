#!/usr/bin/env bash
#SBATCH --job-name=ppo_eval
#SBATCH --output=/home/018270239/orbit_war/jobs/logs/ppo_eval-%j.out
#SBATCH --error=/home/018270239/orbit_war/jobs/logs/ppo_eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wang.15032@osu.edu

set -euo pipefail

# PPO checkpoint EVAL — plays the checkpoints of a finished PPO run against a LIST
# of opponents (default sub_v7 sub_v8) and prints a jax win-rate curve per opponent.
# Mirrors jobs/templates/bc_sweep_eval.slurm, adapted for PPO:
#   * checkpoints are named ckpt_<update>.pkl (+ ckpt_last.pkl), not shard-tagged;
#   * collected RECURSIVELY so split run-id subdirs (e.g. ppo_low_kl/ol5kfsm8/...)
#     are stitched back into one update-sorted curve;
#   * opponents are a list and each is evaluated in its own tournament.
#
# Required env var:
#   RUN_NAME     folder under <OUT_DIR>  ->  checkpoints read from <OUT_DIR>/<RUN_NAME>/
# Optional env vars:
#   OUT_DIR      checkpoint root                 (default artifacts/ppo)
#   OPPONENTS    space-separated agent list      (default "sub_v7 sub_v8")
#   ROUNDS       seed rounds per (ckpt,opp)      (default 64)
#   CKPT_STRIDE  eval every N updates            (default 100)
#   RESULTS_DIR  where match CSVs go             (default results/ppo_eval/<RUN_NAME>)

REPO_ROOT="/home/018270239/orbit_war"
cd "${REPO_ROOT}"
echo "Running from: ${REPO_ROOT}"

: "${RUN_NAME:?Set RUN_NAME (the PPO run folder under OUT_DIR)}"
OUT_DIR="${OUT_DIR:-artifacts/ppo}"
OPPONENTS="${OPPONENTS:-sub_v7 sub_v8}"
ROUNDS="${ROUNDS:-64}"
CKPT_STRIDE="${CKPT_STRIDE:-100}"
RESULTS_DIR="${RESULTS_DIR:-results/ppo_eval/${RUN_NAME}}"
CKPT_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

# Collect ckpt_<update>.pkl recursively, dedupe by update, keep those on the stride,
# then append ckpt_last.pkl. Sorted ascending by update so the curve reads left->right.
declare -A by_update=()
while IFS= read -r f; do
  b="$(basename "${f}" .pkl)"          # ckpt_000300
  n="${b#ckpt_}"                        # 000300
  [[ "${n}" =~ ^[0-9]+$ ]] || continue
  u=$((10#${n}))                        # strip leading zeros
  if (( u % CKPT_STRIDE == 0 )); then
    by_update[${u}]="${f}"             # unique per update across subdirs
  fi
done < <(find "${CKPT_DIR}" -type f -name 'ckpt_*.pkl' ! -name 'ckpt_last.pkl' | sort)

ckpts=()
for u in $(printf '%s\n' "${!by_update[@]}" | sort -n); do
  ckpts+=("${by_update[$u]}")
done
# Final checkpoint (definitive end state). find -> one canonical ckpt_last if present.
last="$(find "${CKPT_DIR}" -type f -name 'ckpt_last.pkl' | sort | tail -1)"
[[ -n "${last}" ]] && ckpts+=("${last}")

if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No checkpoints found under ${CKPT_DIR} (ckpt_<update>.pkl on stride ${CKPT_STRIDE}, or ckpt_last.pkl)"
  exit 1
fi

echo "RUN_NAME=${RUN_NAME}  OUT_DIR=${OUT_DIR}"
echo "Opponents: ${OPPONENTS}"
echo "Evaluating ${#ckpts[@]} checkpoints x $(wc -w <<< "${OPPONENTS}") opponents (${ROUNDS} rounds, ${SLURM_NTASKS:-1} workers)"
echo "========================================================================"

for opp in ${OPPONENTS}; do
  echo ""
  echo "### opponent: ${opp}"
  for ckpt in "${ckpts[@]}"; do
    base="$(basename "${ckpt}" .pkl)"
    out_csv="${RESULTS_DIR}/match_${opp}_${base}.csv"
    echo ""
    echo ">>> ${base} vs ${opp}"
    uv run python arena.py tournament \
      --agents jax "${opp}" \
      --jax-checkpoint "${ckpt}" \
      --rounds "${ROUNDS}" \
      --base-seed 1000 \
      --workers "${SLURM_NTASKS:-1}" \
      --out "${out_csv}" \
      --quiet
    # Summarize jax win rate from the per-slot CSV (outcome: 1=win, 0.5=draw, 0=loss).
    uv run python - "${out_csv}" "${base}" "${opp}" <<'PY'
import csv, sys
path, tag, opp = sys.argv[1], sys.argv[2], sys.argv[3]
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
    print(f"[winrate] {opp}: {tag}: jax = {wr:.3f}  (n={len(vals)})")
else:
    print(f"[winrate] {opp}: {tag}: no jax rows parsed from {path}")
PY
  done
done

echo ""
echo "========================================================================"
echo "PPO eval win-rate curves for ${RUN_NAME}:"
for opp in ${OPPONENTS}; do
  echo "--- vs ${opp} ---"
  grep -h "^\[winrate\] ${opp}:" "/home/018270239/orbit_war/jobs/logs/ppo_eval-${SLURM_JOB_ID}.out" || true
done
echo "CSVs in ${RESULTS_DIR}/"

# --- push the eval curves into W&B (ONLINE, via login node) ---
# sync_ppo_eval_to_wandb.py scrapes the [winrate] lines above and logs a fresh
# standalone run "<RUN_NAME>_eval" with eval/winrate_vs_<opp> vs eval/update.
# A FRESH run (not a resume) is used on purpose: the multi-run folders were
# written by several W&B runs (some in the v7arch worktree dir), so there is no
# single training run to attach to. The compute node has no network, so the
# logging step runs on the login node (coe-hpc1), which does.
echo ""
echo "Logging eval curves to W&B (online, via login node) ..."
ssh coe-hpc1 "cd ${REPO_ROOT} && WANDB_MODE=online \
  .venv/bin/python experiments/sync_ppo_eval_to_wandb.py '${SLURM_JOB_ID}'" \
  || echo "WARN: eval->W&B logging failed; run experiments/sync_ppo_eval_to_wandb.py manually"

echo "Done."
