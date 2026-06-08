#!/usr/bin/env bash
#
# Launch the single-axis BC hyperparameter sweep from experiments/experiments.md.
#
# Design: a baseline run at the configs/bc_entity_policy_12.yaml defaults, then one run
# per non-baseline value of each axis (hidden_size, skill_clip_ratio, recency_half_life,
# loser_weight, candidate_count). Each run = a GPU train job + a dependent cpuqs eval job
# that scores every-100-shard checkpoints against sub_v52.
#
# Run this ON coe-hpc3 (per global policy: submit slurm jobs on coe-hpc3 only):
#   ssh coe-hpc3
#   cd /home/018270239/orbit_war
#   bash experiments/launch_hparam_sweep.sh                 # submit all (serial)
#   DRY_RUN=1 bash experiments/launch_hparam_sweep.sh       # print what would submit
#   bash experiments/launch_hparam_sweep.sh sweep_k_8 ...   # submit only named run(s)
#
# Concurrency: gpuqs OVERSUBSCRIBES GPUs, and two of these jobs on one H100 OOM. So the
# default LANES=1 chains every train job afterany the previous one — all jobs are
# submitted at once but exactly one trains at a time. (LANES>1 allows N concurrent and
# risks OOM; the train slurm uses XLA_PYTHON_CLIENT_PREALLOCATE=false to soften that.)

set -euo pipefail
cd /home/018270239/orbit_war

TRAIN_SLURM="jobs/templates/bc_sweep_train.slurm"
EVAL_SLURM="jobs/templates/bc_sweep_eval.slurm"
DRY_RUN="${DRY_RUN:-0}"
LANES="${LANES:-4}"        # concurrent train jobs. Default 1 = STRICTLY SERIAL: all jobs
                           # are submitted at once but chained so only one trains at a time,
                           # so they never cram an (oversubscribed) gpuqs GPU and OOM.

# Each entry: RUN_NAME|ENV_OVERRIDES   (empty overrides == baseline defaults)
RUNS=(
  "sweep_baseline|"                                  # config defaults: h256 K12 clip1.0 rec14 loser0.1 lr1e-4 bs4096
  "sweep_hidden_128|HIDDEN_SIZE=128"                 # hidden-size axis (256 = baseline)
  "sweep_hidden_512|HIDDEN_SIZE=512"                 # hidden-size axis (256 = baseline)
  "sweep_clip_1_5|SKILL_CLIP_RATIO=1.5"              # skill_clip axis (1.0 = baseline)
  "sweep_clip_3_0|SKILL_CLIP_RATIO=3.0"
  "sweep_recency_21|RECENCY_HALF_LIFE=21.0"          # recency axis (14 = baseline, in config)
  "sweep_recency_28|RECENCY_HALF_LIFE=28.0"
  "sweep_loser_0_0|LOSER_WEIGHT=0.0"                 # loser_weight axis (0.1 = baseline, in config)
  "sweep_k_8|CANDIDATE_COUNT=8"                      # candidate_count axis (12 = baseline)
  "sweep_k_16|CANDIDATE_COUNT=16"
)

# Optional allowlist: pass run names as CLI args to submit only those (e.g. to resubmit
# the runs that failed). No args = submit all. Lane throttling applies to whatever subset
# actually gets submitted.
if [[ $# -gt 0 ]]; then
  declare -A _want; for _a in "$@"; do _want["${_a}"]=1; done
  _filtered=()
  for _e in "${RUNS[@]}"; do
    [[ -n "${_want[${_e%%|*}]:-}" ]] && _filtered+=("${_e}")
  done
  RUNS=("${_filtered[@]}")
  [[ ${#RUNS[@]} -eq 0 ]] && { echo "No matching run names in: $*" >&2; exit 1; }
fi

submit() {  # $1=slurm file, $2=comma-exports, [$3=full --dependency expr] -> echoes jobid
  local file="$1" exports="$2" dep="${3:-}"
  local cmd=(sbatch --parsable)
  [[ -n "${dep}" ]] && cmd+=(--dependency="${dep}")
  cmd+=(--export="ALL,${exports}" "${file}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY: ${cmd[*]}" >&2
    echo "$(( 100000 + RANDOM ))"   # fake distinct jobid so the chain prints sanely
  else
    "${cmd[@]}"
  fi
}

echo "Submitting ${#RUNS[@]} sweep runs (train -> eval; max ${LANES} concurrent train jobs)"
echo "DRY_RUN=${DRY_RUN}"
echo "========================================================================"
declare -a lane_ids   # ring buffer of the last LANES train job ids
i=0
for entry in "${RUNS[@]}"; do
  run_name="${entry%%|*}"
  overrides="${entry#*|}"
  train_exports="RUN_NAME=${run_name}"
  [[ -n "${overrides}" ]] && train_exports="${train_exports},${overrides}"

  # Throttle: this train job waits for the one LANES positions back (afterany = start
  # regardless of that job's exit status, so a failure never stalls the lane).
  train_dep=""
  if (( i >= LANES )); then
    train_dep="afterany:${lane_ids[$(( i % LANES ))]}"
  fi
  train_id="$(submit "${TRAIN_SLURM}" "${train_exports}" "${train_dep}")"
  lane_ids[$(( i % LANES ))]="${train_id}"

  eval_id="$(submit "${EVAL_SLURM}" "RUN_NAME=${run_name}" "afterok:${train_id}")"
  printf "  %-20s train=%-10s eval=%-10s  [%s]\n" \
    "${run_name}" "${train_id}" "${eval_id}" "${overrides:-baseline}"
  i=$(( i + 1 ))
done
echo "========================================================================"
echo "Submitted. Watch with: squeue --me    |    logs: jobs/logs/bc_sweep_*"
