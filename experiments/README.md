# hyperparameter scan

# data
entire episodes from 2026-05-01 to today

## hidden size
- choices: [256, 512]
- purpose: wanna know how much entropy baseline will drop compared with the increased computing resource

## skill_clip_ratio
- choices: [1.0, 1.5, 3.0]
- intuition: 1.0 means high score player's (1600) move have the same weight as low score player (1200). My intuition is this shoudn't be too large

## recency_half_life
- choices: [14.0, 21.0, 28.0]
- intuition: agent gets stronger over time

## loser_weight
- choices: [0.0, 0.1]
- intuition: winning rate will likely monotonically drop with loser_weight since player will be initating the loser's behavior but it better not be completely 0 because it slightly adds diversity to the BC

# candidate_count K 
- choices: [8, 12, 16]
- inuition: smaller K might less prone to overfitting and perform better on single date episodes data which could explained what we saw so far. But larger K might be necessary to capture more coalition patterns from BC data and worth a try on the entire BC experts data. See how much the entropy floor would drop. 

Don't need to do a grid scan since we can't afford the time. Do a single scan with the rest of the hypterparameters default from configs/bc_entity_policy_12.yaml. The result should be illuminating enough. 


After each scan, evaluate the ppo agent against sub_v52 (best heuristic agent) every 100 shards with logging similar to jobs/match_with_sub_v52.slurm. Could write a single experiment slurms to batch this together add the evaluation command after the training whichever is better you decide.

---

# Implementation (prepared 2026-06-07)

## Design: single-axis, shared baseline (no grid)
The **baseline** is whatever is in `configs/bc_entity_policy_12.yaml` (currently hidden=256,
K=12, skill_clip=1.0, recency=14, loser=0.1, lr=1e-4, batch=4096). Each axis runs only its
*non-baseline* values — the baseline point is shared and not rerun. Because the config now
sets recency=14 and loser=0.1, those points coincide with the baseline, so **9 runs** total
(each with a dependent eval job):

| run | override | axis |
|-----|----------|------|
| `sweep_baseline`   | (none)                  | config defaults |
| `sweep_hidden_512` | `HIDDEN_SIZE=512`       | hidden_size (256=baseline) |
| `sweep_clip_1_5`   | `SKILL_CLIP_RATIO=1.5`  | skill_clip (1.0=baseline) |
| `sweep_clip_3_0`   | `SKILL_CLIP_RATIO=3.0`  | skill_clip |
| `sweep_recency_21` | `RECENCY_HALF_LIFE=21.0`| recency (14=baseline) |
| `sweep_recency_28` | `RECENCY_HALF_LIFE=28.0`| recency |
| `sweep_loser_0_0`  | `LOSER_WEIGHT=0.0`      | loser_weight (0.1=baseline) |
| `sweep_k_8`        | `CANDIDATE_COUNT=8`     | candidate_count (12=baseline) |
| `sweep_k_16`       | `CANDIDATE_COUNT=16`    | candidate_count |

## Files
- `src_jax/bc_train.py` — added `--candidate_count` CLI flag (overrides `env.candidate_count`;
  the only swept knob that lacked an override). K∈{8,12,16} all re-slice from the encoded
  **POOL_K=24** shards, so no re-encode is needed.
- `jobs/templates/bc_sweep_train.slurm` — GPU (gpuqs/h100). Trains on all date dirs
  `episodes/2026-0[56]-*` (`--use_shard`); vary one knob via env var (`RUN_NAME` required).
- `jobs/templates/bc_sweep_eval.slurm` — CPU (cpuqs, 16 workers). Scores every-100-shard
  checkpoints (`ckpt_*_shard{0100,0200,...}.pkl`) + `ckpt_last` vs sub_v52 with
  `arena.py tournament` (32 rounds), prints the jax win-rate curve. Mirrors
  `jobs/match_with_sub_v52.slurm`.
- `experiments/launch_hparam_sweep.sh` — submits the train→eval pairs.

Train and eval are split into two jobs because they need different partitions (GPU train
vs cpuqs 16-worker eval). Checkpoints already save every 5 shards, so the every-100-shard
curve needs no training change.

## GPU oversubscription (why serial)
`gpuqs` oversubscribes GPUs — left to run concurrently, multiple sweep jobs land on one H100
and OOM (`--exclusive` is rejected by the partition). The launcher defaults to **`LANES=1`**:
all jobs are submitted at once but chained `afterany` so exactly one trains at a time.

## Run
```bash
ssh coe-hpc3 && cd /home/018270239/orbit_war
DRY_RUN=1 bash experiments/launch_hparam_sweep.sh        # preview
bash experiments/launch_hparam_sweep.sh                  # submit all 9 (serial)
bash experiments/launch_hparam_sweep.sh sweep_k_8 ...    # or only named run(s)
squeue --me                                              # watch; logs in jobs/logs/bc_sweep_*
```



