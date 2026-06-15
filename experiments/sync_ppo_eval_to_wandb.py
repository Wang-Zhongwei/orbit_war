#!/usr/bin/env python3
"""Push a PPO eval win-rate curve from a SLURM eval .out file into W&B.

The PPO eval slurm (jobs/templates/ppo_eval.slurm) prints, for each opponent and
checkpoint, lines like
    [winrate] sub_v7: ckpt_000300: jax = 0.625  (n=64)
    [winrate] sub_v7: ckpt_last:   jax = 0.700  (n=64)
and a footer "PPO eval win-rate curves for <run_name>:".

This script parses those, then logs ONE fresh standalone run named
"<run_name>_eval" (project orbit-war) with, per opponent, the curve
    eval/win_rate_<opp>     logged at  step=update
plus summaries  eval/final_win_rate_<opp>.

A fresh run (rather than resuming the training run) is deliberate: the multi-run
folders (e.g. ppo_low_kl) were written by several W&B runs, some living in the
v7arch worktree's wandb dir — there is no single training run to attach to, and a
training run already owns its own eval/win_rate_<opp> series (so we can't append
to it without a metric collision). A standalone *_eval run sidesteps that.

CRITICAL: the metric key (eval/win_rate_<opp>) and the x-axis (the default _step,
set to the checkpoint's update number) MUST match what src_jax/train.py logs
during training — only then does W&B place these curves on the SAME panel as the
training runs' eval lines, instead of a separate panel.

The x (=_step) for ckpt_last is the highest numbered update + the curve stride (so
it sits just past the last numbered point instead of colliding with it).

Usage (run ONLINE on a host with network — coe-hpc1):
    WANDB_MODE=online uv run python experiments/sync_ppo_eval_to_wandb.py <out_file_or_jobid>
    [--dry]   parse only, log nothing.

Offline + `wandb sync` works here because this is always a FRESH run (no resume),
but online is simplest and matches how the slurm invokes it.
"""
import glob
import os
import re
import sys

REPO = "/home/018270239/orbit_war"
LOGDIR = os.path.join(REPO, "jobs/logs")

# [winrate] <opp>: <ckpt>: jax = <wr>  (n=<n>)
WIN_RE = re.compile(r"\[winrate\]\s+(\S+):\s+(\S+):\s+jax = ([\d.]+)\s+\(n=(\d+)\)")
NAME_RE = re.compile(r"win-rate curves for (\S+?):")
UPD_RE = re.compile(r"ckpt_(\d+)")


def resolve_out(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    fname = arg if arg.endswith(".out") else f"ppo_eval-{arg}.out"
    cand = os.path.join(LOGDIR, fname)
    if os.path.isfile(cand):
        return cand
    sys.exit(f"[err] cannot find eval out file for '{arg}' (tried {cand})")


def parse_out(path: str):
    run_name = None
    # per opponent: list of (ckpt, update_or_None, winrate, n)
    by_opp = {}
    with open(path) as f:
        for line in f:
            m = NAME_RE.search(line)
            if m:
                run_name = m.group(1)
            m = WIN_RE.search(line)
            if m:
                opp, ckpt, wr, n = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
                um = UPD_RE.search(ckpt)
                upd = int(um.group(1)) if um else None  # None == ckpt_last
                by_opp.setdefault(opp, []).append((ckpt, upd, wr, n))
    return run_name, by_opp


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    if not argv:
        sys.exit(__doc__)

    out_path = resolve_out(argv[0])
    run_name, by_opp = parse_out(out_path)
    if run_name is None:
        sys.exit(f"[err] no 'win-rate curves for <run>' footer in {out_path}")
    if not by_opp:
        sys.exit(f"[err] no [winrate] lines parsed from {out_path}")

    print(f"[parse] {out_path}")
    print(f"[parse] run_name = {run_name}  opponents = {sorted(by_opp)}")

    # Decide an x for ckpt_last: max numbered update across all opponents + stride.
    all_upds = sorted({u for pts in by_opp.values() for _, u, _, _ in pts if u is not None})
    stride = (all_upds[1] - all_upds[0]) if len(all_upds) >= 2 else 25
    last_x = (all_upds[-1] + stride) if all_upds else 0

    # Build, per opponent, an update -> (wr, n) map and a final winrate.
    curves = {}   # opp -> sorted list of (x, wr, n, ckpt)
    finals = {}   # opp -> final winrate (ckpt_last if present else highest update)
    for opp, pts in by_opp.items():
        rows = []
        last_wr = None
        for ckpt, upd, wr, n in pts:
            x = upd if upd is not None else last_x
            rows.append((x, wr, n, ckpt))
            if upd is None:
                last_wr = wr
        rows.sort(key=lambda r: r[0])
        curves[opp] = rows
        finals[opp] = last_wr if last_wr is not None else (rows[-1][1] if rows else 0.0)

    # Combined x grid (so each history row can carry all opponents at that x).
    xgrid = sorted({x for rows in curves.values() for x, _, _, _ in rows})

    print("[curve] update  " + "  ".join(f"vs_{o}" for o in sorted(curves)))
    for x in xgrid:
        cells = []
        for o in sorted(curves):
            v = next((wr for xx, wr, _, _ in curves[o] if xx == x), None)
            cells.append(f"{v:.3f}" if v is not None else "  -  ")
        tag = "last" if x == last_x and last_x not in all_upds else str(x)
        print(f"        {tag:<6}  " + "  ".join(cells))
    print("[final] " + "  ".join(f"{o}={finals[o]:.3f}" for o in sorted(finals)))

    if dry:
        print("[dry] nothing logged.")
        return

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", REPO)
    import wandb

    run = wandb.init(project="orbit-war", name=f"{run_name}_eval",
                     config={"eval_run_name": run_name, "opponents": sorted(curves),
                             "source_out": os.path.basename(out_path)})
    # Log eval/win_rate_<opp> at step=update so the curves share both the metric key
    # and the "Step" x-axis used by the training runs -> same panel, not a new one.
    for x in xgrid:
        row = {}
        for opp in sorted(curves):
            v = next((wr for xx, wr, _, _ in curves[opp] if xx == x), None)
            if v is not None:
                row[f"eval/win_rate_{opp}"] = v
        if row:
            wandb.log(row, step=x)
    for opp in sorted(finals):
        run.summary[f"eval/final_win_rate_{opp}"] = finals[opp]
    wandb.finish()

    wb_base = os.path.join(os.environ["WANDB_DIR"], "wandb")
    synced = sorted(glob.glob(os.path.join(wb_base, f"offline-run-*-{run.id}")))
    print(f"\n[done] logged run {run.id} ('{run_name}_eval') "
          + "  ".join(f"{o}={finals[o]:.3f}" for o in sorted(finals)))
    for d in synced:
        print(f"[next] wandb sync {d}")


if __name__ == "__main__":
    main(sys.argv[1:])
