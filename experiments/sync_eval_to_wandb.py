#!/usr/bin/env python3
"""TEMPORARY: push a sweep-eval win-rate curve from a SLURM eval .out file into
the matching training run on W&B.

The eval slurm (jobs/templates/bc_sweep_eval.slurm) prints lines like
    [winrate] ckpt_000001_shard0100: jax vs sub_v52 = 0.000  (n=32)
and a footer "Sweep-eval win-rate curve for <run_name>:". This script parses
those, resumes the training run's offline W&B run, and logs the curve as
    eval/winrate_vs_sub_v52  vs  eval/shard   (custom x-axis, no step clash)
plus a summary  eval/final_winrate.

Usage (run on coe-hpc1 where W&B sync has network):
    uv run python experiments/sync_eval_to_wandb.py <out_file_or_jobid> [wandb_run_id]

  <out_file_or_jobid>  path to jobs/logs/bc_sweep_eval-<id>.out, or just the job id.
  [wandb_run_id]       optional; if omitted it is auto-resolved from RUN_NAME to
                       the most recent local offline run with that name.
  --dry                parse + resolve only, log nothing.

To MERGE the curve into an existing (finished) training run, run ONLINE on a
host with network (coe-hpc1):
    WANDB_MODE=online uv run python experiments/sync_eval_to_wandb.py <jobid> <run_id>
Offline + `wandb sync` does NOT work for resuming a finished run: appending
history to it via offline sync silently drops the time-series (only the summary
scalar survives), so the curve is invisible. Online resume appends it correctly.

Default mode is offline (creates wandb/offline-run-*-<id>); if you go that route
for a FRESH run only, finish the upload with the printed command:
    wandb sync wandb/offline-run-*-<id>
"""
import glob
import os
import re
import sys

REPO = "/home/018270239/orbit_war"
LOGDIR = os.path.join(REPO, "jobs/logs")
WANDB_DIR = os.path.join(REPO, "wandb")

WIN_RE = re.compile(r"\[winrate\]\s+(\S+):\s+jax vs sub_v52 = ([\d.]+)\s+\(n=(\d+)\)")
NAME_RE = re.compile(r"win-rate curve for (\S+?):")
SHARD_RE = re.compile(r"shard(\d+)")


def resolve_out(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    fname = arg if arg.endswith(".out") else f"bc_sweep_eval-{arg}.out"
    cand = os.path.join(LOGDIR, fname)
    if os.path.isfile(cand):
        return cand
    sys.exit(f"[err] cannot find eval out file for '{arg}' (tried {cand})")


def parse_out(path: str):
    run_name = None
    points = []  # (ckpt, shard_or_None, winrate, n)
    with open(path) as f:
        for line in f:
            m = NAME_RE.search(line)
            if m:
                run_name = m.group(1)
            m = WIN_RE.search(line)
            if m:
                ckpt, wr, n = m.group(1), float(m.group(2)), int(m.group(3))
                sm = SHARD_RE.search(ckpt)
                points.append((ckpt, int(sm.group(1)) if sm else None, wr, n))
    return run_name, points


def resolve_run_id(run_name: str):
    """Return (best_id, [(mtime, id, dir), ...]) for offline runs whose .wandb
    record contains run_name. Most recent first."""
    matches = []
    needle = run_name.encode()
    for d in glob.glob(os.path.join(WANDB_DIR, "offline-run-*")):
        rid = d.rsplit("-", 1)[-1]
        wf = os.path.join(d, f"run-{rid}.wandb")
        if not os.path.isfile(wf):
            continue
        try:
            with open(wf, "rb") as fh:
                if needle in fh.read():
                    matches.append((os.path.getmtime(d), rid, d))
        except OSError:
            continue
    matches.sort(reverse=True)
    return (matches[0][1] if matches else None), matches


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    if not argv:
        sys.exit(__doc__)

    out_path = resolve_out(argv[0])
    run_id = argv[1] if len(argv) > 1 else None

    run_name, points = parse_out(out_path)
    if run_name is None:
        sys.exit(f"[err] no 'win-rate curve for <run>' footer in {out_path}")
    if not points:
        sys.exit(f"[err] no [winrate] lines parsed from {out_path}")

    print(f"[parse] {out_path}")
    print(f"[parse] run_name = {run_name}  ({len(points)} checkpoints)")

    if run_id is None:
        # Prefer the deterministic sidecar written by bc_train at wandb.init.
        out_dir = os.environ.get("OUT_DIR", os.path.join(REPO, "artifacts/bc"))
        sidecar = os.path.join(out_dir, run_name, "wandb_run_id.txt")
        if os.path.isfile(sidecar):
            run_id = open(sidecar).read().strip()
            print(f"[resolve] run_id = {run_id}  (sidecar {sidecar})")
    # NOTE: we deliberately do NOT grep-guess an id from old offline folders.
    # A guessed id may belong to a run that was deleted on the server, and
    # resuming/syncing a deleted id fails with HTTP 409. With no sidecar and no
    # explicit id we create a fresh "<run_name>_eval" run instead (always safe).
    if run_id is None:
        cands = resolve_run_id(run_name)[1]
        if cands:
            print(f"[resolve] no sidecar; {len(cands)} old offline run(s) match "
                  f"'{run_name}' but may be deleted on the server — NOT reusing them.")
            print(f"          pass an explicit run_id as 2nd arg to attach to a live run.")
        print(f"[resolve] creating a fresh run '{run_name}_eval' (no id reuse).")
    else:
        src = "provided" if len(argv) > 1 else "sidecar"
        print(f"[resolve] run_id = {run_id}  ({src})")

    # ckpt_last (no shard number) -> place just past the last real shard.
    real_shards = [s for _, s, _, _ in points if s is not None]
    last_step = (max(real_shards) + 100) if real_shards else 0
    rows = []
    for ckpt, shard, wr, n in points:
        rows.append((shard if shard is not None else last_step, wr, n, ckpt))
    rows.sort(key=lambda r: r[0])

    print("[curve] shard  winrate  n   ckpt")
    for step, wr, n, ckpt in rows:
        print(f"        {step:<5}  {wr:.3f}    {n:<3} {ckpt}")

    if dry:
        print("[dry] nothing logged.")
        return

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", REPO)
    import wandb

    if run_id is not None:
        run = wandb.init(project="orbit-war", id=run_id, resume="allow")
    else:
        run = wandb.init(project="orbit-war", name=f"{run_name}_eval")
    wandb.define_metric("eval/shard")
    wandb.define_metric("eval/winrate_vs_sub_v52", step_metric="eval/shard")
    for step, wr, n, ckpt in rows:
        wandb.log({"eval/shard": step, "eval/winrate_vs_sub_v52": wr, "eval/n": n})
    # final = ckpt_last if present, else the highest-shard point
    final_wr = next((wr for _, s, wr, _ in points if s is None), rows[-1][1])
    run.summary["eval/final_winrate"] = final_wr
    wandb.finish()

    # Only the dir we just wrote — never blanket-sync (deleted runs -> HTTP 409).
    wb_base = os.path.join(os.environ["WANDB_DIR"], "wandb")
    synced = sorted(glob.glob(os.path.join(wb_base, f"offline-run-*-{run.id}")))
    print(f"\n[done] logged {len(rows)} points to run {run.id} (final={final_wr:.3f})")
    print("[next] upload with:")
    for d in synced:
        print(f"    wandb sync {d}")
        print(f"SYNC_DIR={d}")
    sdf = os.environ.get("SYNC_DIR_FILE")
    if sdf and synced:
        with open(sdf, "w") as f:
            f.write("\n".join(synced) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
