"""4-player FFA head-to-head between two JAX checkpoints.

Each round randomly picks one of two line-ups, then shuffles seats:

    A-singleton :  1 copy of A  vs  3 copies of B
    B-singleton :  1 copy of B  vs  3 copies of A

The line-up is A-singleton with probability --prob-a-singleton (default 0.5);
within the round the singleton's seat (0-3) is randomized so neither agent is
pinned to a fixed board position. A round is won by the agent controlling the
rank-1 seat; if the rank-1 seats are split across both agents (a tie) the round
is a draw (0.5 each).

Reported per agent:
  * overall win rate   = (round wins + 0.5*draws) / rounds        [both sum to ~1]
  * 1-vs-3 win rate    = wins when that agent was the lone singleton vs 3 of the
                         other  (the cleanest "who is stronger" signal)

Both checkpoints must share the same src_jax architecture (they load into one
process); pass --src-a/--src-b only if a checkpoint was trained from a worktree
whose arch differs from this repo. Mixing two different arches in one process is
unsupported (src_jax module collision).

Usage:
  python -m tools.ffa_h2h --ckpt-a artifacts/ppo/ppo_4player/ckpt_000500.pkl \
                          --ckpt-b artifacts/ppo/ppo_mixed/ckpt_000500.pkl \
                          --rounds 256 --workers 16 --out results/ffa_h2h.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-process state (module-level so multiprocessing workers can hold it)
# ---------------------------------------------------------------------------
_NAME_A = "A"
_NAME_B = "B"


def _register_both(ckpt_a, ckpt_b, name_a, name_b, src_a, src_b, deterministic):
    """Register both checkpoints in arena's AGENTS table (this process)."""
    from arena import _register_jax_agent
    _register_jax_agent(ckpt_a, deterministic=deterministic, name=name_a, src_root=src_a)
    _register_jax_agent(ckpt_b, deterministic=deterministic, name=name_b, src_root=src_b)


def _init_worker(ckpt_a, ckpt_b, name_a, name_b, src_a, src_b, deterministic):
    global _NAME_A, _NAME_B
    _NAME_A, _NAME_B = name_a, name_b
    _children_on_cpu()
    _register_both(ckpt_a, ckpt_b, name_a, name_b, src_a, src_b, deterministic)


def _children_on_cpu():
    # Eval matches are CPU inference; keep JAX off any GPU so many workers coexist.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _lineup(round_idx: int, base_seed: int, prob_a_singleton: float):
    """Deterministic per-round line-up + seat assignment from (base_seed, idx)."""
    rng = random.Random(base_seed * 1_000_003 + round_idx)
    a_is_singleton = rng.random() < prob_a_singleton
    singleton, trio = (_NAME_A, _NAME_B) if a_is_singleton else (_NAME_B, _NAME_A)
    seat = rng.randrange(4)
    agents = [trio, trio, trio, trio]
    agents[seat] = singleton
    return agents, a_is_singleton, seat


def _classify(agents, ranks):
    """Return (a_score, b_score) in {1.0/0.0} or (0.5,0.5) for a split tie."""
    a_seats = {i for i, n in enumerate(agents) if n == _NAME_A}
    b_seats = {i for i, n in enumerate(agents) if n == _NAME_B}
    best = min(ranks)
    winners = {i for i, rk in enumerate(ranks) if rk == best}
    a_won = bool(winners & a_seats)
    b_won = bool(winners & b_seats)
    if a_won and not b_won:
        return 1.0, 0.0
    if b_won and not a_won:
        return 0.0, 1.0
    return 0.5, 0.5  # rank-1 seats split across both agents


def _run_round(task):
    round_idx, base_seed, prob_a_singleton, episode_steps = task
    from src.evaluation import run_match_raw
    agents, a_is_singleton, seat = _lineup(round_idx, base_seed, prob_a_singleton)
    res = run_match_raw(list(agents), seed=base_seed + round_idx, episode_steps=episode_steps)
    a_score, b_score = _classify(agents, res["ranks"])
    return {
        "round": round_idx,
        "singleton": _NAME_A if a_is_singleton else _NAME_B,
        "singleton_seat": seat,
        "ranks": res["ranks"],
        "num_steps": res["num_steps"],
        "a_score": a_score,
        "b_score": b_score,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    p.add_argument("--src-a", default=None, help="src_jax root for ckpt A (default: this repo).")
    p.add_argument("--src-b", default=None, help="src_jax root for ckpt B (default: this repo).")
    p.add_argument("--rounds", type=int, default=256)
    p.add_argument("--prob-a-singleton", type=float, default=0.5,
                   help="P(round is A-vs-3B); otherwise B-vs-3A. Default 0.5.")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--stochastic", action="store_true", help="Sample actions (default: argmax).")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/ffa_h2h.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    name_a, name_b = args.name_a, args.name_b
    if name_a == name_b:
        print("--name-a and --name-b must differ.", file=sys.stderr)
        return 2
    deterministic = not args.stochastic
    init_args = (args.ckpt_a, args.ckpt_b, name_a, name_b, args.src_a, args.src_b, deterministic)

    print(f"FFA h2h: {name_a}={args.ckpt_a}  vs  {name_b}={args.ckpt_b}", flush=True)
    print(f"rounds={args.rounds}  P({name_a}-singleton)={args.prob_a_singleton}  "
          f"workers={args.workers}  deterministic={deterministic}", flush=True)

    tasks = [(r, args.base_seed, args.prob_a_singleton, args.episode_steps) for r in range(args.rounds)]

    results = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_round, tasks), 1):
                results.append(res)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} rounds done", flush=True)
    else:
        global _NAME_A, _NAME_B
        _NAME_A, _NAME_B = name_a, name_b
        _register_both(*init_args)
        for i, t in enumerate(tasks, 1):
            results.append(_run_round(t))
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} rounds done", flush=True)

    # ── Aggregate ───────────────────────────────────────────────────────────
    n = len(results)
    a_total = sum(r["a_score"] for r in results)
    b_total = sum(r["b_score"] for r in results)
    draws = sum(1 for r in results if r["a_score"] == 0.5)

    a_single = [r for r in results if r["singleton"] == name_a]
    b_single = [r for r in results if r["singleton"] == name_b]
    a_single_wr = (sum(r["a_score"] for r in a_single) / len(a_single)) if a_single else float("nan")
    b_single_wr = (sum(r["b_score"] for r in b_single) / len(b_single)) if b_single else float("nan")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "singleton", "singleton_seat", "ranks", "num_steps", "a_score", "b_score"])
        for r in sorted(results, key=lambda x: x["round"]):
            w.writerow([r["round"], r["singleton"], r["singleton_seat"],
                        "|".join(map(str, r["ranks"])), r["num_steps"], r["a_score"], r["b_score"]])

    print("\n================= FFA HEAD-TO-HEAD =================", flush=True)
    print(f"rounds={n}  draws={draws}", flush=True)
    print(f"  {name_a}: overall win rate = {a_total / n:.3f}   "
          f"(as singleton vs 3x{name_b}: {a_single_wr:.3f} over {len(a_single)} rounds)", flush=True)
    print(f"  {name_b}: overall win rate = {b_total / n:.3f}   "
          f"(as singleton vs 3x{name_a}: {b_single_wr:.3f} over {len(b_single)} rounds)", flush=True)
    print(f"CSV: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
