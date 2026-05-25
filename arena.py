"""
arena.py — Match runner & tournament scheduler for Orbit Wars.

Responsibilities
----------------
1. Provide a human-readable CSV log of every match.
2. Run a single match and append one row per slot to ``results/matches.csv``.
3. Schedule a round-robin 1v1 tournament with slot rotation: every pair plays
   the same seed twice (once with each ordering) so starting-position bias
   cancels out.
4. Print a head-to-head win matrix at the end of a tournament.

Elo computation lives in ``evaluate_elo.py`` — this script is write-only over
the log so you can re-tune K, add agents, or change the rating model without
re-running matches.

Shared match execution and evaluation helpers live in ``evaluation.py``.

Usage
-----
    python arena.py list
    python arena.py match --agents sniper roi --seed 42
    python arena.py tournament --rounds 50 --base-seed 1000
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import multiprocessing
import os
import sys
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.evaluation import AGENTS, run_match_raw, format_head_to_head_matrix
from src.kaggle_adapter.ppo_agent import make_ppo_agent


CSV_HEADER: list[str] = [
    "match_id", "timestamp", "seed", "format",
    "num_steps", "slot", "agent",
    "reward", "status", "rank", "outcome",
]

DEFAULT_CSV = Path(__file__).parent / "results" / "matches.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_ppo_agent(
    cfg_path: str,
    checkpoint: str | None,
    deterministic: bool,
    name: str = "ppo",
) -> None:
    """Build a PPO agent from checkpoint and register it in AGENTS under ``name``."""
    from src.config import load_train_config
    cfg = load_train_config(cfg_path)
    AGENTS[name] = make_ppo_agent(cfg, checkpoint, deterministic=deterministic)


def _resolve_agents(names: Sequence[str]) -> None:
    for name in names:
        if name not in AGENTS:
            raise KeyError(f"Unknown agent {name!r}. Known: {sorted(AGENTS)}")


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _flush_rows(rows: list[list], csv_path: Path) -> None:
    """Append buffered rows to the CSV (writes the header first if needed)."""
    if not rows:
        return
    _ensure_csv(csv_path)
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def play_match(
    agent_names: Sequence[str],
    seed: int,
    *,
    episode_steps: int = 500,
    csv_path: Path = DEFAULT_CSV,
    match_id: str | None = None,
    replay_path: Path | None = None,
    _row_buffer: list[list] | None = None,
) -> dict:
    """Run one match and return a result dict.

    By default the per-slot CSV rows are written immediately.  Pass a list as
    ``_row_buffer`` to accumulate rows instead; the caller is then responsible
    for flushing via :func:`_flush_rows`.
    """
    if len(agent_names) not in (2, 4):
        raise ValueError(f"Orbit Wars supports 2 or 4 players, got {len(agent_names)}.")

    _resolve_agents(agent_names)

    raw = run_match_raw(list(agent_names), seed, episode_steps=episode_steps)

    if replay_path is not None:
        from kaggle_environments import make
        fns = [AGENTS[a] if isinstance(a, str) else a for a in agent_names]
        env = make("orbit_wars", configuration={"seed": int(seed), "episodeSteps": int(episode_steps)}, debug=False)
        env.run(fns)
        replay_path = Path(replay_path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(env.render(mode="html"), encoding="utf-8")

    n = len(agent_names)
    if match_id is None:
        match_id = uuid.uuid4().hex[:12]
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    fmt = "1v1" if n == 2 else f"ffa{n}"

    rows = [
        [match_id, timestamp, seed, fmt, raw["num_steps"],
         slot, name, r, st, rk, f"{oc:.3f}"]
        for slot, (name, r, st, rk, oc) in enumerate(zip(
            agent_names,
            raw["rewards"], raw["statuses"], raw["ranks"], raw["outcomes"],
        ))
    ]

    if _row_buffer is None:
        _flush_rows(rows, csv_path)
    else:
        _row_buffer.extend(rows)

    return {
        "match_id": match_id,
        "seed":     seed,
        "agents":   list(agent_names),
        **raw,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker  (must be a module-level function for pickling)
# ---------------------------------------------------------------------------

def _match_task(task: dict) -> dict:
    """Run one match in a worker process and return rows + metadata (no I/O).

    ``task`` keys: ``slot_order`` (tuple[str,...]), ``seed`` (int),
    ``episode_steps`` (int).
    """
    rows: list[list] = []
    res = play_match(
        list(task["slot_order"]),
        seed=task["seed"],
        episode_steps=task["episode_steps"],
        _row_buffer=rows,
    )
    return {
        "rows":        rows,
        "agents":      list(task["slot_order"]),
        "ranks":       res["ranks"],
        "num_steps":   res["num_steps"],
        "wall_time_s": res["wall_time_s"],
        "seed":        task["seed"],
    }


# ---------------------------------------------------------------------------
# Tournament scheduler
# ---------------------------------------------------------------------------

def round_robin_pairs(agents: Sequence[str]) -> list[tuple[str, str]]:
    """All unordered 1v1 pairings of distinct agents."""
    return list(itertools.combinations(agents, 2))


def tournament(
    agents: Sequence[str],
    rounds: int,
    base_seed: int,
    *,
    csv_path: Path = DEFAULT_CSV,
    episode_steps: int = 500,
    flush_period: int = 10,
    workers: int = 1,
    verbose: bool = True,
) -> None:
    """Round-robin 1v1 with slot rotation.

    For each round we pick one seed and play every pair *twice* — once with A
    as player 0 and once with A as player 1 — on the same map. This cancels
    starting-position bias cheaply. Prints a head-to-head win matrix when done.

    ``flush_period`` controls how many matches are completed before buffered
    CSV rows are written to disk (default 10). A final flush always runs at
    the end regardless of the period.

    ``workers`` sets the number of parallel worker processes (default 1 =
    sequential). Pass ``workers=-1`` to use all logical CPU cores.
    """
    pairs = round_robin_pairs(agents)
    if not pairs:
        raise ValueError("Need at least 2 distinct agents.")

    if workers == -1:
        workers = os.cpu_count() or 1

    total = rounds * len(pairs) * 2
    done = 0
    t_start = time.perf_counter()
    match_records: list[dict] = []
    pending_rows: list[list] = []

    tasks = [
        {"slot_order": slot_order, "seed": base_seed + r, "episode_steps": episode_steps}
        for r in range(rounds)
        for a, b in pairs
        for slot_order in [(a, b), (b, a)]
    ]

    def _handle_result(result: dict) -> None:
        nonlocal done
        done += 1
        pending_rows.extend(result["rows"])
        match_records.append({"agents": result["agents"], "ranks": result["ranks"]})
        if done % flush_period == 0:
            _flush_rows(pending_rows, csv_path)
            pending_rows.clear()
        if verbose:
            min_rank = min(result["ranks"])
            top = [result["agents"][i] for i, r in enumerate(result["ranks"]) if r == min_rank]
            winner = top[0] if len(top) == 1 else "draw"
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done)
            slot_order = result["agents"]
            print(
                f"[{done:>4}/{total}] seed={result['seed']:<6} "
                f"{slot_order[0]:>11} vs {slot_order[1]:<11} "
                f"-> winner={winner:<11} "
                f"steps={result['num_steps']:<4} "
                f"({result['wall_time_s']:.2f}s, ETA {eta/60:.1f}m)"
            )

    if workers == 1:
        for task in tasks:
            _handle_result(_match_task(task))
    else:
        with multiprocessing.Pool(workers) as pool:
            for result in pool.imap_unordered(_match_task, tasks):
                _handle_result(result)

    _flush_rows(pending_rows, csv_path)

    print()
    print(format_head_to_head_matrix(match_records, list(agents)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orbit Wars match runner / tournament scheduler.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered agents.")
    p_list.set_defaults(_handler="list")

    def _add_ppo_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("RL agent (optional)")
        g.add_argument("--checkpoint", type=str, default=None,
                       help="Path to a .pt checkpoint. Registers a 'ppo' agent (or --ppo-name).")
        g.add_argument("--config", type=str, default=None,
                       help="Path to train config YAML (default: src default).")
        g.add_argument("--deterministic", action="store_true")
        g.add_argument("--ppo-name", type=str, default="ppo",
                       help="Registry key used for the loaded RL agent (default: ppo).")

    p_match = sub.add_parser("match", help="Run a single match.")
    p_match.add_argument(
        "--agents", nargs="+", required=True,
        help="Agent names in slot order (length 2 or 4). Use 'ppo' for the RL agent.",
    )
    p_match.add_argument("--seed", type=int, default=0)
    p_match.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p_match.add_argument("--episode-steps", type=int, default=500)
    p_match.add_argument("--replay", type=Path, default=None,
                         help="Save HTML replay to this path.")
    _add_ppo_args(p_match)
    p_match.set_defaults(_handler="match")

    real_agents = [a for a in AGENTS if a != "random"]
    p_tour = sub.add_parser("tournament", help="Round-robin 1v1 with slot rotation.")
    p_tour.add_argument(
        "--agents", nargs="+", default=real_agents,
        help=f"Agents to include (default: {real_agents}). Use 'ppo' for the RL agent.",
    )
    p_tour.add_argument("--rounds", type=int, default=10,
                        help="Number of seed rounds. Each round = (#pairs * 2) matches.")
    p_tour.add_argument("--base-seed", type=int, default=1000)
    p_tour.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p_tour.add_argument("--episode-steps", type=int, default=500)
    p_tour.add_argument("--flush-period", type=int, default=10,
                        help="Write buffered CSV rows every N matches (default 10).")
    p_tour.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default 1).")
    p_tour.add_argument("--quiet", action="store_true")
    _add_ppo_args(p_tour)
    p_tour.set_defaults(_handler="tournament")

    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args._handler == "list":
        print("Registered agents:")
        for k, v in AGENTS.items():
            shown = v if isinstance(v, str) else getattr(v, "__name__", repr(v))
            print(f"  {k:<12} -> {shown}")
        return 0

    # Register RL agent before any agent resolution.
    if getattr(args, "checkpoint", None) is not None:
        from src.config import default_train_config_path
        cfg_path = args.config or str(default_train_config_path())
        _register_ppo_agent(cfg_path, args.checkpoint, args.deterministic, args.ppo_name)

    if args._handler == "match":
        if len(args.agents) not in (2, 4):
            print("Orbit Wars supports 2 or 4 player matches.", file=sys.stderr)
            return 2
        if len(set(args.agents)) != len(args.agents):
            print("Duplicate agents in slot list — pass distinct names.", file=sys.stderr)
            return 2
        res = play_match(
            args.agents,
            seed=args.seed,
            episode_steps=args.episode_steps,
            csv_path=args.out,
            replay_path=args.replay,
        )
        print(
            f"match_id={res['match_id']}  seed={res['seed']}  "
            f"steps={res['num_steps']}  rewards={res['rewards']}  "
            f"ranks={res['ranks']}  outcomes={[round(o, 2) for o in res['outcomes']]}  "
            f"wall={res['wall_time_s']:.2f}s"
        )
        if args.replay:
            print(f"replay saved to {args.replay}")
        return 0

    if args._handler == "tournament":
        if len(set(args.agents)) < 2:
            print("Need at least 2 distinct agents.", file=sys.stderr)
            return 2
        tournament(
            list(dict.fromkeys(args.agents)),
            rounds=args.rounds,
            base_seed=args.base_seed,
            csv_path=args.out,
            episode_steps=args.episode_steps,
            flush_period=args.flush_period,
            workers=args.workers,
            verbose=not args.quiet,
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
