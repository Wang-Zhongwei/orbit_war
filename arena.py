"""
arena.py — Match runner & tournament scheduler for Orbit Wars.

Responsibilities
----------------
1. Provide a human-readable CSV log of every match.
2. Run a single match and append one row per slot to ``results/matches.csv``.
3. Schedule a round-robin 1v1 tournament. Because the game is symmetric, every
   pair plays once per seed with a randomized slot ordering.
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
import random
import sys
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.evaluation import (
    AGENTS, run_match_raw, format_head_to_head_matrix,
    _ranks_from_rewards, _outcomes_from_ranks,
)

# Import the batched-eval module NOW, while src_jax resolves to the live tree, and
# hold the reference. Later --jax-checkpoint loading evicts/pins src_jax (possibly
# to a worktree without eval_jax.py), but this held module object survives that.
import src_jax.eval_jax as _eval_jax


CSV_HEADER: list[str] = [
    "match_id", "timestamp", "seed", "format",
    "num_steps", "slot", "agent",
    "reward", "status", "rank", "outcome",
]

DEFAULT_CSV = Path(__file__).parent / "results" / "matches.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_jax_agent(
    checkpoint_path: str,
    deterministic: bool,
    name: str = "jax",
    src_root: str | None = None,
) -> None:
    """Build a JAX BC/RL agent from checkpoint and register it in AGENTS under ``name``.

    ``src_root`` is the directory whose ``src_jax/`` package matches this
    checkpoint's architecture (default: this repo, i.e. arena.py's dir). Pass a
    worktree path when the checkpoint was trained from a tree whose arch differs
    from the working repo (e.g. ``.claude/worktrees/featexp``).

    Why this is fiddly: importing ``src.evaluation`` (top of this module) loads
    the PPO submission opponents (sub_v6/v7/v8…), each of which unpacks its OWN
    embedded ``src_jax`` tree into ``/tmp/owars_ppo_src_<tag>`` and injects it as
    ``sys.modules["src_jax"]``. Whichever imports first wins that slot, so a
    plain ``from src_jax… import build_agent`` here loads the submission's tree
    (e.g. a 30-dim cand_proj) instead of ours, mismatching this checkpoint's
    params (ScopeParamShapeError). Those submission agents are already built at
    import time and hold direct refs to their own modules, so we can safely evict
    the shared slot and re-import the correct tree from ``src_root``.
    """
    import importlib
    import pickle
    import tempfile
    import jax
    import jax.numpy as jnp

    root = src_root or str(Path(__file__).resolve().parent)

    # Evict any hijacked src_jax* (submission /tmp trees) so the import below
    # resolves freshly against ``root``. The submission agent closures keep their
    # own module objects, so dropping these sys.modules keys doesn't affect them.
    for mod_name in [m for m in sys.modules if m == "src_jax" or m.startswith("src_jax.")]:
        del sys.modules[mod_name]
    # Strip the submissions' /tmp src dirs from the path and put ``root`` first
    # so our tree wins resolution.
    tmp_prefix = os.path.join(tempfile.gettempdir(), "owars_ppo_src_")
    sys.path[:] = [p for p in sys.path if not p.startswith(tmp_prefix)]
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)

    build_agent = importlib.import_module("src_jax.jax_agent").build_agent
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    AGENTS[name] = build_agent(params, ckpt["config"], deterministic=deterministic)


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

_WORKER_JAX_CHECKPOINT: str | None = None
_WORKER_JAX_NAME: str = "jax"
_WORKER_JAX_DETERMINISTIC: bool = True


def _worker_init(checkpoint_path: str | None, name: str, deterministic: bool,
                 src_root: str | None = None) -> None:
    """Pool initializer: register the JAX agent once per worker process."""
    global _WORKER_JAX_CHECKPOINT, _WORKER_JAX_NAME, _WORKER_JAX_DETERMINISTIC
    _WORKER_JAX_CHECKPOINT = checkpoint_path
    _WORKER_JAX_NAME = name
    _WORKER_JAX_DETERMINISTIC = deterministic
    if checkpoint_path is not None:
        _register_jax_agent(checkpoint_path, deterministic=deterministic,
                            name=name, src_root=src_root)


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
    jax_checkpoint: str | None = None,
    jax_name: str = "jax",
    jax_deterministic: bool = True,
    jax_src_root: str | None = None,
) -> None:
    """Round-robin 1v1 with randomized slot ordering.

    For each round we pick one seed and play every pair once. Slot ordering is
    randomized per pair because the game is symmetric. Prints a head-to-head
    win matrix when done.

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

    total = rounds * len(pairs)
    done = 0
    t_start = time.perf_counter()
    match_records: list[dict] = []
    pending_rows: list[list] = []

    tasks = []
    for r in range(rounds):
        seed = base_seed + r
        rng = random.Random(seed)
        for a, b in pairs:
            slot_order = (a, b) if rng.randrange(2) == 0 else (b, a)
            tasks.append({"slot_order": slot_order, "seed": seed, "episode_steps": episode_steps})

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
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(jax_checkpoint, jax_name, jax_deterministic, jax_src_root),
        ) as pool:
            for result in pool.imap_unordered(_match_task, tasks):
                _handle_result(result)

    _flush_rows(pending_rows, csv_path)

    print()
    print(format_head_to_head_matrix(match_records, list(agents)))


# ---------------------------------------------------------------------------
# Panel evaluation: one agent vs a list of opponents (1v1 or 1-vs-3 FFA),
# batched on-device when every seat is a JAX policy.
# ---------------------------------------------------------------------------

def _jax_spec_for(name, *, jax_checkpoint, jax_name, jax_src_root, deterministic):
    """Return an eval_jax EvalSpec for ``name`` if it's a JAX-arch agent, else None.

    Recognizes the loaded ``--jax-checkpoint`` agent (registered under jax_name)
    and the JAX submission opponents (sub_vN). Anything else (rule-based) → None,
    signaling the caller to fall back to env.run.
    """
    ej = _eval_jax

    if jax_checkpoint is not None and name == jax_name:
        return ej.spec_from_checkpoint(jax_checkpoint, jax_src_root, deterministic)
    if name in ej._JAX_SUBMISSION_KEYS:
        return ej.spec_from_submission(name)
    return None


def _rows_for_game(match_id, seed, fmt, num_steps, slot_names, rewards):
    """Build per-slot CSV rows + a head-to-head record for one finished game."""
    rewards = [float(r) for r in rewards]
    ranks    = _ranks_from_rewards(rewards)
    outcomes = _outcomes_from_ranks(ranks)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = [
        [match_id, timestamp, seed, fmt, num_steps, slot, name, r, "DONE", rk, f"{oc:.3f}"]
        for slot, (name, r, rk, oc) in enumerate(zip(slot_names, rewards, ranks, outcomes))
    ]
    return rows, {"agents": list(slot_names), "ranks": ranks}, outcomes


def panel(
    eval_agent: str,
    opponents: Sequence[str],
    rounds: int,
    base_seed: int,
    *,
    ffa: bool = False,
    csv_path: Path = DEFAULT_CSV,
    episode_steps: int = 500,
    verbose: bool = True,
    jax_checkpoint: str | None = None,
    jax_name: str = "jax",
    jax_deterministic: bool = True,
    jax_src_root: str | None = None,
) -> dict[str, float]:
    """Evaluate ``eval_agent`` vs each opponent over ``rounds`` seeds.

    1v1 (default): eval vs opponent, side alternated across rounds.
    ``ffa=True``: 1-vs-3 — eval at seat 0 against 3 copies of the opponent.

    For an opponent where eval AND opponent are both JAX-arch, all ``rounds`` games
    play in ONE on-device jitted scan (eval_jax.play_match_batched). Otherwise the
    opponent falls back to the kaggle env.run path (one game per round). Returns
    {opponent: win_rate in [0,1]} (draws = 0.5) and writes the usual CSV rows.
    """
    n_seats = 4 if ffa else 2
    fmt = f"ffa{n_seats}" if ffa else "1v1"

    eval_spec = _jax_spec_for(
        eval_agent, jax_checkpoint=jax_checkpoint, jax_name=jax_name,
        jax_src_root=jax_src_root, deterministic=jax_deterministic,
    )

    all_rows: list[list] = []
    match_records: list[dict] = []
    win_rates: dict[str, float] = {}

    for opp in opponents:
        opp_spec = _jax_spec_for(
            opp, jax_checkpoint=jax_checkpoint, jax_name=jax_name,
            jax_src_root=jax_src_root, deterministic=jax_deterministic,
        )
        batched = eval_spec is not None and opp_spec is not None

        wins = draws = 0
        t0 = time.perf_counter()

        if batched:
            ej = _eval_jax
            # spec 0 = eval, spec 1 = opponent.
            if ffa:
                seat_assign = [[0, 1, 1, 1] for _ in range(rounds)]
            else:
                seat_assign = [[0, 1] if e % 2 == 0 else [1, 0] for e in range(rounds)]
            rewards, steps = ej.play_match_batched(
                [eval_spec, opp_spec], seat_assign, rounds, base_seed, episode_steps,
            )
            for e in range(rounds):
                sa = seat_assign[e]
                slot_names = [eval_agent if j == 0 else opp for j in sa]
                rows, rec, outcomes = _rows_for_game(
                    uuid.uuid4().hex[:12], base_seed + e, fmt, int(steps[e]),
                    slot_names, rewards[e],
                )
                all_rows.extend(rows)
                match_records.append(rec)
                eval_seat = sa.index(0)
                oc = outcomes[eval_seat]
                wins  += int(oc == 1.0)
                draws += int(oc == 0.5)
        else:
            for e in range(rounds):
                seed = base_seed + e
                if ffa:
                    slot_names = [eval_agent, opp, opp, opp]
                else:
                    slot_names = [eval_agent, opp] if e % 2 == 0 else [opp, eval_agent]
                raw = run_match_raw(list(slot_names), seed, episode_steps=episode_steps)
                rows, rec, _ = _rows_for_game(
                    uuid.uuid4().hex[:12], seed, fmt, raw["num_steps"],
                    slot_names, raw["rewards"],
                )
                all_rows.extend(rows)
                match_records.append(rec)
                eval_seat = slot_names.index(eval_agent)
                oc = raw["outcomes"][eval_seat]
                wins  += int(oc == 1.0)
                draws += int(oc == 0.5)

        wr = (wins + 0.5 * draws) / rounds if rounds else 0.0
        win_rates[opp] = wr
        if verbose:
            mode = "batched" if batched else "env.run"
            print(f"  {eval_agent} vs {opp:<11} [{mode:>7}]: "
                  f"win_rate={wr:.3f} ({wins}W {draws}D {rounds - wins - draws}L) "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    _flush_rows(all_rows, csv_path)

    print()
    print(format_head_to_head_matrix(match_records, [eval_agent, *opponents]))
    print()
    print(f"{eval_agent} win rates ({'1-vs-3 FFA' if ffa else '1v1'}, "
          f"{rounds} rounds each):")
    for opp, wr in win_rates.items():
        print(f"  vs {opp:<12} {wr * 100:5.1f}%")
    return win_rates


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

    def _add_jax_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("JAX agent (optional)")
        g.add_argument("--jax-checkpoint", type=str, default=None,
                       help="Path to a .pkl JAX checkpoint. Registers a 'jax' agent (or --jax-name).")
        g.add_argument("--jax-name", type=str, default="jax",
                       help="Registry key used for the loaded JAX agent (default: jax).")
        g.add_argument("--jax-stochastic", action="store_true",
                       help="Sample from the JAX policy instead of taking argmax.")
        g.add_argument("--jax-src-root", type=str, default=None,
                       help="Directory whose src_jax/ matches the checkpoint's arch "
                            "(default: this repo). Set to a .claude/worktrees/<wt> path "
                            "when the checkpoint was trained from a tree whose arch "
                            "differs from the working repo.")

    p_match = sub.add_parser("match", help="Run a single match.")
    p_match.add_argument(
        "--agents", nargs="+", required=True,
        help="Agent names in slot order (length 2 or 4).",
    )
    p_match.add_argument(
        "--ffa", action="store_true",
        help="1-vs-3 free-for-all: play agents[0] against 3 copies of agents[1] "
             "(a 4-player game). Requires exactly 2 --agents; slot 0 is the "
             "evaluated agent, slots 1-3 are the opponent.",
    )
    p_match.add_argument("--seed", type=int, default=0)
    p_match.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p_match.add_argument("--episode-steps", type=int, default=500)
    p_match.add_argument("--replay", type=Path, default=None,
                         help="Save HTML replay to this path.")
    _add_jax_args(p_match)
    p_match.set_defaults(_handler="match")

    real_agents = [a for a in AGENTS if a != "random"]
    real_agents.append('jax')
    p_tour = sub.add_parser("tournament", help="Round-robin 1v1 with randomized slots.")
    p_tour.add_argument(
        "--agents", nargs="+", default=real_agents,
        help=f"Agents to include (default: {real_agents}).",
    )
    p_tour.add_argument("--rounds", type=int, default=20,
                        help="Number of seed rounds. Each round = #pairs matches.")
    p_tour.add_argument("--base-seed", type=int, default=1000)
    p_tour.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p_tour.add_argument("--episode-steps", type=int, default=500)
    p_tour.add_argument("--flush-period", type=int, default=10,
                        help="Write buffered CSV rows every N matches (default 10).")
    p_tour.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default 1).")
    p_tour.add_argument("--quiet", action="store_true")
    _add_jax_args(p_tour)
    p_tour.set_defaults(_handler="tournament")

    p_panel = sub.add_parser(
        "panel",
        help="Evaluate one agent vs a panel of opponents (1v1 or 1-vs-3 FFA), "
             "batched on-device when both sides are JAX policies.",
    )
    p_panel.add_argument("--eval", dest="eval_agent", required=True,
                         help="Agent to evaluate (e.g. 'jax' for --jax-checkpoint, or sub_vN).")
    p_panel.add_argument("--opponents", nargs="+", required=True,
                         help="Opponent agent names to evaluate against.")
    p_panel.add_argument("--rounds", type=int, default=64,
                         help="Games per opponent (default 64).")
    p_panel.add_argument("--ffa", action="store_true",
                         help="1-vs-3 FFA: eval at seat 0 vs 3 copies of each opponent.")
    p_panel.add_argument("--base-seed", type=int, default=1000)
    p_panel.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p_panel.add_argument("--episode-steps", type=int, default=500)
    p_panel.add_argument("--quiet", action="store_true")
    _add_jax_args(p_panel)
    p_panel.set_defaults(_handler="panel")

    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args._handler == "list":
        print("Registered agents:")
        for k, v in AGENTS.items():
            shown = v if isinstance(v, str) else getattr(v, "__name__", repr(v))
            print(f"  {k:<12} -> {shown}")
        return 0

    if getattr(args, "jax_checkpoint", None) is not None:
        _register_jax_agent(
            args.jax_checkpoint,
            deterministic=not args.jax_stochastic,
            name=args.jax_name,
            src_root=getattr(args, "jax_src_root", None),
        )

    if args._handler == "match":
        agents = list(args.agents)
        if args.ffa:
            # 1-vs-3: agents[0] against 3 copies of agents[1] (4-player FFA).
            if len(agents) != 2:
                print("--ffa needs exactly 2 --agents (evaluated opponent); it "
                      "expands to [agent0, agent1, agent1, agent1].", file=sys.stderr)
                return 2
            agents = [agents[0], agents[1], agents[1], agents[1]]
        elif len(agents) not in (2, 4):
            print("Orbit Wars supports 2 or 4 player matches.", file=sys.stderr)
            return 2
        elif len(set(agents)) != len(agents):
            print("Duplicate agents in slot list — pass distinct names "
                  "(or use --ffa for 1-vs-3 copies).", file=sys.stderr)
            return 2
        res = play_match(
            agents,
            seed=args.seed,
            episode_steps=args.episode_steps,
            csv_path=args.out,
            replay_path=args.replay,
        )
        print(
            f"match_id={res['match_id']}  seed={res['seed']}  "
            f"{'ffa(1v3) ' if args.ffa else ''}steps={res['num_steps']}  "
            f"rewards={res['rewards']}  "
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
            jax_checkpoint=getattr(args, "jax_checkpoint", None),
            jax_name=getattr(args, "jax_name", "jax"),
            jax_deterministic=not getattr(args, "jax_stochastic", False),
            jax_src_root=getattr(args, "jax_src_root", None),
        )
        return 0

    if args._handler == "panel":
        panel(
            args.eval_agent,
            list(dict.fromkeys(args.opponents)),
            rounds=args.rounds,
            base_seed=args.base_seed,
            ffa=args.ffa,
            csv_path=args.out,
            episode_steps=args.episode_steps,
            verbose=not args.quiet,
            jax_checkpoint=getattr(args, "jax_checkpoint", None),
            jax_name=getattr(args, "jax_name", "jax"),
            jax_deterministic=not getattr(args, "jax_stochastic", False),
            jax_src_root=getattr(args, "jax_src_root", None),
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
