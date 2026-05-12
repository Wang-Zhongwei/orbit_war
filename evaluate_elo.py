"""
evaluate_elo.py — compute Elo ratings from matches.csv.

Reads the long-format match log written by ``arena.py`` and walks through every
match in chronological order applying pairwise Elo updates per match:

* For 1v1 matches this is the standard Elo formula.
* For N>2 matches we use the pairwise-sum trick with K' = K / (N - 1) so the
  per-match rating change is comparable to a single 1v1 result.

This script is read-only over the CSV. You can re-run it with different K
values, after appending more matches, or after adding new agents — no replays
needed.

Usage
-----
    python evaluate_elo.py
    python evaluate_elo.py --csv results/matches.csv --k 32 --bootstrap 1000
    python evaluate_elo.py --plot results/ratings.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from evaluation import head_to_head, format_head_to_head_matrix

DEFAULT_CSV = Path(__file__).parent / "results" / "matches.csv"
DEFAULT_INIT = 1500.0
DEFAULT_K = 32.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_matches(csv_path: Path) -> list[dict]:
    """Group rows into matches sorted chronologically (timestamp, then match_id).

    Each returned dict has ``match_id``, ``timestamp``, ``format``, ``agents``
    (slot-ordered list), ``ranks`` (slot-ordered list), and the raw rows for
    downstream win-rate counting.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    by_id: dict[str, list[dict]] = defaultdict(list)
    first_seen: dict[str, str] = {}
    for r in rows:
        mid = r["match_id"]
        by_id[mid].append(r)
        first_seen.setdefault(mid, r["timestamp"])

    ordered_ids = sorted(by_id, key=lambda m: (first_seen[m], m))

    matches: list[dict] = []
    for mid in ordered_ids:
        slot_rows = sorted(by_id[mid], key=lambda r: int(r["slot"]))
        matches.append({
            "match_id":  mid,
            "timestamp": first_seen[mid],
            "format":    slot_rows[0]["format"],
            "agents":    [r["agent"] for r in slot_rows],
            "ranks":     [int(r["rank"]) for r in slot_rows],
            "outcomes":  [float(r["outcome"]) for r in slot_rows],
            "rows":      slot_rows,
        })
    return matches


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------
def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def _pairwise_outcome(rank_i: int, rank_j: int) -> float:
    if rank_i < rank_j:
        return 1.0
    if rank_i == rank_j:
        return 0.5
    return 0.0


def compute_elo(
    matches: Iterable[dict],
    *,
    k: float = DEFAULT_K,
    init: float = DEFAULT_INIT,
) -> tuple[dict[str, float], dict[str, int], list[tuple[int, dict[str, float]]]]:
    """Walk the match list, return (ratings, games_played, history).

    ``history`` is a list of ``(match_idx, ratings_snapshot)`` after each match,
    useful for plotting convergence.
    """
    ratings: dict[str, float] = defaultdict(lambda: init)
    games: dict[str, int] = defaultdict(int)
    history: list[tuple[int, dict[str, float]]] = []

    for idx, m in enumerate(matches):
        agents = m["agents"]
        ranks = m["ranks"]
        n = len(agents)
        if n < 2:
            continue
        # K' = K / (N-1) so a player's per-match swing matches a 1v1 game.
        k_eff = k / (n - 1)

        # Accumulate deltas first so all pairs are scored against the same
        # pre-match ratings (otherwise updates leak within the match).
        delta: dict[str, float] = defaultdict(float)
        for i in range(n):
            for j in range(i + 1, n):
                a_i, a_j = agents[i], agents[j]
                if a_i == a_j:
                    continue
                s_ij = _pairwise_outcome(ranks[i], ranks[j])
                e_ij = expected_score(ratings[a_i], ratings[a_j])
                delta[a_i] += k_eff * (s_ij - e_ij)
                delta[a_j] += k_eff * ((1.0 - s_ij) - (1.0 - e_ij))

        for a, d in delta.items():
            ratings[a] += d
        for a in agents:
            games[a] += 1
        history.append((idx, dict(ratings)))

    return dict(ratings), dict(games), history


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def bootstrap_ci(
    matches: list[dict],
    *,
    n_resample: int,
    k: float,
    init: float,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Block-bootstrap 95% CIs by resampling matches with replacement.

    Caveat: Elo is order-dependent, so resampling reshuffles match order — this
    captures sampling uncertainty but conflates it with order effects. Good
    enough for a sanity check; switch to TrueSkill for a proper posterior.
    """
    rng = random.Random(seed)
    n = len(matches)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resample):
        resampled = [matches[rng.randrange(n)] for _ in range(n)]
        ratings, _, _ = compute_elo(resampled, k=k, init=init)
        for a, r in ratings.items():
            samples[a].append(r)
    out: dict[str, tuple[float, float]] = {}
    for a, vals in samples.items():
        vals.sort()
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        out[a] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_summary(
    ratings: dict[str, float],
    games: dict[str, int],
    matches: list[dict],
    *,
    init: float,
    ci: dict[str, tuple[float, float]] | None,
) -> list[str]:
    # Per-agent average outcome (= win rate in 1v1, win-share in FFA).
    score_sum: dict[str, float] = defaultdict(float)
    score_n: dict[str, int] = defaultdict(int)
    for m in matches:
        for a, o in zip(m["agents"], m["outcomes"]):
            score_sum[a] += o
            score_n[a] += 1

    agent_order = sorted(ratings, key=lambda a: ratings[a], reverse=True)
    name_w = max(10, max((len(a) for a in agent_order), default=0))
    cols = [
        f"{'agent':<{name_w}}",
        f"{'games':>6}",
        f"{'score%':>7}",
        f"{'elo':>8}",
        f"{'Δ init':>8}",
    ]
    if ci:
        cols.append(f"{'95% CI':>20}")
    print("  ".join(cols))
    print("-" * len("  ".join(cols)))
    for a in agent_order:
        n = score_n.get(a, 0)
        sc = (score_sum[a] / n * 100.0) if n else float("nan")
        row = [
            f"{a:<{name_w}}",
            f"{games[a]:>6}",
            f"{sc:>6.1f}%",
            f"{ratings[a]:>8.1f}",
            f"{ratings[a] - init:>+8.1f}",
        ]
        if ci and a in ci:
            row.append(f"[{ci[a][0]:>7.1f}, {ci[a][1]:>7.1f}]")
        print("  ".join(row))
    return agent_order


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _plot_history(history, init, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot.", file=sys.stderr)
        return

    all_agents = sorted({a for _, snap in history for a in snap})
    fig, ax = plt.subplots(figsize=(10, 6))
    for a in all_agents:
        xs, ys, current = [], [], init
        for idx, snap in history:
            if a in snap:
                current = snap[a]
            xs.append(idx)
            ys.append(current)
        ax.plot(xs, ys, label=a, linewidth=1.5)
    ax.axhline(init, linestyle="--", color="grey", alpha=0.5, linewidth=1)
    ax.set_xlabel("match #")
    ax.set_ylabel("Elo rating")
    ax.set_title("Rating trajectory")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute Elo ratings from matches.csv.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--k", type=float, default=DEFAULT_K, help="Base K-factor (default: 32).")
    p.add_argument("--init", type=float, default=DEFAULT_INIT, help="Starting Elo (default: 1500).")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap resamples for 95%% CIs (0 disables). 500-1000 is plenty.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--plot", type=Path, default=None,
                   help="Save Elo-over-time PNG to this path.")
    p.add_argument("--filter-format", default=None,
                   help="Only score matches whose format matches this string (e.g. '1v1').")
    args = p.parse_args(argv)

    try:
        matches = load_matches(args.csv)
    except FileNotFoundError:
        print(f"{args.csv} not found. Run arena.py first.", file=sys.stderr)
        return 1

    if args.filter_format:
        matches = [m for m in matches if m["format"] == args.filter_format]

    if not matches:
        print("No matches in log.", file=sys.stderr)
        return 1

    ratings, games, history = compute_elo(matches, k=args.k, init=args.init)

    ci = (
        bootstrap_ci(matches, n_resample=args.bootstrap, k=args.k,
                     init=args.init, seed=args.bootstrap_seed)
        if args.bootstrap > 0 else None
    )

    print(f"\nLoaded {len(matches)} matches from {args.csv}")
    print(f"K-factor = {args.k}, init rating = {args.init}", end="")
    if args.filter_format:
        print(f", format = {args.filter_format}", end="")
    print("\n")

    agent_order = _print_summary(ratings, games, matches, init=args.init, ci=ci)

    print()
    print(format_head_to_head_matrix(matches, agent_order))

    if args.plot:
        _plot_history(history, args.init, args.plot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
