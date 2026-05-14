"""
evaluation.py — shared match execution and evaluation utilities.

Provides the agent registry, core match runner, result interpretation helpers,
and head-to-head win-rate reporting. Imported by arena.py, tune.py, and
evaluate_elo.py so this logic lives in one place.
"""
from __future__ import annotations

import importlib.util
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from agents.rule_based import nearest_planet_sniper, roi_agent, never_miss_agent, stateful_agent, agent_1000_score  # noqa: E402

# ---------------------------------------------------------------------------
# submission/1000_score.py — load with isolated module state and reset wrapper
# ---------------------------------------------------------------------------
_sub_spec = importlib.util.spec_from_file_location(
    "submission_1000_score",
    os.path.join(os.path.dirname(__file__), "submission", "1000_score.py"),
)
_sub_mod = importlib.util.module_from_spec(_sub_spec)
_sub_spec.loader.exec_module(_sub_mod)


def sub_1000_score(obs) -> list:
    """submission/1000_score.py agent with per-game global-state reset."""
    if obs.get("step", 0) == 0:
        _sub_mod.fleet_trajectories.clear()
        _sub_mod.reinforcement_trajectories.clear()
        _sub_mod.moving_planets.clear()
        _sub_mod.steps = 0
    return _sub_mod.agent(obs)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
AGENTS: dict[str, object] = {
    "sniper":       nearest_planet_sniper,
    "roi":          roi_agent,
    "never_miss":   never_miss_agent,
    "stateful":     stateful_agent,
    "1000_score":   agent_1000_score,
    "sub_1000":     sub_1000_score,
    "random":       "random",
}


# ---------------------------------------------------------------------------
# Result interpretation helpers
# ---------------------------------------------------------------------------

def _ranks_from_rewards(rewards: Sequence[float]) -> list[int]:
    """Dense 1-indexed ranks from rewards (higher reward = better rank).

    Tied rewards share the same rank.
    """
    sorted_unique = sorted(set(rewards), reverse=True)
    rank_of = {r: i + 1 for i, r in enumerate(sorted_unique)}
    return [rank_of[r] for r in rewards]


def _outcomes_from_ranks(ranks: Sequence[int]) -> list[float]:
    """Per-slot win share in [0, 1].

    For 1v1: {1.0, 0.5, 0.0}. For N-player: ``1 - (rank - 1) / (N - 1)``
    averaged over tied slots.
    """
    n = len(ranks)
    if n <= 1:
        return [0.5] * n

    out: list[float] = []
    for r in ranks:
        ties = ranks.count(r)
        if ties > 1:
            shares = [1.0 - (rr - 1) / (n - 1) for rr in range(r, r + ties)]
            out.append(sum(shares) / ties)
        else:
            out.append(1.0 - (r - 1) / (n - 1))
    return out


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match_raw(
    agents: Sequence,
    seed: int,
    *,
    episode_steps: int = 500,
) -> dict:
    """Run one match without any CSV logging.

    ``agents`` may contain string names (resolved via AGENTS) or callables.

    Returns a dict with keys:
        rewards, statuses, ranks, outcomes, num_steps, wall_time_s
    """
    from kaggle_environments import make  # lazy: avoid slow import on --help

    fns = [AGENTS[a] if isinstance(a, str) else a for a in agents]
    n = len(fns)

    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "episodeSteps": int(episode_steps)},
        debug=False,
    )

    t0 = time.perf_counter()
    env.run(fns)
    wall_time = time.perf_counter() - t0

    final = env.steps[-1]
    rewards: list[float] = []
    statuses: list[str] = []
    for s in final:
        if isinstance(s, dict):
            rewards.append(s.get("reward", -1))
            statuses.append(s.get("status", "UNKNOWN"))
        else:
            rewards.append(getattr(s, "reward", -1))
            statuses.append(getattr(s, "status", "UNKNOWN"))

    # The env hands out reward=-1 to *all* players when max_score == 0. Treat
    # that case as a draw.
    if all(r == -1 for r in rewards):
        adjusted = [0.0] * n
    else:
        adjusted = [float(r) for r in rewards]

    ranks = _ranks_from_rewards(adjusted)

    # Force agents that did not finish cleanly to the worst rank.
    bad_slots = [i for i, st in enumerate(statuses) if st != "DONE"]
    if bad_slots:
        good_slots = [i for i in range(n) if i not in bad_slots]
        good_rewards = [adjusted[i] for i in good_slots]
        good_ranks = _ranks_from_rewards(good_rewards) if good_rewards else []
        ranks = [n] * n
        for j, i in enumerate(good_slots):
            ranks[i] = good_ranks[j]

    return {
        "rewards":     rewards,
        "statuses":    statuses,
        "ranks":       ranks,
        "outcomes":    _outcomes_from_ranks(ranks),
        "num_steps":   len(env.steps),
        "wall_time_s": wall_time,
    }


# ---------------------------------------------------------------------------
# Head-to-head evaluation
# ---------------------------------------------------------------------------

def head_to_head(matches: Iterable[dict]) -> tuple[dict, dict]:
    """Return (wins, games) where wins[(a, b)] = matches in which a outranked b.

    Ties contribute 0.5 to both directions.
    """
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    for m in matches:
        agents = m["agents"]
        ranks = m["ranks"]
        n = len(agents)
        for i in range(n):
            for j in range(n):
                if i == j or agents[i] == agents[j]:
                    continue
                games[(agents[i], agents[j])] += 1
                if ranks[i] < ranks[j]:
                    wins[(agents[i], agents[j])] += 1.0
                elif ranks[i] == ranks[j]:
                    wins[(agents[i], agents[j])] += 0.5
    return wins, games


def format_head_to_head_matrix(matches: Iterable[dict], agent_order: list[str]) -> str:
    """Format a head-to-head win-rate table as a printable string.

    Each match dict must have ``agents`` (slot-ordered list of names) and
    ``ranks`` (slot-ordered list of integers) keys.
    """
    wins, games = head_to_head(matches)
    cell_w = 14
    name_w = max(12, max((len(a) for a in agent_order), default=0))
    lines = ["Head-to-head win rate (row beats col):"]
    header = " " * name_w + "".join(f"{a:>{cell_w}}" for a in agent_order)
    lines.append(header)
    for a in agent_order:
        cells = []
        for b in agent_order:
            if a == b:
                cells.append(f"{'-':>{cell_w}}")
            else:
                g = games.get((a, b), 0)
                if g == 0:
                    cells.append(f"{'.':>{cell_w}}")
                else:
                    pct = 100.0 * wins[(a, b)] / g
                    cells.append(f"{pct:>6.1f}% ({g:>3})".rjust(cell_w))
        lines.append(f"{a:<{name_w}}" + "".join(cells))
    return "\n".join(lines)
