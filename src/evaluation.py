from __future__ import annotations

import importlib.util
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .agents.rule_based import agent_1000_score, never_miss_agent, nearest_planet_sniper, roi_agent, stateful_agent, exp30_agent
from .config import TrainConfig
from .agents.rl.ppo_agent import build_moves, build_policy
from .features import encode_turn

_SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submission"


# ---------------------------------------------------------------------------
# submission/1000_score.py — load with isolated module state and reset wrapper
# ---------------------------------------------------------------------------
_sub_spec = importlib.util.spec_from_file_location(
    "submission_1000_score",
    str(_SUBMISSION_DIR / "1000_score.py"),
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


def _load_submission(name: str):
    mod_name = f"submission_{name}"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(_SUBMISSION_DIR / f"{name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_v2_mod = _load_submission("stateful_v2")
_v3_mod = _load_submission("stateful_v3")
_v5_mod = _load_submission("stateful_v5")
_v51_mod = _load_submission("stateful_v5.1")
_v52_mod = _load_submission("stateful_v5.2")
_v6_mod = _load_submission("ppo_from_bc_loser0")
_v7_mod = _load_submission("ppo_low_kl_v7")


def sub_stateful_v2(obs) -> list:
    return _v2_mod.agent(obs)


def sub_stateful_v3(obs) -> list:
    return _v3_mod.agent(obs)


def sub_stateful_v5(obs) -> list:
    return _v5_mod.agent(obs)


def sub_stateful_v51(obs) -> list:
    return _v51_mod.agent(obs)


def sub_stateful_v52(obs) -> list:
    return _v52_mod.agent(obs)


def sub_ppo_from_bc_loser0(obs) -> list:
    return _v6_mod.agent(obs)


def sub_ppo_low_kl_v7(obs) -> list:
    return _v7_mod.agent(obs)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
def _make_random_opponent():
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
    def _random(observation: Any) -> list:
        return list(random_agent({
            "player": obs_get(observation, "player", 0),
            "planets": list(obs_get(observation, "planets", [])),
        }))
    return _random


AGENTS: dict[str, object] = {
    "sniper":     nearest_planet_sniper,
    "roi":        roi_agent,
    "never_miss": never_miss_agent,
    "stateful":   stateful_agent,
    "1000_score": agent_1000_score,
    "sub_1000":   sub_1000_score,
    "sub_v2":     sub_stateful_v2,
    "sub_v3":     sub_stateful_v3,
    "sub_v5":     sub_stateful_v5,
    "sub_v51":    sub_stateful_v51,
    "sub_v52":    sub_stateful_v52,
    "sub_v6":     sub_ppo_from_bc_loser0,
    "sub_v7":     sub_ppo_low_kl_v7,
    "exp30":      exp30_agent,
    "random":     _make_random_opponent(),
}



class SelfPlayOpponent:
    """Live self-play opponent — synced with the training policy each interval."""

    is_live: bool = True

    def __init__(self, cfg: TrainConfig, device: torch.device, deterministic: bool = True) -> None:
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy = build_policy(cfg, device)
        self.policy.eval()

    def sync_from(self, state_dict: dict) -> None:
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def __call__(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        return build_moves(batch, self.policy, self.device, self.deterministic)


class CheckpointOpponent(SelfPlayOpponent):
    """Fixed-checkpoint opponent — loaded once from disk, never synced."""

    is_live: bool = False

    def __init__(self, checkpoint_path: str, cfg: TrainConfig, device: torch.device) -> None:
        super().__init__(cfg, device, deterministic=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> Any:
    """Build an opponent by name.

    Supported names:
    - ``"self"``       live self-play (synced each interval)
    - ``"ckpt:PATH"``  fixed checkpoint at PATH (never synced)
    - ``"random"``     Kaggle built-in random agent
    - any key in AGENTS (e.g. ``"never_miss"``, ``"sub_v3"``)
    """
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for self opponent")
        return SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    if name.startswith("ckpt:"):
        if cfg is None or device is None:
            raise ValueError("cfg and device are required for checkpoint opponent")
        return CheckpointOpponent(name[5:], cfg, device)
    if name in AGENTS:
        fn = AGENTS[name]
        if callable(fn):
            return fn
        raise ValueError(f"Agent '{name}' is registered but not callable: {fn!r}")
    raise ValueError(
        f"Unknown opponent {name!r}. Available: {sorted(AGENTS)} + 'self' + 'ckpt:PATH'"
    )


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)


# ---------------------------------------------------------------------------
# Result interpretation helpers
# ---------------------------------------------------------------------------

def _ranks_from_rewards(rewards: Sequence[float]) -> list[int]:
    """Dense 1-indexed ranks from rewards (higher reward = better rank)."""
    sorted_unique = sorted(set(rewards), reverse=True)
    rank_of = {r: i + 1 for i, r in enumerate(sorted_unique)}
    return [rank_of[r] for r in rewards]


def _outcomes_from_ranks(ranks: Sequence[int]) -> list[float]:
    """Per-slot win share in [0, 1]."""
    n = len(ranks)
    if n <= 1:
        return [0.5] * n
    out: list[float] = []
    for r in ranks:
        ties = ranks.count(r)
        if ties > 1:
            shares = [max(0.0, 1.0 - (rr - 1) / (n - 1)) for rr in range(r, r + ties)]
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
    """Run one match without any CSV logging."""
    from kaggle_environments import make

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
            r = s.get("reward")
            rewards.append(-1 if r is None else r)
            statuses.append(s.get("status", "UNKNOWN"))
        else:
            r = getattr(s, "reward", None)
            rewards.append(-1 if r is None else r)
            statuses.append(getattr(s, "status", "UNKNOWN"))

    if all(r == -1 for r in rewards):
        adjusted = [0.0] * n
    else:
        adjusted = [float(r) for r in rewards]

    ranks = _ranks_from_rewards(adjusted)

    bad_slots = [i for i, st in enumerate(statuses) if st != "DONE"]
    if bad_slots:
        good_slots = [i for i in range(n) if i not in bad_slots]
        if good_slots:
            good_rewards = [adjusted[i] for i in good_slots]
            good_ranks = _ranks_from_rewards(good_rewards)
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

def eval_agent(
    agent,
    opponent_fns: dict[str, object],
    rounds: int,
    base_seed: int,
) -> dict[str, float]:
    """Evaluate a callable agent against named opponents.

    Each opponent gets `rounds` matches with slot randomised per seed.
    Returns win rates in [0, 1] (draws count as 0.5).
    """
    import random as _random
    win_rates: dict[str, float] = {}
    for name, opp_fn in opponent_fns.items():
        wins = draws = total = 0
        for r in range(rounds):
            seed = base_seed + r
            agent_slot = _random.Random(seed).randrange(2)
            slot_order = [(agent, opp_fn), (opp_fn, agent)][agent_slot]
            res = run_match_raw(list(slot_order), seed=seed)
            outcome = res["outcomes"][agent_slot]
            if outcome == 1.0:
                wins += 1
            elif outcome == 0.5:
                draws += 1
            total += 1
        win_rates[name] = (wins + 0.5 * draws) / total if total else 0.0
    return win_rates



# ---------------------------------------------------------------------------
# Head-to-head evaluation
# ---------------------------------------------------------------------------

def head_to_head(matches: Iterable[dict]) -> tuple[dict, dict]:
    """Return (wins, games) where wins[(a, b)] = matches in which a outranked b."""
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
    """Format a head-to-head win-rate table as a printable string."""
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
