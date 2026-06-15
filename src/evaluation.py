from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
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


# Rule-based submissions — pure-Python heuristics with no src_jax dependency, so
# these always load and are safe to import eagerly. sub_v52 (the default PPO eval
# opponent) lives here.
_v2_mod = _load_submission("stateful_v2")
_v3_mod = _load_submission("stateful_v3")
_v5_mod = _load_submission("stateful_v5")
_v51_mod = _load_submission("stateful_v5.1")
_v52_mod = _load_submission("stateful_v5.2")


# ---------------------------------------------------------------------------
# JAX/PPO submission opponents — load against a pinned worktree's src_jax
# ---------------------------------------------------------------------------
# Each of these submissions base64-embeds its OWN src_jax tree (v7 architecture:
# MAX_TICKS_OUT=60 / MAX_TICKS_IN=45, 30-dim cand_proj) and at import time does
#   importlib.import_module("src_jax.jax_agent").build_agent(...)
# to warm-compile its agent. The catch: `import_module` returns whatever is
# already cached in sys.modules["src_jax"]. Inside a PPO trainer's eval workers,
# the LIVE src_jax (current 26-dim tree) is imported first (eval_mp._init_worker
# builds the trainee agent before this module is ever imported), so the
# submission's own `sys.path.insert(0, _SRC_DIR)` is a no-op and it builds its
# 30-dim checkpoint against the live 26-dim encoder → flax.ScopeParamShapeError.
# Eagerly loading them at module scope therefore used to crash the ENTIRE
# src.evaluation import, killing every eval worker and failing the whole PPO job.
#
# Fix: load these opponents with the v7arch worktree pinned as the src_jax
# resolution root — evict any live src_jax* from sys.modules, prime v7arch's
# tree, exec the submission (its import_module now resolves to v7arch's matching
# 30-dim tree), then restore the live tree. The built agents keep direct refs to
# v7arch's module objects, so restoring is safe (same invariant arena.py relies
# on). Loading is also TOLERANT: a submission that still fails (missing worktree,
# unexpected arch) is skipped with a warning so the rule-based opponents — incl.
# sub_v52 — always stay available and the eval never hard-crashes again.
# src_jax resolution roots, keyed by a short label. Each PPO submission embeds a
# src_jax tree built against a specific encoder architecture; it must be loaded
# with the matching tree pinned (see _pinned_src_jax below). Most submissions
# target the v7 architecture, which lives in the v7arch worktree. The "main" root
# is the primary repo checkout itself — it is NOT a registered git worktree under
# .claude/worktrees, just the live working tree — and exists so submissions built
# against the current main src_jax can be assigned to it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_JAX_ROOTS = {
    "v7arch": _REPO_ROOT / ".claude" / "worktrees" / "v7arch",
    "main":   _REPO_ROOT,
}

# Map each PPO submission opponent to (submission_file, src_jax_root_label). The
# root label selects which tree in _SRC_JAX_ROOTS to pin while loading it.
_PPO_SUBMISSIONS = {
    "sub_v6":  ("ppo_from_bc_loser0", "v7arch"),
    "sub_v7":  ("ppo_low_kl_v7",      "v7arch"),
    "sub_v8":  ("sub_v8",             "v7arch"),
    "sub_v81": ("sub_v8.1",           "v7arch"),
    "sub_v82": ("sub_v8.2",           "v7arch"),
}


@contextmanager
def _pinned_src_jax(root: Path):
    """Temporarily make ``root``'s ``src_jax`` the only importable one.

    Evicts any live ``src_jax*`` from ``sys.modules`` and prepends ``root`` to
    ``sys.path`` so a fresh ``import src_jax`` resolves against ``root``; restores
    the previous module objects and path on exit. Already-built agent closures
    keep direct references to ``root``'s modules, so the restore is harmless.
    """
    def _srcjax_keys():
        return [m for m in list(sys.modules) if m == "src_jax" or m.startswith("src_jax.")]

    saved_mods = {m: sys.modules[m] for m in _srcjax_keys()}
    saved_path = list(sys.path)
    for m in saved_mods:
        del sys.modules[m]
    sys.path.insert(0, str(root))
    try:
        # Prime the pinned tree so submissions' import_module("src_jax.*") hits it.
        importlib.import_module("src_jax.jax_agent")
        yield
    finally:
        sys.path[:] = saved_path
        for m in _srcjax_keys():
            del sys.modules[m]
        sys.modules.update(saved_mods)


# Group submissions by their src_jax root so each tree is primed once, then load
# every submission in that group under it. A missing root or a single failing
# submission is skipped with a warning — the rule-based opponents (incl. sub_v52)
# always stay available and the eval never hard-crashes.
_ppo_sub_mods: dict[str, Any] = {}
_by_root: dict[str, list[tuple[str, str]]] = defaultdict(list)
for _agent_key, (_sub_file, _root_label) in _PPO_SUBMISSIONS.items():
    _by_root[_root_label].append((_agent_key, _sub_file))

for _root_label, _group in _by_root.items():
    _root = _SRC_JAX_ROOTS.get(_root_label)
    _names = ", ".join(k for k, _ in _group)
    if _root is None:
        warnings.warn(
            f"src.evaluation: unknown src_jax root {_root_label!r}; JAX opponents "
            f"{_names} are unavailable. Rule-based opponents (incl. sub_v52) still work.",
            stacklevel=2,
        )
        continue
    if not _root.is_dir():
        warnings.warn(
            f"src.evaluation: {_root_label!r} src_jax root not found at {_root}; "
            f"JAX opponents {_names} are unavailable. Rule-based opponents "
            f"(incl. sub_v52) still work.",
            stacklevel=2,
        )
        continue
    try:
        with _pinned_src_jax(_root):
            for _agent_key, _sub_file in _group:
                try:
                    _ppo_sub_mods[_agent_key] = _load_submission(_sub_file)
                except Exception as _exc:  # ScopeParamShapeError, missing ckpt, …
                    warnings.warn(
                        f"src.evaluation: skipping JAX opponent {_agent_key!r} "
                        f"({_sub_file!r}) — failed to load against the pinned "
                        f"{_root_label!r} src_jax: {type(_exc).__name__}: {_exc}",
                        stacklevel=2,
                    )
    except Exception as _exc:  # priming this root's src_jax failed
        warnings.warn(
            f"src.evaluation: could not pin {_root_label!r} src_jax at {_root} "
            f"({type(_exc).__name__}: {_exc}); JAX opponents {_names} are "
            f"unavailable. Rule-based opponents (incl. sub_v52) still work.",
            stacklevel=2,
        )


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


def _make_ppo_opponent(agent_key: str):
    """Wrap a (successfully loaded) JAX submission module as an opponent fn."""
    mod = _ppo_sub_mods[agent_key]
    return lambda obs: mod.agent(obs)


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
    "exp30":      exp30_agent,
    "random":     _make_random_opponent(),
}

# Register only the JAX opponents that actually loaded against the pinned v7arch
# tree; ones that were skipped above are simply absent (callers that name them
# get a clear KeyError instead of an import-time crash).
for _agent_key in _ppo_sub_mods:
    AGENTS[_agent_key] = _make_ppo_opponent(_agent_key)



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
