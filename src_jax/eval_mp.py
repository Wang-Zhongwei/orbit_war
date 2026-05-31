"""
Multiprocessing evaluation of a JAX policy against rule-based opponents.

The in-loop PPO eval (``train.run_eval``) plays every (opponent, round) match
sequentially in the trainer process, invoking the policy on the GPU once per
env step — slow and fully serialized.  This module fans those matches out
across worker processes, mirroring the ``Pool.imap_unordered`` pattern in
``arena.py``.

Workers are pinned to CPU so they never contend with the trainer for the GPU.
``JAX_PLATFORMS=cpu`` is set for the whole pool lifetime: spawned children
inherit it (and select the CPU backend when they import JAX), while the parent's
already-initialized GPU backend is unaffected — the env var is only read at the
first JAX initialization, which the trainer did long ago.

Win-rate semantics are identical to ``src.evaluation.eval_agent``: ``rounds``
matches per opponent, slot randomized per seed, draws count as 0.5.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from contextlib import contextmanager

# Per-worker globals, populated once by ``_init_worker`` (avoids rebuilding /
# recompiling the policy for every match).
_AGENT = None
_OPPONENTS: dict = {}


def _init_worker(params_host, cfg, deterministic: bool, eval_opponents) -> None:
    # Defensive: also set CPU here in case the child reached the initializer
    # without having imported JAX yet.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from .jax_agent import build_agent
    from src.evaluation import AGENTS

    global _AGENT, _OPPONENTS
    params = jax.tree_util.tree_map(jnp.asarray, params_host)
    _AGENT = build_agent(params, cfg, deterministic=deterministic)
    _OPPONENTS = {opp: AGENTS[opp] for opp in eval_opponents}


def _eval_task(task) -> tuple[str, float]:
    """Play one (opponent, seed) match and return ``(opp_name, outcome)``."""
    import random as _random

    from src.evaluation import run_match_raw

    opp_name, seed = task
    opp_fn = _OPPONENTS[opp_name]
    agent_slot = _random.Random(seed).randrange(2)
    slot_order = [(_AGENT, opp_fn), (opp_fn, _AGENT)][agent_slot]
    res = run_match_raw(list(slot_order), seed=seed)
    return opp_name, res["outcomes"][agent_slot]


@contextmanager
def _children_on_cpu():
    """Force spawned children onto the CPU JAX backend; restore parent env after."""
    saved = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = saved


def run_eval_mp(
    params,
    cfg,
    base_seed: int = 1000,
    *,
    workers: int = -1,
    deterministic: bool = True,
) -> dict[str, float]:
    """Win rate of ``params`` vs each ``cfg.eval_opponents`` entry, matches run
    in parallel across worker processes.

    ``workers=-1`` uses all logical CPU cores (capped at the number of matches).
    """
    eval_opponents = list(cfg.eval_opponents)
    if not eval_opponents:
        return {}

    import jax
    import numpy as np

    # Pull params to host once (parent GPU backend) so we never pickle device
    # arrays to the children.
    params_host = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    rounds = cfg.eval_rounds
    tasks = [(opp, base_seed + r) for opp in eval_opponents for r in range(rounds)]

    if workers == -1:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, len(tasks)))

    wins = {opp: 0.0 for opp in eval_opponents}
    totals = {opp: 0 for opp in eval_opponents}

    ctx = mp.get_context("spawn")
    with _children_on_cpu():
        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(params_host, cfg, deterministic, eval_opponents),
        ) as pool:
            for opp_name, outcome in pool.imap_unordered(_eval_task, tasks):
                totals[opp_name] += 1
                if outcome == 1.0:
                    wins[opp_name] += 1.0
                elif outcome == 0.5:
                    wins[opp_name] += 0.5

    return {
        opp: (wins[opp] / totals[opp] if totals[opp] else 0.0)
        for opp in eval_opponents
    }
