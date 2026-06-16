"""
Fully-JAX batched evaluation: learner vs a JAX opponent, all on-device.

The default eval path (``eval_mp.run_eval_mp`` → ``src.evaluation.run_match_raw``
→ kaggle ``env.run``) plays every match in a Python per-tick loop, invoking each
agent's closure on the host once per step. When BOTH players are JAX policies that
is pure overhead: a scalar (N=1) forward pass per tick, per player, ~500 ticks per
game, games run one-at-a-time. The training rollout, by contrast, runs the whole
episode — env step → encode → policy → action build — as one ``jax.jit`` /
``jax.vmap`` / ``jax.lax.scan`` kernel across all envs at once.

This module gives eval the same treatment for the all-JAX-agents case: it plays
``rounds`` games in parallel inside a single jitted scan, learner alternating
sides, and reads each game's terminal reward to get win/draw/loss. It handles:

  * ``"self"``        — opponent = current learner params (same architecture)
  * ``"ckpt:PATH"``   — opponent = a checkpoint (its own embedded config/arch)
  * JAX submissions   — ``sub_v6/v7/v8/...`` whose 30-dim v7arch tree shares this
                        tree's byte-identical ``physics.py`` / ``EnvState`` /
                        ``obs_to_env_state``; only the encoder (feature horizons /
                        cand-dim) and policy weights differ. Each side therefore
                        encodes the *shared* ``EnvState`` with its OWN encoder +
                        module — the only cross-tree interface is the (identical)
                        ``EnvState`` pytree and the ``[planet_id, angle, ships]``
                        action-row format that ``jax_step`` consumes.

Rule-based opponents (sub_v52, sniper, …) are NOT handled here — they have no JAX
policy and stay on the mp/``env.run`` path. ``train.run_eval`` routes opponents
to the right backend.

NOTE on parity: win-rate numbers from this path are NOT bit-identical to the mp
path — boards come from ``env.make_board_pool`` (JAX generation) rather than the
kaggle ``make(seed=…)`` generator, and physics is the deterministic JAX engine.
It is a fair eval (slots split evenly via alternating sides), just a different
board distribution, so absolute numbers shift when switching backends.
"""
from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import (
    jax_step,
    jax_step_jit,
    init_jax_state,
    MAX_BASE_PLANETS,
    MAX_ACTIONS_PA,
    N_AGENTS,
)
from .features import MIN_SOURCE_SHIPS, _make_env_encoder
from .feature_versions import make_env_encoder
from .policies import build_policy
from .rollout import _build_actions_single
from .env import make_board_pool, select_boards

MAX_SRC   = MAX_BASE_PLANETS
_REPO_ROOT = Path(__file__).resolve().parent.parent


class EvalSpec(NamedTuple):
    """Everything needed to run one JAX policy in the batched eval scan.

    ``module``/``encode_fn`` may come from this tree OR a pinned worktree — they
    only ever touch the shared (byte-identical) EnvState and emit standard
    ``[planet_id, angle, ships]`` action rows, so specs from different trees mix
    freely in one scan.
    """
    module: object
    encode_fn: object
    params: object
    deterministic: bool

# Opponent names whose policy is a JAX network this module can run on-device.
# (The submission keys mirror ``src.evaluation._PPO_SUBMISSIONS``.)
_JAX_SUBMISSION_KEYS = ("sub_v6", "sub_v7", "sub_v8", "sub_v81", "sub_v82")


def is_jax_arch_opponent(name: str) -> bool:
    """True for opponents whose action comes from a JAX policy (runnable here)."""
    return name == "self" or name.startswith("ckpt:") or name in _JAX_SUBMISSION_KEYS


# ---------------------------------------------------------------------------
# Per-side action function (batched, mirrors jax_agent.build_agent's selection)
# ---------------------------------------------------------------------------

def _make_act_fn(module, encode_fn, *, deterministic: bool):
    """Return ``act(params, state_batch[N,...], players[N], key) -> (rows[N,A,3], key)``.

    ``params`` is a runtime argument (not closed over) so the same compiled scan
    can be reused across eval intervals as the learner updates — only the param
    *values* change, not the shapes, so no recompile.

    Mirrors ``jax_agent.build_agent`` exactly (safe target-logits → argmax /
    categorical → ships = max(MIN_SOURCE_SHIPS, src_ships), exact precomputed
    angles), so the on-device eval reproduces what the agent does in real matches.
    Works with the live OR a v7arch module/encoder — the orchestration is generic
    and the shared constants (MAX_SRC, MAX_ACTIONS_PA, MIN_SOURCE_SHIPS) are
    byte-identical across the two trees.
    """

    def _one_env_actions(src_ids, src_ships, cand_mask, angles, src_valid, ti):
        row_idx    = jnp.arange(MAX_SRC, dtype=jnp.int32)
        ships_send = jnp.maximum(MIN_SOURCE_SHIPS, src_ships)
        tgt_angles = angles[row_idx, ti]
        return _build_actions_single(src_ids, cand_mask, src_valid, ti, ships_send, tgt_angles)

    def act(params, state_batch, players, key):
        n   = players.shape[0]
        enc = encode_fn(state_batch, players)            # leading axis = n

        gdim = enc.global_feat.shape[-1]
        gf   = jnp.broadcast_to(enc.global_feat[:, None, :], (n, MAX_SRC, gdim))

        out = jax.vmap(
            lambda sf, cf, gf_i, cm, sv: module.apply(params, sf, cf, gf_i, cm, sv)
        )(enc.self_feats, enc.cand_feats, gf, enc.cand_mask, enc.src_valid)

        tl      = out.target_logits                      # [n, MAX_SRC, K]
        invalid = ~jnp.isfinite(tl).any(axis=-1)
        tl      = jnp.where(invalid[..., None], jnp.zeros_like(tl), tl)
        if deterministic:
            ti = jnp.argmax(tl, axis=-1)                 # [n, MAX_SRC]
        else:
            key, sub = jax.random.split(key)
            ti = jax.random.categorical(sub, tl, axis=-1)

        rows = jax.vmap(_one_env_actions)(
            enc.src_ids, enc.src_ships, enc.cand_mask, enc.angles, enc.src_valid, ti,
        )                                                # [n, MAX_ACTIONS_PA, 3]
        return rows, key

    return act


# ---------------------------------------------------------------------------
# Joint action assembly + scan
# ---------------------------------------------------------------------------

def _joint_actions(learner_players, l_acts, o_acts, n_envs):
    """[N, N_AGENTS, MAX_ACTIONS_PA, 3] with learner/opp in slots 0/1 per side."""
    dummy = jnp.full((n_envs, N_AGENTS - 2, MAX_ACTIONS_PA, 3), -1.0)

    def one_env(lp, l_act, o_act):
        p0 = jax.lax.cond(lp == 0, lambda: l_act, lambda: o_act)
        p1 = jax.lax.cond(lp == 0, lambda: o_act, lambda: l_act)
        return jnp.stack([p0, p1], axis=0)

    two = jax.vmap(one_env)(learner_players, l_acts, o_acts)
    return jnp.concatenate([two, dummy], axis=1)


def _make_run_fn(learner_act, opp_act, episode_steps: int, n_envs: int):
    """Build the jitted scan that plays one full game per env and returns the
    per-env outcome from the learner's perspective (1.0 win / 0.5 draw / 0.0 lose).

    ``learner_params`` / ``opp_params`` are run() arguments so the compilation is
    reused across eval intervals (only learner param values change).
    """
    batched_step = jax.vmap(functools.partial(jax_step, episode_steps=episode_steps))
    env_idx      = jnp.arange(n_envs, dtype=jnp.int32)

    def step(learner_params, opp_params, carry, _):
        state, learner_players, key, finished, outcome = carry

        l_rows, key = learner_act(learner_params, state, learner_players, key)
        o_rows, key = opp_act(opp_params, state, 1 - learner_players, key)
        joint       = _joint_actions(learner_players, l_rows, o_rows, n_envs)

        new_state, statuses, rewards = batched_step(state, joint)

        done = statuses[env_idx, learner_players].astype(jnp.bool_)        # [N]
        l_r  = rewards[env_idx, learner_players]
        o_r  = rewards[env_idx, 1 - learner_players]
        this = jnp.where(l_r > o_r, 1.0, jnp.where(l_r == o_r, 0.5, 0.0))   # [N]

        newly    = done & (~finished)
        outcome  = jnp.where(newly, this, outcome)
        finished = finished | done
        return (new_state, learner_players, key, finished, outcome), None

    @jax.jit
    def run(learner_params, opp_params, initial_state, learner_players, key):
        init = (
            initial_state,
            learner_players,
            key,
            jnp.zeros(n_envs, dtype=jnp.bool_),
            jnp.zeros(n_envs, dtype=jnp.float32),
        )
        scan_step = functools.partial(step, learner_params, opp_params)
        (_, _, _, _, outcome), _ = jax.lax.scan(scan_step, init, None, length=episode_steps)
        return outcome

    return run


# ---------------------------------------------------------------------------
# Opponent spec resolution
# ---------------------------------------------------------------------------

def _self_or_ckpt_spec(name, learner_cfg, learner_params):
    """(module, encode_fn, params, deterministic) for self / ckpt opponents."""
    if name == "self":
        cfg    = learner_cfg
        params = learner_params
        det    = bool(getattr(learner_cfg, "self_play_deterministic", False))
    else:  # "ckpt:PATH"
        with open(name[5:], "rb") as f:
            ckpt = pickle.load(f)
        cfg    = ckpt["config"]
        params = ckpt["params"]
        det    = True
    module    = build_policy(cfg)
    encode_fn = make_env_encoder(
        cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7")
    )
    return module, encode_fn, params, det


def _submission_spec(name):
    """(module, encode_fn, params, deterministic) for a JAX submission opponent.

    A submission's policy lives in a specific src_jax tree (e.g. the v7arch
    worktree: 30-dim cand_proj, horizons 60/45). We rebuild its module + encoder
    from *that* tree (pinned the same way src.evaluation loads it) so the param
    tree matches, then run them on the *shared* (byte-identical) EnvState inside
    the eval scan. The only requirement is that the chosen tree's physics/EnvState
    match this tree's — which holds for every registered root.

    Works with both src.evaluation API shapes: the per-root one on main
    (``_SRC_JAX_ROOTS`` + ``_PPO_SUBMISSIONS[name] = (file, root_label)``) and the
    older single-root one on v7_arch (``_V7ARCH_ROOT``).
    """
    import importlib

    from src import evaluation as E

    if name not in E._ppo_sub_mods:
        raise KeyError(
            f"JAX submission opponent {name!r} did not load (see src.evaluation "
            f"warnings); cannot run on the JAX eval fast-path."
        )
    ckpt = E._ppo_sub_mods[name]._ckpt   # {"params": ..., "config": TrainConfig}
    if hasattr(E, "_SRC_JAX_ROOTS"):                       # per-root API (main)
        root = E._SRC_JAX_ROOTS[E._PPO_SUBMISSIONS[name][1]]
    else:                                                  # single-root API (v7_arch)
        root = E._V7ARCH_ROOT

    with E._pinned_src_jax(root):
        pinned_policies = importlib.import_module("src_jax.policies")
        pinned_features = importlib.import_module("src_jax.features")
        cfg       = ckpt["config"]
        params    = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        module    = pinned_policies.build_policy(cfg)
        encode_fn = pinned_features._make_env_encoder(cfg.env.candidate_count, cfg.env)
    return module, encode_fn, params, True


def _resolve_opp_spec(name, learner_cfg, learner_params):
    if name == "self" or name.startswith("ckpt:"):
        return _self_or_ckpt_spec(name, learner_cfg, learner_params)
    if name in _JAX_SUBMISSION_KEYS:
        return _submission_spec(name)
    raise ValueError(f"{name!r} is not a JAX-arch opponent for eval_jax")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Per-process cache of compiled eval scans, keyed by (opponent, n_envs,
# episode_steps, learner_deterministic). The learner module/encoder are baked in
# from the FIRST call; within a training process the arch is constant, so reusing
# them is correct and lets each eval interval skip recompilation (only the learner
# param *values* change, and those are run() arguments).
_RUN_CACHE: dict = {}


def _get_run_and_opp_params(name, cfg, learner_act, n_envs, episode_steps, deterministic_learner):
    key = (name, n_envs, episode_steps, deterministic_learner)
    cached = _RUN_CACHE.get(key)
    if cached is not None:
        return cached
    o_module, o_encode_fn, o_params, o_det = _resolve_opp_spec(name, cfg, None)
    opp_act = _make_act_fn(o_module, o_encode_fn, deterministic=o_det)
    run = _make_run_fn(learner_act, opp_act, episode_steps, n_envs)
    _RUN_CACHE[key] = (run, o_params)
    return run, o_params


# Cache the fixed eval board set (host-side generation) per (n_envs, base_seed).
_BOARDS_CACHE: dict = {}


def _get_init_boards(n_envs, base_seed):
    key = (n_envs, base_seed)
    cached = _BOARDS_CACHE.get(key)
    if cached is not None:
        return cached
    board_pool = make_board_pool(n_envs, base_seed)
    indices    = jnp.arange(n_envs, dtype=jnp.int32)
    init_state = select_boards(board_pool, indices, n_envs)
    _BOARDS_CACHE[key] = init_state
    return init_state


def clear_eval_cache() -> None:
    """Drop cached compiled eval scans + board sets (e.g. if cfg / arch changes)."""
    _RUN_CACHE.clear()
    _BOARDS_CACHE.clear()


def run_eval_jax(
    learner_params,
    cfg,
    opponents,
    base_seed: int = 1000,
    *,
    deterministic_learner: bool = True,
) -> dict[str, float]:
    """Win rate of ``learner_params`` vs each JAX-arch opponent in ``opponents``.

    Plays ``cfg.eval_rounds`` games per opponent in parallel inside one jitted
    scan, learner alternating player sides (even fair split). Returns
    {opponent_name: win_rate in [0, 1]} with draws counting 0.5.

    The compiled scan is cached per opponent (see ``_RUN_CACHE``), so only the
    FIRST eval interval pays the (encoder-dominated) compile cost; later intervals
    just stream the updated learner params through the cached kernel.
    """
    opponents = list(opponents)
    if not opponents:
        return {}

    rounds        = int(cfg.eval_rounds)
    episode_steps = int(cfg.env.episode_steps)
    n_envs        = rounds

    # One shared board set for every opponent (comparable + cheap). Alternate the
    # learner's side across envs so each opponent is met equally from both seats.
    # Boards depend only on (n_envs, base_seed), constant across eval intervals, so
    # cache the host-side generation — it's the same fixed eval set every interval.
    init_state = _get_init_boards(n_envs, base_seed)
    learner_players = (jnp.arange(n_envs) % 2).astype(jnp.int32)

    learner_module    = build_policy(cfg)
    learner_encode_fn = make_env_encoder(
        cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7")
    )
    learner_act = _make_act_fn(
        learner_module, learner_encode_fn, deterministic=deterministic_learner,
    )
    learner_params = jax.tree_util.tree_map(jnp.asarray, learner_params)

    key = jax.random.PRNGKey(base_seed)

    results: dict[str, float] = {}
    for name in opponents:
        run, o_params = _get_run_and_opp_params(
            name, cfg, learner_act, n_envs, episode_steps, deterministic_learner,
        )
        # "self" opponent = the current learner; its params (None in the cache)
        # must follow the live learner each interval, not be frozen at first call.
        if o_params is None:
            o_params = learner_params
        key, subkey = jax.random.split(key)
        outcome = run(learner_params, o_params, init_state, learner_players, subkey)
        results[name] = float(jnp.mean(outcome))

    return results


def run_eval_jax_4p(
    learner_params,
    cfg,
    opponents,
    base_seed: int = 1000,
    *,
    deterministic_learner: bool = True,
) -> dict[str, float]:
    """1-vs-3 FFA win rate of ``learner_params`` vs 3 copies of each JAX-arch opponent.

    The in-training counterpart to the 2-player ``run_eval_jax``: it plays a
    4-seat free-for-all where the learner holds ONE seat and the opponent policy
    fills the other three. The scoring is byte-for-byte the same scheme as
    ``arena.panel(num_players=4)`` so in-training and post-training 4p numbers are
    directly comparable:

      * learner = spec 0, opponent = spec 1;
      * the learner's seat is balanced (``i % 4``) then shuffled with a
        ``base_seed``-seeded RNG, one assignment per round, reused across opponents;
      * win credit per game = 1.0 if the learner holds the sole rank-1 (top-reward)
        seat, 0.5 if tied for rank-1, else 0.0 — so the even baseline is 1/4.

    Returns ``{opponent_name: win_rate in [0, 1]}``. Reuses the shared K-player
    batched match (``play_match_batched``); all four seats play on-device in one
    jitted scan per opponent.
    """
    import random as _random

    import numpy as np

    opponents = list(opponents)
    if not opponents:
        return {}

    rounds        = int(cfg.eval_rounds)
    episode_steps = int(cfg.env.episode_steps)
    n_seats       = 4

    learner_spec = EvalSpec(
        build_policy(cfg),
        make_env_encoder(
            cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7")
        ),
        learner_params,
        deterministic_learner,
    )

    # Balanced + shuffled learner seat per round (identical scheme to arena.panel),
    # so positional bias averages out and per-opponent numbers stay comparable.
    eval_seats = [i % n_seats for i in range(rounds)]
    _random.Random(base_seed).shuffle(eval_seats)
    seat_assign = [[0 if s == eval_seats[e] else 1 for s in range(n_seats)]
                   for e in range(rounds)]
    eval_seats_arr = np.asarray(eval_seats)
    env_idx = np.arange(rounds)

    results: dict[str, float] = {}
    for name in opponents:
        module, encode_fn, o_params, det = _resolve_opp_spec(name, cfg, learner_params)
        opp_spec = EvalSpec(module, encode_fn, o_params, det)
        rewards, _ = play_match_batched(
            [learner_spec, opp_spec], seat_assign, rounds, base_seed, episode_steps,
        )  # rewards[rounds, n_seats] terminal per-seat (+1 win / -1 lose / 0 draw)
        learner_r = rewards[env_idx, eval_seats_arr]
        max_r     = rewards.max(axis=1)
        n_at_max  = (rewards == max_r[:, None]).sum(axis=1)
        won       = learner_r == max_r                       # learner holds a top seat
        score     = np.where(won, np.where(n_at_max == 1, 1.0, 0.5), 0.0)
        results[name] = float(score.mean())

    return results


# ===========================================================================
# General K-player batched match (2 OR 4 seats) — used by arena post-training
# evaluation AND the in-training 4p eval above (run_eval_jax_4p).
# ===========================================================================

# Cache K-player board sets, keyed by (n_envs, base_seed, num_agents).
_KBOARDS_CACHE: dict = {}


def _get_kplayer_boards(n_envs, base_seed, num_agents):
    """Stacked EnvState[n_envs] of fresh ``num_agents``-player boards (init-tick
    applied, exactly like env._init_one but with a player count)."""
    cache_key = (n_envs, base_seed, num_agents)
    cached = _KBOARDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    empty = jnp.full((N_AGENTS, MAX_ACTIONS_PA, 3), -1.0, dtype=jnp.float32)
    states = []
    for i in range(n_envs):
        s = init_jax_state(base_seed + i, num_agents=num_agents)
        s, _, _ = jax_step_jit(s, empty)   # kaggle init-tick
        states.append(s)
    boards = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
    _KBOARDS_CACHE[cache_key] = boards
    return boards


def _make_match_run_fn(act_fns, needed_pairs, n_specs, n_seats, n_envs, episode_steps):
    """Build a jitted scan that plays one full ``n_seats``-player game per env.

    ``act_fns``      : list[n_specs] of ``act(params, state, players, key)`` (from
                       _make_act_fn) — one per distinct policy.
    ``needed_pairs`` : iterable of (spec_idx, seat) that actually occur in
                       ``seat_assign`` — only those (policy, seat) actions are traced.
    Returns ``run(params_tuple, seat_assign, init_state, key) -> (rewards[N,S], steps[N])``
    where rewards/steps are latched at each env's terminal tick.
    """
    batched_step = jax.vmap(functools.partial(jax_step, episode_steps=episode_steps))
    env_ar  = jnp.arange(n_envs, dtype=jnp.int32)
    seat_ar = jnp.broadcast_to(jnp.arange(n_seats, dtype=jnp.int32)[None, :], (n_envs, n_seats))
    env_grid = jnp.broadcast_to(env_ar[:, None], (n_envs, n_seats))
    needed_pairs = list(needed_pairs)

    def step(params_tuple, seat_assign, carry, _):
        state, key, finished, latched_r, latched_steps = carry

        # Each distinct (policy, seat) computes its action across all envs; the
        # policy occupying a seat encodes egocentrically as that seat's player.
        acts = jnp.zeros((n_specs, n_seats, n_envs, MAX_ACTIONS_PA, 3), dtype=jnp.float32)
        for (j, s) in needed_pairs:
            players = jnp.full((n_envs,), s, dtype=jnp.int32)
            rows, key = act_fns[j](params_tuple[j], state, players, key)
            acts = acts.at[j, s].set(rows)

        # joint[e, seat] = action of the policy assigned to that seat in env e.
        joint_seats = acts[seat_assign, seat_ar, env_grid]      # [N, S, A, 3]
        if n_seats < N_AGENTS:
            dummy = jnp.full((n_envs, N_AGENTS - n_seats, MAX_ACTIONS_PA, 3), -1.0)
            joint = jnp.concatenate([joint_seats, dummy], axis=1)
        else:
            joint = joint_seats

        new_state, statuses, rewards = batched_step(state, joint)

        done  = statuses[:, 0].astype(jnp.bool_)               # seat 0 always active
        newly = done & (~finished)
        latched_r     = jnp.where(newly[:, None], rewards[:, :n_seats], latched_r)
        latched_steps = jnp.where(newly, new_state.step, latched_steps)
        finished = finished | done
        return (new_state, key, finished, latched_r, latched_steps), None

    @jax.jit
    def run(params_tuple, seat_assign, init_state, key):
        init = (
            init_state,
            key,
            jnp.zeros(n_envs, dtype=jnp.bool_),
            jnp.zeros((n_envs, n_seats), dtype=jnp.float32),
            jnp.zeros(n_envs, dtype=jnp.int32),
        )
        scan_step = functools.partial(step, params_tuple, seat_assign)
        (_, _, _, latched_r, latched_steps), _ = jax.lax.scan(
            scan_step, init, None, length=episode_steps
        )
        return latched_r, latched_steps

    return run


def play_match_batched(specs, seat_assign, n_envs, base_seed, episode_steps):
    """Play ``n_envs`` full games of an ``n_seats``-player match in one scan.

    ``specs``       : list[n_specs] of EvalSpec (distinct policies).
    ``seat_assign`` : int array/list [n_envs, n_seats] — spec index occupying each
                      seat per env (e.g. 1-vs-3 FFA = [[0,1,1,1]]*n_envs; 1v1 with
                      side alternation = [[0,1],[1,0],...]).
    Returns (rewards [n_envs, n_seats] float, steps [n_envs] int) as host arrays.
    rewards are the terminal per-seat rewards (+1 win / -1 lose / 0 draw).
    """
    import numpy as np

    seat_assign = jnp.asarray(seat_assign, dtype=jnp.int32)
    n_seats     = int(seat_assign.shape[1])
    n_specs     = len(specs)

    boards = _get_kplayer_boards(n_envs, base_seed, n_seats)

    act_fns = [
        _make_act_fn(sp.module, sp.encode_fn, deterministic=sp.deterministic)
        for sp in specs
    ]
    params_tuple = tuple(jax.tree_util.tree_map(jnp.asarray, sp.params) for sp in specs)

    sa_host = np.asarray(seat_assign)
    needed_pairs = sorted({(int(sa_host[e, s]), s)
                           for e in range(sa_host.shape[0])
                           for s in range(n_seats)})

    run = _make_match_run_fn(act_fns, needed_pairs, n_specs, n_seats, n_envs, episode_steps)
    key = jax.random.PRNGKey(base_seed)
    rewards, steps = run(params_tuple, seat_assign, boards, key)
    return np.asarray(rewards), np.asarray(steps)


# ---------------------------------------------------------------------------
# Spec builders for arena agents
# ---------------------------------------------------------------------------

def spec_from_checkpoint(ckpt_path, deterministic=True) -> EvalSpec:
    """Build an EvalSpec from a JAX checkpoint .pkl.

    Loaded under a pinned import of this repo's ``src_jax`` so a hijacked
    ``sys.modules['src_jax']`` (from submission opponents) can't mismatch shapes —
    same robustness as ``arena._register_jax_agent``. (Feature arch is selected
    in-tree via ``cfg.model.feat_version``; there is no external src tree to pin.)
    """
    import importlib

    from src.evaluation import _pinned_src_jax

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    with _pinned_src_jax(_REPO_ROOT):
        pol = importlib.import_module("src_jax.policies")
        fv  = importlib.import_module("src_jax.feature_versions")
        params    = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        module    = pol.build_policy(cfg)
        encode_fn = fv.make_env_encoder(
            cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7")
        )
    return EvalSpec(module, encode_fn, params, deterministic)


def spec_from_submission(name) -> EvalSpec:
    """EvalSpec for a registered JAX submission opponent (sub_vN)."""
    module, encode_fn, params, det = _submission_spec(name)
    return EvalSpec(module, encode_fn, params, det)
