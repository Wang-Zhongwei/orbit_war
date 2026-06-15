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

import jax
import jax.numpy as jnp

from .physics import (
    jax_step,
    MAX_BASE_PLANETS,
    MAX_ACTIONS_PA,
    N_AGENTS,
)
from .features import MIN_SOURCE_SHIPS, _make_env_encoder
from .policies import build_policy
from .rollout import _build_actions_single
from .env import make_board_pool, select_boards

MAX_SRC = MAX_BASE_PLANETS

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
    encode_fn = _make_env_encoder(cfg.env.candidate_count, cfg.env)
    return module, encode_fn, params, det


def _submission_spec(name):
    """(module, encode_fn, params, deterministic) for a JAX submission opponent.

    A submission's policy lives in a specific src_jax tree (e.g. the v7arch
    worktree: 30-dim cand_proj, horizons 60/45), recorded in
    ``src.evaluation._PPO_SUBMISSIONS`` as a root label. We rebuild its module +
    encoder from *that* tree (pinned the same way src.evaluation loads it) so the
    param tree matches, then run them on the *shared* (byte-identical) EnvState
    inside the eval scan. The only requirement is that the chosen tree's
    physics/EnvState match this tree's — which holds for every registered root.
    """
    import importlib

    from src.evaluation import (
        _pinned_src_jax, _SRC_JAX_ROOTS, _PPO_SUBMISSIONS, _ppo_sub_mods,
    )

    if name not in _ppo_sub_mods:
        raise KeyError(
            f"JAX submission opponent {name!r} did not load (see src.evaluation "
            f"warnings); cannot run on the JAX eval fast-path."
        )
    root_label = _PPO_SUBMISSIONS[name][1]
    root       = _SRC_JAX_ROOTS[root_label]
    ckpt       = _ppo_sub_mods[name]._ckpt   # {"params": ..., "config": TrainConfig}

    with _pinned_src_jax(root):
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
    learner_encode_fn = _make_env_encoder(cfg.env.candidate_count, cfg.env)
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
