"""
JAX policy → kaggle-environment callable agent.

Usage:
    from src_jax.jax_agent import make_jax_agent

    agent = make_jax_agent("sub_v86")  # short name from submission/checkpoint_mapping.json
    agent = make_jax_agent("artifacts/bc/bc_pretrain/ckpt_last.pkl")
    env.run([agent, opponent])
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from .bc_data import obs_to_env_state
from .features import MIN_SOURCE_SHIPS, _make_env_encoder, self_feature_dim, candidate_feature_dim, aim_fleets
from .feature_versions import make_env_encoder
from .physics import (
    MAX_BASE_PLANETS, MAX_PLANETS, MAX_FLEETS, N_COMET_GROUPS, COMET_PER_GROUP,
    COMET_SLOT_BASE, EnvState,
)
from .policies import build_policy
from .rollout import GLOB_DIM

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT_MAPPING_PATH = _REPO_ROOT / "submission" / "checkpoint_mapping.json"
_CHECKPOINT_MAPPING: dict[str, str] | None = None


def _load_checkpoint_mapping() -> dict[str, str]:
    global _CHECKPOINT_MAPPING
    if _CHECKPOINT_MAPPING is None:
        with open(_CHECKPOINT_MAPPING_PATH) as f:
            _CHECKPOINT_MAPPING = json.load(f)
    return _CHECKPOINT_MAPPING


def _resolve_checkpoint_path(name_or_path: str | Path) -> Path:
    """Map a short name (e.g. ``sub_v86``) or checkpoint path to an absolute path."""
    key = str(name_or_path)
    mapped = _load_checkpoint_mapping().get(key)
    if mapped is not None:
        return (_REPO_ROOT / mapped).resolve()
    path = Path(name_or_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _obs_to_dict(obs) -> dict:
    """Accept either a plain dict or a kaggle StructuredObservation object."""
    if isinstance(obs, dict):
        return obs
    try:
        return dict(obs)
    except TypeError:
        pass
    try:
        return vars(obs)
    except TypeError:
        pass
    keys = [
        "player", "step", "planets", "fleets", "initial_planets",
        "angular_velocity", "comet_planet_ids", "comets", "remainingOverageTime",
        "next_fleet_id",
    ]
    return {k: getattr(obs, k) for k in keys if hasattr(obs, k)}


def _safe_target_logits(target_logits: jax.Array) -> jax.Array:
    invalid = ~jnp.isfinite(target_logits).any(axis=-1)
    return jnp.where(invalid[:, None], jnp.zeros_like(target_logits), target_logits)


def _evac_plan(state: EnvState, player: jax.Array):
    """Comet evacuation plan (one env, no batch dim).

    An owned comet whose path is exhausted is about to leave the board next tick;
    any ships still garrisoned on it are lost permanently. For each such comet we
    dump its ENTIRE garrison to the nearest friendly planet whose lane is clear,
    and if that lane is blocked (or there is no friendly planet) to the nearest
    enemy planet instead. Lane blocking reuses ``aim_fleets`` — the same
    static + rotating-planet blocker check the encoder uses — so an evac fleet
    never gets aimed straight into the sun or a moving planet.

    Returns per-planet-slot arrays [MAX_PLANETS]:
        valid : this slot is an evacuating comet with a usable destination
        ids   : source planet id  (== state.p_id)
        angle : launch angle toward the chosen destination
        ships : ships to send (the comet's full garrison)
    """
    pl  = player[0]
    idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    # Ticks of path left on each comet slot. comet_path_idx is per-GROUP; map each
    # slot to its (group, member) and compare against that member's path length.
    gi    = jnp.clip((idx - COMET_SLOT_BASE) // COMET_PER_GROUP, 0, N_COMET_GROUPS - 1)
    sj    = jnp.clip((idx - COMET_SLOT_BASE) %  COMET_PER_GROUP, 0, COMET_PER_GROUP - 1)
    idx_g = jnp.maximum(state.comet_path_idx, jnp.int32(0))            # [N_CG]
    ticks_rem = state.comet_path_lens[gi, sj] - idx_g[gi]              # [MAX_PLANETS]

    ships     = jnp.floor(state.p_ships).astype(jnp.int32)           # send everything (floor: never exceed garrison)
    evac_is   = (
        state.p_is_comet
        & (ticks_rem <= 0)                # path exhausted -> leaving the board
        & state.p_alive
        & (state.p_owner == pl)
        & (ships >= 1)
    )                                                                 # [MAX_PLANETS]

    # Distance from every (potential source) planet to every (potential dest).
    dx   = state.p_x[:, None] - state.p_x[None, :]
    dy   = state.p_y[:, None] - state.p_y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)                          # [P, P]
    not_self    = idx[:, None] != idx[None, :]
    is_friendly = state.p_alive[None, :] & (state.p_owner[None, :] == pl) \
        & (~state.p_is_comet[None, :]) & not_self
    is_enemy    = state.p_alive[None, :] & (state.p_owner[None, :] != pl) \
        & (state.p_owner[None, :] != -1) & (~state.p_is_comet[None, :]) & not_self

    BIG = jnp.float32(1e30)
    nf  = jnp.argmin(jnp.where(is_friendly, dist, BIG), axis=1).astype(jnp.int32)
    ne  = jnp.argmin(jnp.where(is_enemy,    dist, BIG), axis=1).astype(jnp.int32)
    has_f = jnp.any(is_friendly, axis=1)
    has_e = jnp.any(is_enemy,    axis=1)

    angle_f, blk_f = aim_fleets(
        state.p_x, state.p_y, state.p_radius, state.p_alive, state.p_is_comet,
        state.angular_velocity, idx, nf, ships,
    )
    angle_e, blk_e = aim_fleets(
        state.p_x, state.p_y, state.p_radius, state.p_alive, state.p_is_comet,
        state.angular_velocity, idx, ne, ships,
    )

    use_f = has_f & (~blk_f)                       # nearest friend, lane clear
    use_e = (~use_f) & has_e & (~blk_e)            # else nearest enemy, lane clear
    valid = evac_is & (use_f | use_e)
    angle = jnp.where(use_f, angle_f, angle_e)
    return valid, state.p_id, angle, ships


def _make_dummy_state() -> "EnvState":
    """Synthetic EnvState whose shapes/dtypes mirror obs_to_env_state's output.
    Used to warm up — or AOT-lower — the encoder; module-level so the gen-time
    AOT build and runtime build_agent compile on byte-identical input avals."""
    from .physics import MAX_PATH_LEN
    return EnvState(
        p_id=jnp.zeros(MAX_PLANETS, jnp.int32),
        p_owner=jnp.full(MAX_PLANETS, -1, jnp.int32),
        p_x=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_y=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_radius=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_ships=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_production=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_alive=jnp.zeros(MAX_PLANETS, bool),
        p_init_x=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_init_y=jnp.zeros(MAX_PLANETS, jnp.float32),
        p_is_comet=jnp.zeros(MAX_PLANETS, bool),
        f_owner=jnp.zeros(MAX_FLEETS, jnp.int32),
        f_x=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_y=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_angle=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_ships=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_alive=jnp.zeros(MAX_FLEETS, bool),
        comet_paths=jnp.zeros((N_COMET_GROUPS, 4, MAX_PATH_LEN, 2), jnp.float32),
        comet_path_lens=jnp.zeros((N_COMET_GROUPS, 4), jnp.int32),
        comet_ships=jnp.zeros(N_COMET_GROUPS, jnp.float32),
        comet_path_idx=jnp.full(N_COMET_GROUPS, -2, jnp.int32),
        step=jnp.int32(0),
        angular_velocity=jnp.float32(0.0),
        num_agents=jnp.int32(2),
        comet_id_base=jnp.int32(0),
    )


def build_agent(
    params,
    cfg,
    deterministic: bool = True,
    aot_encode=None,
) -> Callable:
    """Return an obs→moves callable from in-memory params.

    ``aot_encode`` is an optional ``(serialized, in_tree, out_tree)`` triple from
    ``jax.experimental.serialize_executable`` holding a PRE-COMPILED encoder
    executable (produced at submission-gen time). The encoder is the ~5s trace+
    compile that dominates the first act; loading it instead of compiling matters
    on Kaggle 4-player games, where FOUR agent instances compile at once and
    saturate the CPU, ~doubling each compile and pushing the first act past the
    ~62s overage bank (TIMEOUT). Loading barely contends, so it dodges that.
    Version-locked to jaxlib: a mismatch makes deserialize raise → we fall back
    to the freshly-traced encoder (no worse than today)."""
    params  = jax.tree_util.tree_map(jnp.asarray, params)
    rng_key = jax.random.PRNGKey(int(getattr(cfg, "seed", 0)))

    K = cfg.env.candidate_count

    module    = build_policy(cfg)
    encode_fn = make_env_encoder(K, cfg.env, getattr(cfg.model, "feat_version", "v7"))
    if aot_encode is not None:
        import sys as _sys
        try:
            from jax.experimental import serialize_executable as _se
            _enc_exe = _se.deserialize_and_load(*aot_encode)
            def encode_fn(batched, player, _exe=_enc_exe):   # noqa: replaces the jitted encoder
                return _exe(jax.tree_util.tree_map(jnp.asarray, batched), jnp.asarray(player))
            # Logged to the (downloadable) agent stderr so the episode log shows
            # the runtime jax version AND whether AOT engaged or fell back.
            print(f"[agent] AOT encoder LOADED (jax {jax.__version__})", file=_sys.__stderr__)
        except Exception as _e:
            # Most commonly a jaxlib mismatch (e.g. a 0.5.1-built blob loaded under
            # a different local jaxlib for a quick test) — just compile the encoder.
            # Clean one-liner, NOT a traceback, so it doesn't look like a crash.
            # encode_fn stays the freshly-traced encoder, so the agent is correct.
            print(f"[agent] AOT encoder unavailable ({type(_e).__name__}); "
                  f"compiling instead (jax {jax.__version__})", file=_sys.__stderr__)

    MAX_SRC = MAX_BASE_PLANETS

    @jax.jit
    def _forward(sf, cf, gf, cm, sv):
        return module.apply(params, sf, cf, gf, cm, sv)

    # _evac (comet-garrison evacuation) intentionally removed from the hot path —
    # its compile (~1.4s, ~7s on Kaggle CPU) was a meaningful slice of the first-act
    # budget and the salvage behavior is negligible at this level. _evac_plan stays
    # defined above for easy re-enable.

    # Player count is not in the obs, so infer it from seat ownership: 2-player
    # boards seat owners {0,1}, 4-player (FFA) boards {0,1,2,3}. Use the static
    # initial_planets layout and keep a running max across the episode so an
    # eliminated seat never downgrades a 4p game to 2p mid-game. num_agents only
    # drives the encoder's is_4player global feature; the policy was trained with
    # is_4player=1 in FFA, so feeding the constant 2 here makes it misread every
    # FFA board as a 1v1 and collapse to no-ops.
    seen_max_owner = 0

    def agent(obs) -> list:
        nonlocal rng_key, seen_max_owner

        obs_dict = _obs_to_dict(obs)
        for src in (obs_dict.get("initial_planets"), obs_dict.get("planets")):
            for r in (src or []):
                o = int(r[1])
                if o > seen_max_owner:
                    seen_max_owner = o
        num_agents = 4 if seen_max_owner >= 2 else 2
        state  = obs_to_env_state(obs_dict, num_agents)
        player = jnp.array([int(obs_dict.get("player", 0))], dtype=jnp.int32)

        batched = jax.tree_util.tree_map(lambda x: x[None], state)
        enc     = encode_fn(batched, player)

        sf = enc.self_feats[0]               # [MAX_SRC, SELF_DIM]
        cf = enc.cand_feats[0]               # [MAX_SRC, K, CAND_DIM]
        gf = jnp.broadcast_to(enc.global_feat[0][None, :], (MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask[0]                # [MAX_SRC, K]
        sv = enc.src_valid[0]

        out = _forward(sf, cf, gf, cm, sv)

        target_logits = _safe_target_logits(out.target_logits)
        if deterministic:
            ti = jnp.argmax(target_logits, axis=-1)
        else:
            rng_key, subkey = jax.random.split(rng_key)
            ti = jax.random.categorical(subkey, target_logits)

        row_idx = jnp.arange(MAX_SRC)

        src_valid = enc.src_valid[0]
        src_ids   = enc.src_ids[0]
        src_ships = enc.src_ships[0]
        cand_mask = enc.cand_mask[0]
        angles    = enc.angles[0]            # [MAX_SRC, K] — exact, precomputed by encoder

        ships_send      = jnp.maximum(MIN_SOURCE_SHIPS, src_ships)
        selected_angles = angles[row_idx, ti]

        # NOTE: the comet-evacuation override (dump a leaving comet's garrison to a
        # nearby planet) was removed — its _evac JIT cost ~1.4s to compile (~7s on
        # Kaggle's CPU), enough to matter against the ~60s first-act overage bank
        # (see the timeout analysis). _evac_plan/_evac are kept defined but unused
        # so re-enabling is a small change if the salvage behavior is ever wanted.
        moves: list = []
        for i in range(MAX_SRC):
            if not bool(src_valid[i]):
                continue
            t = int(ti[i])
            if t == 0:
                continue
            if not bool(cand_mask[i, t]):
                continue
            moves.append([int(src_ids[i]), float(selected_angles[i]), int(ships_send[i])])

        return moves

    # Warm up the JITs so the first in-game call doesn't blow the actTimeout.
    # With an AOT encoder the encode warmup is just a fast run; otherwise it
    # triggers the encoder trace+compile here.
    _dummy_state = _make_dummy_state()
    _dummy_player = jnp.array([0], dtype=jnp.int32)
    _dummy_batched = jax.tree_util.tree_map(lambda x: x[None], _dummy_state)
    _dummy_enc = encode_fn(_dummy_batched, _dummy_player)
    _dummy_sf = _dummy_enc.self_feats[0]
    _dummy_cf = _dummy_enc.cand_feats[0]
    _dummy_gf = jnp.broadcast_to(_dummy_enc.global_feat[0][None, :], (MAX_SRC, GLOB_DIM))
    _dummy_cm = _dummy_enc.cand_mask[0]
    _dummy_sv = _dummy_enc.src_valid[0]
    jax.block_until_ready(_forward(_dummy_sf, _dummy_cf, _dummy_gf, _dummy_cm, _dummy_sv))

    return agent


def make_jax_agent(
    checkpoint_path: str | Path,
    deterministic: bool = True,
) -> Callable:
    """Return an obs->moves callable that runs a policy checkpoint.

    ``checkpoint_path`` may be a filesystem path or a short name listed in
    ``submission/checkpoint_mapping.json`` (e.g. ``"sub_v86"``).

    This is the public checkpoint-loading wrapper used by notebooks and arena
    scripts. The lower-level ``build_agent`` is useful when params/config are
    already loaded in memory.
    """
    resolved = _resolve_checkpoint_path(checkpoint_path)
    with open(resolved, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    return build_agent(params, ckpt["config"], deterministic=deterministic)
