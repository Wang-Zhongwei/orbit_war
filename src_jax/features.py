"""
JAX-accelerated feature encoding, operating directly on JaxState.

Replaces the Python-loop encode_turn for the training hot path.

Key differences from features.py:
  - No observation dict or GameState conversion — reads JaxState arrays directly.
  - find_angle via JAX vmap over all (src, tgt) pairs simultaneously.
  - Blocking check: sun collision + static planet positions (no per-tick moving blocker).
  - Comet intercept: falls back to direct angle (no waypoint extrapolation).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .jax_physics import (
    JaxState, MAX_PLANETS, MAX_BASE_PLANETS,
    _fleet_speed,
)
from .constants import CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS

_MAX_TICKS: int = 25


# ---------------------------------------------------------------------------
# Core JAX kernel: find angle + blocking for a single (src, tgt) pair
# ---------------------------------------------------------------------------

def _find_angle_single(
    src_x: jnp.ndarray,         # scalar
    src_y: jnp.ndarray,
    src_radius: jnp.ndarray,
    src_slot: jnp.ndarray,      # int32 — to exclude from blockers
    tgt_x: jnp.ndarray,         # scalar (current position)
    tgt_y: jnp.ndarray,
    tgt_radius: jnp.ndarray,
    tgt_init_x: jnp.ndarray,    # initial position, used for rotation
    tgt_init_y: jnp.ndarray,
    tgt_is_comet: jnp.ndarray,  # bool
    ships: jnp.ndarray,         # scalar — for speed
    angular_velocity: jnp.ndarray,
    all_p_x: jnp.ndarray,       # [MAX_PLANETS] — for blocker check
    all_p_y: jnp.ndarray,
    all_p_r: jnp.ndarray,
    all_p_alive: jnp.ndarray,
    tgt_slot: jnp.ndarray,      # int32 — to exclude from blockers
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (angle float32, is_blocked bool) for this (src, tgt) pair."""
    spd = _fleet_speed(ships)

    # --- Rotating target: binary-search intercept via fixed scan ---
    dx0 = tgt_init_x - CENTER
    dy0 = tgt_init_y - CENTER
    orb_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    rotating = (orb_r + tgt_radius < ROTATION_RADIUS_LIMIT) & ~tgt_is_comet
    init_ang = jnp.arctan2(dy0, dx0)

    def scan_body(carry, tick):
        best_angle, found = carry
        t = tick.astype(jnp.float32)
        ang_t = init_ang + angular_velocity * t
        tx = CENTER + orb_r * jnp.cos(ang_t)
        ty = CENTER + orb_r * jnp.sin(ang_t)
        dx = tx - src_x
        dy = ty - src_y
        gap = spd * t - (jnp.sqrt(dx * dx + dy * dy) - src_radius)
        valid = (~found) & (gap > -tgt_radius)
        angle_t = jnp.arctan2(dy, dx)
        return (jnp.where(valid, angle_t, best_angle), found | valid), None

    (rot_angle, rot_found), _ = jax.lax.scan(
        scan_body,
        (jnp.float32(0.0), jnp.bool_(False)),
        jnp.arange(1, _MAX_TICKS + 1, dtype=jnp.int32),
    )

    # --- Static / fallback angle (also used for comets) ---
    dx_s = tgt_x - src_x
    dy_s = tgt_y - src_y
    static_angle = jnp.arctan2(dy_s, dx_s)

    # Use static angle when: not rotating, OR rotating but no intercept found
    no_intercept = rotating & ~rot_found
    angle = jnp.where(rotating & rot_found, rot_angle, static_angle)

    # --- Blocking: segment from src centre to current target position ---
    seg_dx = tgt_x - src_x
    seg_dy = tgt_y - src_y
    seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy + 1e-30

    # Sun: min-distance from (CENTER, CENTER) to segment
    t_sun = jnp.clip(
        ((CENTER - src_x) * seg_dx + (CENTER - src_y) * seg_dy) / seg_l2, 0.0, 1.0
    )
    proj_x = src_x + t_sun * seg_dx
    proj_y = src_y + t_sun * seg_dy
    sun_blocked = (proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2 < SUN_RADIUS ** 2

    # Planets: vectorised distance from each alive blocker centre to the segment
    planet_slots = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    blocker_mask = all_p_alive & (planet_slots != src_slot) & (planet_slots != tgt_slot)
    t_b = jnp.clip(
        ((all_p_x - src_x) * seg_dx + (all_p_y - src_y) * seg_dy) / seg_l2, 0.0, 1.0
    )
    proj_bx = src_x + t_b * seg_dx
    proj_by = src_y + t_b * seg_dy
    dist_sq = (all_p_x - proj_bx) ** 2 + (all_p_y - proj_by) ** 2
    planet_blocked = jnp.any(blocker_mask & (dist_sq <= all_p_r ** 2))

    is_blocked = sun_blocked | planet_blocked | no_intercept
    return angle, is_blocked


# ===========================================================================
# Batched encoder — processes N envs in one JIT+vmap call
# ===========================================================================

class _EncodedTurn(NamedTuple):
    """Fixed-size encoded observation for one env (no batch dim).
    After vmap the leading axis is N (number of envs)."""
    src_valid:       jnp.ndarray  # [MAX_SRC]       bool
    src_ids:         jnp.ndarray  # [MAX_SRC]       int32
    src_ships:       jnp.ndarray  # [MAX_SRC]       int32
    self_feats:      jnp.ndarray  # [MAX_SRC, 11]
    cand_slots_safe: jnp.ndarray  # [MAX_SRC, K]    int32 (no-op clamped to 0)
    cand_ids:        jnp.ndarray  # [MAX_SRC, K]    int32 (-1 for no-op/invalid)
    cand_mask:       jnp.ndarray  # [MAX_SRC, K]    bool
    ship_counts:     jnp.ndarray  # [MAX_SRC, K]    int32
    angles:          jnp.ndarray  # [MAX_SRC, K]    float32
    cand_feats:      jnp.ndarray  # [MAX_SRC, K, 10]
    global_feat:     jnp.ndarray  # [8]
    step:            jnp.ndarray  # []              int32


def _make_env_encoder(K: int, env_cfg):
    """Return a JIT+vmap encoder: (batched_state, players[N int32]) -> _EncodedTurn[N,...].

    Call once at JaxVecEnv init; reuse every tick.
    """
    from .obs_schema import MIN_SOURCE_SHIPS
    from .constants import MAX_ANGULAR_VELOCITY

    K_e     = K // 3
    K_n     = K // 3
    K_f_eff = K - 1 - K_e - K_n   # friendly quota; total non-noop slots = K-1

    bs  = jnp.float32(env_cfg.board_size)
    ms  = jnp.float32(env_cfg.max_ships)
    mp  = jnp.float32(env_cfg.max_production)
    mp_ = jnp.float32(env_cfg.max_planets)
    es  = jnp.float32(env_cfg.episode_steps)
    max_av = jnp.float32(MAX_ANGULAR_VELOCITY)

    def _encode_single(state: JaxState, player: jnp.ndarray) -> _EncodedTurn:
        planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
        INF = jnp.float32(1e30)

        # ── Source selection ──────────────────────────────────────────────
        src_mask = (
            state.p_alive
            & (state.p_owner == player)
            & (state.p_ships > MIN_SOURCE_SHIPS)
        )
        sort_key  = jnp.where(src_mask,
                              jnp.arange(MAX_PLANETS, dtype=jnp.int32),
                              MAX_PLANETS + 1)
        src_slots = jnp.argsort(sort_key)[:MAX_BASE_PLANETS]   # [MAX_SRC]
        src_valid = src_mask[src_slots]                         # [MAX_SRC]

        # ── Candidate selection (vmapped over sources) ────────────────────
        def _select_cands(src_slot):
            sx_   = state.p_x[src_slot]
            sy_   = state.p_y[src_slot]
            dist_ = jnp.hypot(state.p_x - sx_, state.p_y - sy_)
            excl_ = state.p_alive & (planet_idx != src_slot)

            def top_k_(mask, k):
                d_     = jnp.where(mask & excl_, dist_, INF)
                slots_ = jnp.argsort(d_)[:k]
                valid_ = dist_[slots_] < INF
                return slots_, valid_

            is_enem_ = excl_ & (state.p_owner != -1) & (state.p_owner != player)
            is_neut_ = excl_ & (state.p_owner == -1)
            is_frd_  = excl_ & (state.p_owner == player)

            e_s, e_v = top_k_(is_enem_, K_e)
            n_s, n_v = top_k_(is_neut_, K_n)
            f_s, f_v = top_k_(is_frd_,  K_f_eff)

            cands_  = jnp.concatenate([jnp.array([-1], jnp.int32), e_s, n_s, f_s])
            valids_ = jnp.concatenate([jnp.array([True], bool),    e_v, n_v, f_v])
            return cands_, valids_

        all_cand_slots, all_cand_valid = jax.vmap(_select_cands)(src_slots)
        # [MAX_SRC, K], [MAX_SRC, K]
        safe_slots = jnp.clip(all_cand_slots, 0, MAX_PLANETS - 1)  # [MAX_SRC, K]

        # ── Find angles (flat vmap over MAX_SRC*K pairs) ──────────────────
        src_rep      = jnp.repeat(src_slots[:, None], K, axis=1).reshape(-1)  # [MAX_SRC*K]
        tgt_flat     = safe_slots.reshape(-1)                                  # [MAX_SRC*K]
        tgt_ships_2d = state.p_ships[safe_slots]                               # [MAX_SRC, K]
        ships_flat   = jnp.maximum(tgt_ships_2d + 1.0, 13.0).reshape(-1)      # [MAX_SRC*K]

        def _angle_pair(ss, ts, sh):
            return _find_angle_single(
                state.p_x[ss], state.p_y[ss], state.p_radius[ss], ss,
                state.p_x[ts], state.p_y[ts], state.p_radius[ts],
                state.p_init_x[ts], state.p_init_y[ts], state.p_is_comet[ts],
                sh, state.angular_velocity,
                state.p_x, state.p_y, state.p_radius, state.p_alive,
                ts,
            )

        angles_flat, blocked_flat = jax.vmap(_angle_pair)(src_rep, tgt_flat, ships_flat)
        angles  = angles_flat.reshape(MAX_BASE_PLANETS, K)    # [MAX_SRC, K]
        blocked = blocked_flat.reshape(MAX_BASE_PLANETS, K)   # [MAX_SRC, K]

        # ── Ship counts + candidate mask ──────────────────────────────────
        tgt_ships_i = jnp.round(tgt_ships_2d).astype(jnp.int32)
        ship_counts = jnp.maximum(tgt_ships_i + 1, 13)                     # [MAX_SRC, K]
        ship_counts = ship_counts.at[:, 0].set(0)                          # no-op slot

        src_ships = jnp.round(state.p_ships[src_slots]).astype(jnp.int32)  # [MAX_SRC]
        enough    = src_ships[:, None] >= ship_counts                       # [MAX_SRC, K]
        cand_mask = (all_cand_valid & ~blocked & enough).at[:, 0].set(True) # no-op always valid

        # ── Candidate IDs ─────────────────────────────────────────────────
        cand_ids = jnp.where(
            all_cand_slots >= 0,
            state.p_id[safe_slots],
            jnp.int32(-1),
        )  # [MAX_SRC, K]

        # ── Global features ───────────────────────────────────────────────
        my_p_  = state.p_alive & (state.p_owner == player)
        en_p_  = state.p_alive & (state.p_owner != -1) & (state.p_owner != player)
        ne_p_  = state.p_alive & (state.p_owner == -1)
        my_f_  = state.f_alive & (state.f_owner == player)
        en_f_  = state.f_alive & (state.f_owner != player)

        n_my = my_p_.sum().astype(jnp.float32)
        n_en = en_p_.sum().astype(jnp.float32)
        n_ne = ne_p_.sum().astype(jnp.float32)
        my_ps  = (state.p_ships * my_p_).sum()
        en_ps  = (state.p_ships * en_p_).sum()
        my_fs  = (state.f_ships * my_f_).sum()
        en_fs  = (state.f_ships * en_f_).sum()
        norm_  = mp_ * ms

        global_feat = jnp.stack([
            state.step / es,
            n_my / mp_,
            n_en / mp_,
            n_ne / mp_,
            my_ps / norm_,
            en_ps / norm_,
            my_fs / norm_,
            en_fs / norm_,
        ]).astype(jnp.float32)

        # ── Self features  [MAX_SRC, 11] ──────────────────────────────────
        sx_  = state.p_x[src_slots]
        sy_  = state.p_y[src_slots]
        ix_  = state.p_init_x[src_slots]
        iy_  = state.p_init_y[src_slots]
        orb_ = jnp.hypot(ix_ - CENTER, iy_ - CENTER)
        rot_ = (orb_ + state.p_radius[src_slots]) < ROTATION_RADIUS_LIMIT
        av_s = jnp.where(rot_, state.angular_velocity / max_av, jnp.float32(0.0))

        per_src = jnp.stack([
            jnp.ones(MAX_BASE_PLANETS, jnp.float32),
            sx_ / bs,
            sy_ / bs,
            state.p_radius[src_slots] / 5.0,
            jnp.clip(state.p_ships[src_slots], 0.0, ms) / ms,
            state.p_production[src_slots] / mp,
            av_s,
        ], axis=1)  # [MAX_SRC, 7]

        stats_row = jnp.array([
            n_my / mp_, n_en / mp_,
            my_ps / norm_, en_ps / norm_,
        ], jnp.float32)
        self_feats = jnp.concatenate(
            [per_src, jnp.broadcast_to(stats_row[None, :], (MAX_BASE_PLANETS, 4))],
            axis=1,
        )  # [MAX_SRC, 11]

        # ── Candidate features  [MAX_SRC, K, 10] ─────────────────────────
        tx_   = state.p_x[safe_slots]         # [MAX_SRC, K]
        ty_   = state.p_y[safe_slots]
        dx_   = tx_ - sx_[:, None]
        dy_   = ty_ - sy_[:, None]
        dist_ = jnp.hypot(dx_, dy_)

        t_own_  = state.p_owner[safe_slots]
        is_ne_c = (t_own_ == -1).astype(jnp.float32)
        is_fr_c = (t_own_ == player).astype(jnp.float32)
        is_en_c = ((t_own_ != -1) & (t_own_ != player)).astype(jnp.float32)

        tix_ = state.p_init_x[safe_slots]
        tiy_ = state.p_init_y[safe_slots]
        torb_ = jnp.hypot(tix_ - CENTER, tiy_ - CENTER)
        trot_ = (torb_ + state.p_radius[safe_slots]) < ROTATION_RADIUS_LIMIT
        av_t  = jnp.where(trot_, state.angular_velocity / max_av, jnp.float32(0.0))

        cand_feats = jnp.stack([
            jnp.ones((MAX_BASE_PLANETS, K), jnp.float32),
            is_ne_c, is_fr_c, is_en_c,
            dx_ / bs, dy_ / bs, dist_ / bs,
            jnp.clip(tgt_ships_2d, 0.0, ms) / ms,
            state.p_production[safe_slots] / mp,
            av_t,
        ], axis=2).at[:, 0, :].set(0.0)  # [MAX_SRC, K, 10]; no-op slot = zeros

        return _EncodedTurn(
            src_valid       = src_valid,
            src_ids         = state.p_id[src_slots],
            src_ships       = src_ships,
            self_feats      = self_feats,
            cand_slots_safe = safe_slots,
            cand_ids        = cand_ids,
            cand_mask       = cand_mask,
            ship_counts     = ship_counts,
            angles          = angles,
            cand_feats      = cand_feats,
            global_feat     = global_feat,
            step            = state.step,
        )

    return jax.jit(jax.vmap(_encode_single))
