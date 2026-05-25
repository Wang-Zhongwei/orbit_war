"""
JAX-accelerated feature encoding, operating directly on EnvState.

Encodes observations for all envs in one JIT+vmap call:
  - Reads EnvState arrays directly (no dict conversion).
  - find_angle via JAX vmap over all (src, tgt) pairs simultaneously.
  - Blocking check: sun collision + planet positions.
  - Comet intercept: falls back to direct angle (no waypoint extrapolation).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import (
    EnvState, MAX_PLANETS, MAX_BASE_PLANETS, MAX_FLEETS,
    _fleet_speed,
)
from .constants import CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS

MIN_SOURCE_SHIPS: int = 13
_MAX_TICKS: int = 25


def self_feature_dim() -> int:
    return 11


def candidate_feature_dim() -> int:
    return 10


def global_feature_dim() -> int:
    return 8


def fleet_feature_dim() -> int:
    return 6


def fleet_count() -> int:
    return MAX_FLEETS


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
    # Use current position for init_ang so ang_t = init_ang + av*t predicts
    # where the planet will be t ticks from NOW, not t ticks from game start.
    dx0 = tgt_x - CENTER
    dy0 = tgt_y - CENTER
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
    src_slots:       jnp.ndarray  # [MAX_SRC]       int32 — planet slot for each source
    cand_slots_safe: jnp.ndarray  # [MAX_SRC, K]    int32 (no-op clamped to 0)
    cand_ids:        jnp.ndarray  # [MAX_SRC, K]    int32 (-1 for no-op/invalid)
    cand_mask:       jnp.ndarray  # [MAX_SRC, K]    bool
    ship_counts:     jnp.ndarray  # [MAX_SRC, K]    int32
    angles:          jnp.ndarray  # [MAX_SRC, K]    float32
    cand_feats:      jnp.ndarray  # [MAX_SRC, K, 10]
    global_feat:     jnp.ndarray  # [8]
    fleet_feats:     jnp.ndarray  # [MAX_FLEETS, 6]
    fleet_mask:      jnp.ndarray  # [MAX_FLEETS]    bool
    step:            jnp.ndarray  # []              int32


def _make_env_encoder(K: int, env_cfg):
    """Return a JIT+vmap encoder: (batched_state, players[N int32]) -> _EncodedTurn[N,...].

    Call once at JaxVecEnv init; reuse every tick.
    """
    from .constants import MAX_ANGULAR_VELOCITY

    K_enemy    = K // 3
    K_neutral  = K // 3
    K_friendly = K - 1 - K_enemy - K_neutral   # friendly quota; total non-noop slots = K-1

    board_size           = jnp.float32(env_cfg.board_size)
    max_ships            = jnp.float32(env_cfg.max_ships)
    max_production       = jnp.float32(env_cfg.max_production)
    max_planets          = jnp.float32(env_cfg.max_planets)
    episode_steps        = jnp.float32(env_cfg.episode_steps)
    max_angular_velocity = jnp.float32(MAX_ANGULAR_VELOCITY)

    def _encode_single(state: EnvState, player: jnp.ndarray) -> _EncodedTurn:
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
            src_x       = state.p_x[src_slot]
            src_y       = state.p_y[src_slot]
            dist_to_src = jnp.hypot(state.p_x - src_x, state.p_y - src_y)
            # Exclude comets: the static-angle fallback aims at the comet's
            # current position, which it will have left by arrival time, so
            # fleets typically miss and may continue into the sun.
            excl_mask   = state.p_alive & (planet_idx != src_slot) & ~state.p_is_comet

            def top_k(mask, k):
                sorted_dist = jnp.where(mask & excl_mask, dist_to_src, INF)
                slots       = jnp.argsort(sorted_dist)[:k]
                valid       = dist_to_src[slots] < INF
                return slots, valid

            is_enemy_planet   = excl_mask & (state.p_owner != -1) & (state.p_owner != player)
            is_neutral_planet = excl_mask & (state.p_owner == -1)
            is_friendly_planet = excl_mask & (state.p_owner == player)

            enemy_slots,   enemy_valid   = top_k(is_enemy_planet,   K_enemy)
            neutral_slots, neutral_valid = top_k(is_neutral_planet, K_neutral)
            friendly_slots, friendly_valid = top_k(is_friendly_planet, K_friendly)

            cand_slots = jnp.concatenate([jnp.array([-1], jnp.int32), enemy_slots, neutral_slots, friendly_slots])
            cand_valid = jnp.concatenate([jnp.array([True], bool),     enemy_valid, neutral_valid, friendly_valid])
            return cand_slots, cand_valid

        all_cand_slots, all_cand_valid = jax.vmap(_select_cands)(src_slots)
        # [MAX_SRC, K], [MAX_SRC, K]
        safe_slots = jnp.clip(all_cand_slots, 0, MAX_PLANETS - 1)  # [MAX_SRC, K]

        # ── Find angles (flat vmap over MAX_SRC*K pairs) ──────────────────
        src_slots_flat   = jnp.repeat(src_slots[:, None], K, axis=1).reshape(-1)  # [MAX_SRC*K]
        tgt_slots_flat   = safe_slots.reshape(-1)                                  # [MAX_SRC*K]
        tgt_ships_2d     = state.p_ships[safe_slots]                               # [MAX_SRC, K]
        launch_ships_flat = jnp.maximum(tgt_ships_2d + 1.0, 13.0).reshape(-1)     # [MAX_SRC*K]

        def _angle_pair(src_slot, tgt_slot, launch_ships):
            return _find_angle_single(
                state.p_x[src_slot], state.p_y[src_slot], state.p_radius[src_slot], src_slot,
                state.p_x[tgt_slot], state.p_y[tgt_slot], state.p_radius[tgt_slot], state.p_is_comet[tgt_slot],
                launch_ships, state.angular_velocity,
                state.p_x, state.p_y, state.p_radius, state.p_alive,
                tgt_slot,
            )

        angles_flat, blocked_flat = jax.vmap(_angle_pair)(src_slots_flat, tgt_slots_flat, launch_ships_flat)
        angles  = angles_flat.reshape(MAX_BASE_PLANETS, K)    # [MAX_SRC, K]
        blocked = blocked_flat.reshape(MAX_BASE_PLANETS, K)   # [MAX_SRC, K]

        # ── Ship counts + candidate mask ──────────────────────────────────
        tgt_ships_int = jnp.round(tgt_ships_2d).astype(jnp.int32)
        ship_counts   = jnp.maximum(tgt_ships_int + 1, 13)                    # [MAX_SRC, K]
        ship_counts   = ship_counts.at[:, 0].set(0)                           # no-op slot

        src_ships        = jnp.round(state.p_ships[src_slots]).astype(jnp.int32)  # [MAX_SRC]
        has_enough_ships = src_ships[:, None] >= ship_counts                       # [MAX_SRC, K]
        cand_mask = (all_cand_valid & ~blocked & has_enough_ships).at[:, 0].set(True)  # no-op always valid

        # ── Candidate IDs ─────────────────────────────────────────────────
        cand_ids = jnp.where(
            all_cand_slots >= 0,
            state.p_id[safe_slots],
            jnp.int32(-1),
        )  # [MAX_SRC, K]

        # ── Global features  [8] ─────────────────────────────────────────
        my_planet_mask      = state.p_alive & (state.p_owner == player)
        enemy_planet_mask   = state.p_alive & (state.p_owner != -1) & (state.p_owner != player)
        neutral_planet_mask = state.p_alive & (state.p_owner == -1)
        my_fleet_mask       = state.f_alive & (state.f_owner == player)
        enemy_fleet_mask    = state.f_alive & (state.f_owner != player)

        n_my_planets   = my_planet_mask.sum().astype(jnp.float32)
        n_enemy_planets = enemy_planet_mask.sum().astype(jnp.float32)
        n_neutral_planets = neutral_planet_mask.sum().astype(jnp.float32)
        my_planet_ships   = (state.p_ships * my_planet_mask).sum()
        enemy_planet_ships = (state.p_ships * enemy_planet_mask).sum()
        my_fleet_ships    = (state.f_ships * my_fleet_mask).sum()
        enemy_fleet_ships = (state.f_ships * enemy_fleet_mask).sum()
        ship_norm         = max_planets * max_ships

        global_feat = jnp.stack([
            state.step / episode_steps,           # [0] game-time fraction — late-game urgency signal
            n_my_planets / max_planets,           # [1] friendly planet fraction — macro board control
            n_enemy_planets / max_planets,        # [2] enemy planet fraction
            n_neutral_planets / max_planets,      # [3] unclaimed planets remaining — expansion opportunity
            my_planet_ships / ship_norm,          # [4] total ships sitting on friendly planets
            enemy_planet_ships / ship_norm,       # [5] total ships sitting on enemy planets
            my_fleet_ships / ship_norm,           # [6] friendly ships currently in flight
            enemy_fleet_ships / ship_norm,        # [7] enemy ships currently in flight — inbound threat
        ]).astype(jnp.float32)

        # ── Self features  [MAX_SRC, 11] ──────────────────────────────────
        src_x         = state.p_x[src_slots]
        src_y         = state.p_y[src_slots]
        src_init_x    = state.p_init_x[src_slots]
        src_init_y    = state.p_init_y[src_slots]
        src_orb_radius = jnp.hypot(src_init_x - CENTER, src_init_y - CENTER)
        src_rotating  = (src_orb_radius + state.p_radius[src_slots]) < ROTATION_RADIUS_LIMIT
        src_angular_vel = jnp.where(src_rotating, state.angular_velocity / max_angular_velocity, jnp.float32(0.0))

        src_per_planet_feats = jnp.stack([
            jnp.ones(MAX_BASE_PLANETS, jnp.float32),                              # [0] bias term
            src_x / board_size,                                                    # [1] normalized board-x position
            src_y / board_size,                                                    # [2] normalized board-y position
            state.p_radius[src_slots] / 5.0,                                      # [3] planet size (larger = more ships to capture)
            jnp.clip(state.p_ships[src_slots], 0.0, max_ships) / max_ships,      # [4] current garrison — ships available to send
            state.p_production[src_slots] / max_production,                       # [5] ships-per-tick income — high-production planets are worth defending
            src_angular_vel,                                                       # [6] orbital speed (0 for static outer planets); affects fleet intercept geometry
        ], axis=1)  # [MAX_SRC, 7]

        src_global_stats = jnp.array([
            n_my_planets / max_planets,     # [7] friendly planet count — proxy for board control
            n_enemy_planets / max_planets,  # [8] enemy planet count — pressure indicator
            my_planet_ships / ship_norm,    # [9] total friendly fleet strength — overall offensive/defensive capacity
            enemy_planet_ships / ship_norm, # [10] total enemy fleet strength — threat level
        ], jnp.float32)
        self_feats = jnp.concatenate(
            [src_per_planet_feats, jnp.broadcast_to(src_global_stats[None, :], (MAX_BASE_PLANETS, 4))],
            axis=1,
        )  # [MAX_SRC, 11]

        # ── Candidate features  [MAX_SRC, K, 10] ─────────────────────────
        # NOTE: 10 dims vs 15 in src/features.py — fleet arrival signals omitted
        # because per-fleet vmap over variable-length lists is incompatible with JAX tracing.
        tgt_x   = state.p_x[safe_slots]         # [MAX_SRC, K]
        tgt_y   = state.p_y[safe_slots]
        delta_x = tgt_x - src_x[:, None]
        delta_y = tgt_y - src_y[:, None]
        tgt_dist = jnp.hypot(delta_x, delta_y)

        tgt_owner  = state.p_owner[safe_slots]
        is_neutral = (tgt_owner == -1).astype(jnp.float32)
        is_friendly = (tgt_owner == player).astype(jnp.float32)
        is_enemy   = ((tgt_owner != -1) & (tgt_owner != player)).astype(jnp.float32)

        tgt_init_x    = state.p_init_x[safe_slots]
        tgt_init_y    = state.p_init_y[safe_slots]
        tgt_orb_radius = jnp.hypot(tgt_init_x - CENTER, tgt_init_y - CENTER)
        tgt_rotating  = (tgt_orb_radius + state.p_radius[safe_slots]) < ROTATION_RADIUS_LIMIT
        tgt_angular_vel = jnp.where(tgt_rotating, state.angular_velocity / max_angular_velocity, jnp.float32(0.0))

        cand_feats = jnp.stack([
            jnp.ones((MAX_BASE_PLANETS, K), jnp.float32),             # [0] bias / valid-slot indicator (0 = padding/no-op)
            is_neutral,                                                 # [1] target is neutral — cheap to capture, no retaliation
            is_friendly,                                                # [2] target is friendly — reinforcement candidate
            is_enemy,                                                   # [3] target is enemy — offensive strike
            delta_x / board_size,                                       # [4] signed x-offset from src → encodes direction
            delta_y / board_size,                                       # [5] signed y-offset from src → encodes direction
            tgt_dist / board_size,                                      # [6] Euclidean distance — longer = more ticks in flight
            jnp.clip(tgt_ships_2d, 0.0, max_ships) / max_ships,       # [7] target garrison — ships needed to capture / reinforce
            state.p_production[safe_slots] / max_production,           # [8] target income rate — strategic value of capture
            tgt_angular_vel,                                            # [9] orbital speed of target; rotating targets are harder to hit
        ], axis=2).at[:, 0, :].set(0.0)  # [MAX_SRC, K, 10]; no-op slot = zeros

        # ── Fleet features  [MAX_FLEETS, 6] ──────────────────────────────
        # Velocity (not pure heading) so the cross-attention can extrapolate
        # "fleet_pos + t·vel" geometrically without re-learning _fleet_speed.
        f_is_enemy = (state.f_owner != player).astype(jnp.float32)
        f_spd      = _fleet_speed(state.f_ships)                              # [MAX_FLEETS]
        fleet_feats = jnp.stack([
            f_is_enemy,                                                       # [0] 1 if enemy fleet, else 0
            jnp.clip(state.f_ships, 0.0, max_ships) / max_ships,             # [1] fleet size — damage potential
            state.f_x / board_size,                                           # [2] normalized x position
            state.f_y / board_size,                                           # [3] normalized y position
            f_spd * jnp.cos(state.f_angle) / board_size,                      # [4] vx per tick, board-normalized
            f_spd * jnp.sin(state.f_angle) / board_size,                      # [5] vy per tick, board-normalized
        ], axis=1).astype(jnp.float32)                                        # [MAX_FLEETS, 6]
        fleet_mask = state.f_alive                                            # [MAX_FLEETS]

        return _EncodedTurn(
            src_valid       = src_valid,
            src_ids         = state.p_id[src_slots],
            src_ships       = src_ships,
            self_feats      = self_feats,
            src_slots       = src_slots,
            cand_slots_safe = safe_slots,
            cand_ids        = cand_ids,
            cand_mask       = cand_mask,
            ship_counts     = ship_counts,
            angles          = angles,
            cand_feats      = cand_feats,
            global_feat     = global_feat,
            fleet_feats     = fleet_feats,
            fleet_mask      = fleet_mask,
            step            = state.step,
        )

    return jax.jit(jax.vmap(_encode_single))
