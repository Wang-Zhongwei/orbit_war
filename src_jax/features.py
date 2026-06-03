"""
JAX-accelerated feature encoding, operating directly on EnvState.

Encodes observations for all envs in one JIT+vmap call:
  - Reads EnvState arrays directly (no dict conversion).
  - Fleet size = full source-planet ships, so speed is exact at encode time.
  - cand_mask and angles: computed once via the iterative intercept solver
    (sun + static-planet blocking) for all [MAX_SRC, K] candidate pairs.
    No re-aim is needed after policy output.
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

MIN_SOURCE_SHIPS: int = 10
_MAX_TICKS: int = 25
_T_INBOUND: int = 25  # tick horizon for inbound fleet simulation


def self_feature_dim() -> int:
    return 25


def candidate_feature_dim() -> int:
    return 30


def global_feature_dim() -> int:
    return 12



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
    tgt_slot: jnp.ndarray,      # int32 — to exclude from blockers
    ships: jnp.ndarray,         # scalar — for speed
    angular_velocity: jnp.ndarray,
    all_p_x: jnp.ndarray,       # [MAX_PLANETS] — for blocker check
    all_p_y: jnp.ndarray,
    all_p_r: jnp.ndarray,
    all_p_alive: jnp.ndarray,
    all_p_is_comet: jnp.ndarray,  # [MAX_PLANETS] bool — comets excluded as blockers
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (angle float32, is_blocked bool, eta_ticks float32) for this (src, tgt) pair.

    Rotating targets: iterative tick-scan for the lead angle; single blocking
    check at the predicted intercept point.
    Static targets: tries three directions — center, left-tangent edge,
    right-tangent edge (each 95% of the half-angle to the planet rim). Uses
    the first unblocked direction (center preferred). is_blocked=True only if
    all three are blocked.
    eta_ticks: intercept tick for rotating targets; snapshot dist/speed for static.
    """
    spd = _fleet_speed(ships)

    # --- Rotating target: binary-search intercept via fixed scan ---
    # Carry the current orbit basis through the scan so the loop pays one
    # sin/cos pair for the fixed angular-velocity step instead of per tick.
    dx0 = tgt_x - CENTER
    dy0 = tgt_y - CENTER
    orb_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    rotating = orb_r + tgt_radius < ROTATION_RADIUS_LIMIT
    inv_orb_r = jnp.reciprocal(jnp.maximum(orb_r, jnp.float32(1e-6)))
    c_now = dx0 * inv_orb_r
    s_now = dy0 * inv_orb_r
    c_step = jnp.cos(angular_velocity)
    s_step = jnp.sin(angular_velocity)
    c_tick1 = c_step * c_now - s_step * s_now
    s_tick1 = s_step * c_now + c_step * s_now

    def scan_body(carry, tick):
        best_angle, best_tx, best_ty, best_tick, found, c_t, s_t = carry
        t = tick.astype(jnp.float32)
        tx = CENTER + orb_r * c_t
        ty = CENTER + orb_r * s_t
        dx = tx - src_x
        dy = ty - src_y
        gap = spd * t - (jnp.sqrt(dx * dx + dy * dy) - src_radius)
        valid = (~found) & (jnp.abs(gap) < tgt_radius)
        angle_t = jnp.arctan2(dy, dx)
        next_c_t = c_step * c_t - s_step * s_t
        next_s_t = s_step * c_t + c_step * s_t
        return (
            jnp.where(valid, angle_t, best_angle),
            jnp.where(valid, tx, best_tx),
            jnp.where(valid, ty, best_ty),
            jnp.where(valid, t, best_tick),
            found | valid,
            next_c_t,
            next_s_t,
        ), None

    (rot_angle, rot_tx, rot_ty, rot_tick, rot_found, _, _), _ = jax.lax.scan(
        scan_body,
        (
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.bool_(False),
            c_tick1,
            s_tick1,
        ),
        jnp.arange(1, _MAX_TICKS + 1, dtype=jnp.int32),
    )

    # --- Static candidate angles ---
    dx_s = tgt_x - src_x
    dy_s = tgt_y - src_y
    d_s  = jnp.sqrt(dx_s * dx_s + dy_s * dy_s + 1e-12)
    center_ang = jnp.arctan2(dy_s, dx_s)
    # Half-angle of the tangent cone: asin(r / d). Aiming within ±half_ang
    # of center_ang still lands inside the planet's disc.
    half_ang = jnp.arcsin(jnp.clip(tgt_radius / jnp.maximum(d_s, tgt_radius + 1e-6), 0.0, 1.0))

    no_intercept = rotating & ~rot_found
    use_rot      = rotating & rot_found

    # --- Shared blocker mask (same for all segment checks) ---
    planet_slots = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    p_orb_r      = jnp.hypot(all_p_x - CENTER, all_p_y - CENTER)
    is_static_p  = (p_orb_r + all_p_r >= ROTATION_RADIUS_LIMIT) & ~all_p_is_comet
    blocker_mask = (
        all_p_alive
        & is_static_p
        & (planet_slots != src_slot)
        & (planet_slots != tgt_slot)
    )

    def _seg_blocked(end_x, end_y):
        """True if the chord src→end passes through the sun or a blocking planet."""
        seg_dx = end_x - src_x
        seg_dy = end_y - src_y
        seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy + 1e-30
        t_sun  = jnp.clip(
            ((CENTER - src_x) * seg_dx + (CENTER - src_y) * seg_dy) / seg_l2, 0.0, 1.0
        )
        sun_b = (
            (src_x + t_sun * seg_dx - CENTER) ** 2
            + (src_y + t_sun * seg_dy - CENTER) ** 2
        ) <= SUN_RADIUS ** 2
        t_b     = jnp.clip(
            ((all_p_x - src_x) * seg_dx + (all_p_y - src_y) * seg_dy) / seg_l2, 0.0, 1.0
        )
        proj_bx = src_x + t_b * seg_dx
        proj_by = src_y + t_b * seg_dy
        planet_b = jnp.any(
            blocker_mask & ((all_p_x - proj_bx) ** 2 + (all_p_y - proj_by) ** 2 <= all_p_r ** 2)
        )
        return sun_b | planet_b

    def _static_endpoint(theta):
        """First contact point on the static target planet for aim angle theta."""
        dxr     = jnp.cos(theta)
        dyr     = jnp.sin(theta)
        t_c     = dx_s * dxr + dy_s * dyr
        perp_sq = jnp.maximum(dx_s * dx_s + dy_s * dy_s - t_c * t_c, 0.0)
        t_hit   = t_c - jnp.sqrt(jnp.maximum(tgt_radius * tgt_radius - perp_sq, 0.0))
        return src_x + t_hit * dxr, src_y + t_hit * dyr

    # Three candidate aim angles for static targets.
    # 0.95 factor keeps the fleet inside the planet disc (not a true tangent miss).
    _TANG = jnp.float32(0.95)
    ang_c = center_ang
    ang_l = center_ang - half_ang * _TANG
    ang_r = center_ang + half_ang * _TANG

    ec_x, ec_y = _static_endpoint(ang_c)
    el_x, el_y = _static_endpoint(ang_l)
    er_x, er_y = _static_endpoint(ang_r)

    blk_c = _seg_blocked(ec_x, ec_y)
    blk_l = _seg_blocked(el_x, el_y)
    blk_r = _seg_blocked(er_x, er_y)

    # Prefer center; fall back to left then right. Blocked iff all three fail.
    static_ang = jnp.where(~blk_c, ang_c, jnp.where(~blk_l, ang_l, ang_r))
    static_blk = blk_c & blk_l & blk_r

    # --- Rotating case: single blocking check at predicted intercept point ---
    rot_end_x = jnp.where(use_rot, rot_tx, tgt_x)
    rot_end_y = jnp.where(use_rot, rot_ty, tgt_y)
    rot_blk   = _seg_blocked(rot_end_x, rot_end_y)

    # --- Final selection ---
    angle      = jnp.where(use_rot, rot_angle, static_ang)
    is_blocked = no_intercept | jnp.where(rotating, rot_blk, static_blk)
    static_eta = d_s / jnp.maximum(spd, jnp.float32(1e-6))
    eta_ticks  = jnp.where(use_rot, rot_tick, static_eta)
    return angle, is_blocked, eta_ticks


def aim_fleets(
    p_x: jnp.ndarray,
    p_y: jnp.ndarray,
    p_r: jnp.ndarray,
    p_alive: jnp.ndarray,
    p_is_comet: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    src_slots: jnp.ndarray,
    tgt_slots: jnp.ndarray,
    ships_send: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vmapped re-aim for encoder-selected non-comet targets."""
    def _per(src_slot, tgt_slot, n_ships):
        angle, is_blocked, _ = _find_angle_single(
            p_x[src_slot], p_y[src_slot], p_r[src_slot], src_slot,
            p_x[tgt_slot], p_y[tgt_slot], p_r[tgt_slot], tgt_slot,
            n_ships.astype(jnp.float32), angular_velocity,
            p_x, p_y, p_r, p_alive, p_is_comet,
        )
        return angle, is_blocked
    return jax.vmap(_per)(src_slots, tgt_slots, ships_send)


# ===========================================================================
# Batched encoder — processes N envs in one JIT+vmap call
# ===========================================================================

class _EncodedTurn(NamedTuple):
    """Fixed-size encoded observation for one env (no batch dim).
    After vmap the leading axis is N (number of envs)."""
    src_valid:       jnp.ndarray  # [MAX_SRC]       bool
    src_ids:         jnp.ndarray  # [MAX_SRC]       int32
    src_ships:       jnp.ndarray  # [MAX_SRC]       int32
    self_feats:      jnp.ndarray  # [MAX_SRC, 25]
    src_slots:       jnp.ndarray  # [MAX_SRC]       int32 — planet slot for each source
    cand_slots_safe: jnp.ndarray  # [MAX_SRC, K]    int32 (no-op clamped to 0)
    cand_ids:        jnp.ndarray  # [MAX_SRC, K]    int32 (-1 for no-op/invalid)
    cand_mask:       jnp.ndarray  # [MAX_SRC, K]    bool
    angles:          jnp.ndarray  # [MAX_SRC, K]    float32
    cand_feats:      jnp.ndarray  # [MAX_SRC, K, 30]
    global_feat:     jnp.ndarray  # [12]
    step:            jnp.ndarray  # []              int32


def _make_env_encoder(K: int, env_cfg):
    """Return a JIT+vmap encoder: (batched_state, players[N int32]) -> _EncodedTurn[N,...].

    Call once at JaxVecEnv init; reuse every tick.
    """
    from .constants import MAX_ANGULAR_VELOCITY, BOARD_SIZE, MAX_PRODUCTION

    K_enemy    = K // 3
    K_friendly = K // 3
    K_neutral  = K - 1 - K_enemy - K_friendly

    board_size           = jnp.float32(BOARD_SIZE)
    max_ships            = jnp.float32(env_cfg.max_ships)
    max_production       = jnp.float32(MAX_PRODUCTION)
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
            & (state.p_ships >= MIN_SOURCE_SHIPS)
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
                valid       = sorted_dist[slots] < INF
                return slots, valid

            is_enemy_planet    = excl_mask & (state.p_owner != -1) & (state.p_owner != player)
            is_neutral_planet  = excl_mask & (state.p_owner == -1)
            is_friendly_planet = excl_mask & (state.p_owner == player)

            enemy_slots,    enemy_valid    = top_k(is_enemy_planet,    K_enemy)
            neutral_slots,  neutral_valid  = top_k(is_neutral_planet,  K_neutral)
            friendly_slots, friendly_valid = top_k(is_friendly_planet, K_friendly)

            # Flatten per-category picks [K-1]
            per_slots = jnp.concatenate([enemy_slots,  neutral_slots,  friendly_slots])
            per_valid = jnp.concatenate([enemy_valid,  neutral_valid,  friendly_valid])

            # Overflow pool: nearest planets not already claimed by a quota slot.
            # Handles the case where a category has fewer planets than its quota —
            # instead of wasting those slots, fill them with the next-closest planets
            # from any category.
            selected   = jnp.zeros(MAX_PLANETS, dtype=jnp.bool_).at[per_slots].set(per_valid)
            ovfl_dist  = jnp.where(excl_mask & ~selected, dist_to_src, INF)
            ovfl_slots = jnp.argsort(ovfl_dist)[:K - 1]   # [K-1] — max deficit is K-1
            ovfl_valid = ovfl_dist[ovfl_slots] < INF

            # Exclusive prefix-sum of invalids → overflow index for each slot.
            # per_valid[i]=False means: use ovfl_slots[# of invalids before i].
            inv      = (~per_valid).astype(jnp.int32)
            ovfl_idx = jnp.minimum(jnp.cumsum(inv) - inv, K - 2)  # clamp to [0, K-2]

            final_slots = jnp.where(per_valid, per_slots, ovfl_slots[ovfl_idx])
            final_valid = jnp.where(per_valid, True,      ovfl_valid[ovfl_idx])

            cand_slots = jnp.concatenate([jnp.array([-1], jnp.int32), final_slots])
            cand_valid = jnp.concatenate([jnp.array([True], bool),     final_valid])
            return cand_slots, cand_valid

        all_cand_slots, all_cand_valid = jax.vmap(_select_cands)(src_slots)
        # [MAX_SRC, K], [MAX_SRC, K]
        safe_slots = jnp.clip(all_cand_slots, 0, MAX_PLANETS - 1)  # [MAX_SRC, K]

        # ── Ship counts (per-candidate offensive estimate, unused for blocking) ──
        tgt_ships_2d  = state.p_ships[safe_slots]                             # [MAX_SRC, K]

        # ── Source ships (actual full fleet size) ───────────────────────────
        src_ships = jnp.round(state.p_ships[src_slots]).astype(jnp.int32)    # [MAX_SRC]

        # ── Exact aim angles + blocking for all [MAX_SRC, K] pairs ─────────
        # Fleet size = full src_ships so speed is known exactly. The iterative
        # intercept solver checks sun + static-planet blocking — no re-aim
        # needed after policy output.
        def _row(src_slot, tgt_slots_row, n_ships):
            def _pair(tgt_slot):
                return _find_angle_single(
                    state.p_x[src_slot], state.p_y[src_slot], state.p_radius[src_slot], src_slot,
                    state.p_x[tgt_slot], state.p_y[tgt_slot], state.p_radius[tgt_slot], tgt_slot,
                    n_ships.astype(jnp.float32), state.angular_velocity,
                    state.p_x, state.p_y, state.p_radius, state.p_alive, state.p_is_comet,
                )
            return jax.vmap(_pair)(tgt_slots_row)

        angles, blocked_2d, eta_ticks = jax.vmap(_row)(src_slots, safe_slots, src_ships)
        # angles: [MAX_SRC, K], blocked_2d: [MAX_SRC, K], eta_ticks_2d: [MAX_SRC, K]

        cand_mask = (all_cand_valid & ~blocked_2d).at[:, 0].set(True)

        # ── Candidate IDs ─────────────────────────────────────────────────
        cand_ids = jnp.where(
            all_cand_slots >= 0,
            state.p_id[safe_slots],
            jnp.int32(-1),
        )  # [MAX_SRC, K]

        # ── Global features  [13] ────────────────────────────────────────
        my_planet_mask      = state.p_alive & (state.p_owner == player)
        enemy_planet_mask   = state.p_alive & (state.p_owner != -1) & (state.p_owner != player)
        neutral_planet_mask = state.p_alive & (state.p_owner == -1)
        my_fleet_mask       = state.f_alive & (state.f_owner == player)
        enemy_fleet_mask    = state.f_alive & (state.f_owner != player)

        n_my_planets      = my_planet_mask.sum().astype(jnp.float32)
        n_enemy_planets   = enemy_planet_mask.sum().astype(jnp.float32)
        n_neutral_planets = neutral_planet_mask.sum().astype(jnp.float32)
        my_planet_ships   = (state.p_ships * my_planet_mask).sum()
        enemy_planet_ships = (state.p_ships * enemy_planet_mask).sum()
        my_fleet_ships    = (state.f_ships * my_fleet_mask).sum()
        enemy_fleet_ships = (state.f_ships * enemy_fleet_mask).sum()
        my_production     = (state.p_production * my_planet_mask).sum()
        enemy_production  = (state.p_production * enemy_planet_mask).sum()
        ship_norm         = max_planets * max_ships
        production_norm   = max_planets * max_production

        is_4player = (state.num_agents == 4).astype(jnp.float32)           # 0=1v1, 1=4-player

        global_feat = jnp.stack([
            state.step / episode_steps,                                   # [0] game-time fraction
            is_4player,                                                   # [1] 0=1v1, 1=4-player
            n_my_planets / max_planets,                                   # [2] friendly planet fraction
            n_enemy_planets / max_planets,                                # [3] enemy planet fraction
            n_neutral_planets / max_planets,                              # [4] unclaimed planets
            my_planet_ships / ship_norm,                                  # [5] ships on friendly planets
            enemy_planet_ships / ship_norm,                               # [6] ships on enemy planets
            my_fleet_ships / ship_norm,                                   # [7] friendly ships in flight
            enemy_fleet_ships / ship_norm,                                # [8] enemy ships in flight
            my_production / production_norm,                              # [9] total friendly production
            enemy_production / production_norm,                           # [10] total enemy production
            state.angular_velocity / max_angular_velocity,               # [11] orbital speed (shared by all planets)
        ]).astype(jnp.float32)

        # ── Self features  [MAX_SRC, 13] ─────────────────────────────────
        src_x         = state.p_x[src_slots]
        src_y         = state.p_y[src_slots]
        src_init_x    = state.p_init_x[src_slots]
        src_init_y    = state.p_init_y[src_slots]
        src_orb_radius = jnp.hypot(src_init_x - CENTER, src_init_y - CENTER)
        src_rotating  = (src_orb_radius + state.p_radius[src_slots]) < ROTATION_RADIUS_LIMIT

        src_orbit_r  = jnp.hypot(src_x - CENTER, src_y - CENTER) / (board_size * jnp.float32(0.5))
        src_rot_flag = src_rotating.astype(jnp.float32)

        # ── Candidate features  [MAX_SRC, K, 23] ───────────────────────
        tgt_x   = state.p_x[safe_slots]         # [MAX_SRC, K]
        tgt_y   = state.p_y[safe_slots]
        delta_x = tgt_x - src_x[:, None]
        delta_y = tgt_y - src_y[:, None]
        tgt_dist = jnp.hypot(delta_x, delta_y)

        tgt_owner   = state.p_owner[safe_slots]
        is_neutral  = (tgt_owner == -1).astype(jnp.float32)
        is_friendly = (tgt_owner == player).astype(jnp.float32)
        is_enemy    = ((tgt_owner != -1) & (tgt_owner != player)).astype(jnp.float32)

        tgt_init_x  = state.p_init_x[safe_slots]
        tgt_init_y  = state.p_init_y[safe_slots]
        tgt_orb_r   = jnp.hypot(tgt_init_x - CENTER, tgt_init_y - CENTER)
        tgt_rotating = (tgt_orb_r + state.p_radius[safe_slots]) < ROTATION_RADIUS_LIMIT

        tgt_orbit_r  = tgt_orb_r / (board_size * jnp.float32(0.5))   # [MAX_SRC, K] orbit radius feature
        tgt_rot_flag = tgt_rotating.astype(jnp.float32)                                                 # [MAX_SRC, K]

        # ── Inbound fleet detection via forward simulation ────────────────
        # Simulate over all MAX_PLANETS planets (carry [MAX_PLANETS, MAX_FLEETS])
        # then gather into candidate shape. Much smaller than [MAX_SRC, K, MAX_FLEETS].
        f_spd_1d    = _fleet_speed(state.f_ships.astype(jnp.float32))     # [MAX_FLEETS]
        f_vx_1d     = f_spd_1d * jnp.cos(state.f_angle)                   # [MAX_FLEETS]
        f_vy_1d     = f_spd_1d * jnp.sin(state.f_angle)
        all_orb_dx  = state.p_x - CENTER                                   # [MAX_PLANETS]
        all_orb_dy  = state.p_y - CENTER
        all_orb_r   = jnp.hypot(all_orb_dx, all_orb_dy)
        all_rotating = (all_orb_r + state.p_radius) < ROTATION_RADIUS_LIMIT  # [MAX_PLANETS]
        all_r_sq    = state.p_radius ** 2                                  # [MAX_PLANETS]

        # Signed garrison: positive = player owns, negative = enemy/neutral owns.
        # Production applied each tick in the sign direction; fleet arrivals add
        # (player) or subtract (enemy) from the signed value. Ownership flips
        # when the sign changes. Computed alongside the inbound accumulator in
        # the same scan so planet positions are only computed once per tick.
        _f_ships_f       = state.f_ships.astype(jnp.float32)                  # [MAX_FLEETS]
        _garrison_init   = jnp.where(state.p_alive, state.p_ships, jnp.float32(0.0))
        _owner_init      = state.p_owner.astype(jnp.int32)                    # -1 for neutral
        # prev_owner: last owner before garrison hit 0; used for production when
        # garrison == 0 so a battle-draw planet still produces for its last owner.
        _prev_owner_init = state.p_owner.astype(jnp.int32)

        tick_rot_c = jnp.cos(state.angular_velocity)
        tick_rot_s = jnp.sin(state.angular_velocity)

        def _sim_tick(carry, t):
            acc, acc_5, acc_10, acc_15, garrison, owner, prev_owner, owner_5, owner_10, owner_15, c_t, s_t = carry
            t_f  = t.astype(jnp.float32)
            fx_t = state.f_x + t_f * f_vx_1d                              # [MAX_FLEETS]
            fy_t = state.f_y + t_f * f_vy_1d
            px_t = CENTER + jnp.where(all_rotating, c_t * all_orb_dx - s_t * all_orb_dy, all_orb_dx)
            py_t = CENTER + jnp.where(all_rotating, s_t * all_orb_dx + c_t * all_orb_dy, all_orb_dy)
            dx   = px_t[:, None] - fx_t[None, :]                          # [MAX_PLANETS, MAX_FLEETS]
            dy   = py_t[:, None] - fy_t[None, :]
            already_landed = jnp.any(acc, axis=0)                              # [MAX_FLEETS]
            hit     = ((dx ** 2 + dy ** 2) < all_r_sq[:, None]) & ~already_landed[None, :]
            new_acc = acc | hit
            # Production: any owned planet produces. When garrison == 0 (draw),
            # fall back to prev_owner so the planet keeps producing.
            prod_owner = jnp.where(garrison > 0, owner, prev_owner)
            garrison   = garrison + state.p_production * (prod_owner != -1).astype(jnp.float32)
            # Mirror _phase_resolve_combat exactly:
            #   1. Sum arriving ships grouped by player → combat_ships [MAX_P, 4]
            #   2. top player = argmax; survivor = top − second_largest
            #   3. survivor > 0 only: friendly → reinforce garrison, hostile → attack garrison
            #   4. Garrison goes negative → top player takes over
            alive_hit = hit & state.f_alive[None, :]                      # [MAX_PLANETS, MAX_FLEETS]
            combat_ships = jnp.stack([
                jnp.where(alive_hit & (state.f_owner[None, :] == pid),
                          _f_ships_f[None, :], jnp.float32(0.0)).sum(axis=1)
                for pid in range(4)
            ], axis=1)                                                     # [MAX_PLANETS, 4]
            has_combat   = jnp.any(combat_ships > 0, axis=1)              # [MAX_PLANETS]
            top_idx      = jnp.argmax(combat_ships, axis=1).astype(jnp.int32)
            top_ships    = jnp.max(combat_ships, axis=1)
            second_ships = jnp.sort(combat_ships, axis=1)[:, -2]
            survivor     = top_ships - second_ships                        # net after inter-fleet battle
            apply        = has_combat & (survivor > jnp.float32(0.0))
            friendly     = (top_idx == owner)
            garrison     = jnp.where(
                apply,
                jnp.where(friendly, garrison + survivor, garrison - survivor),
                garrison,
            )
            flipped  = apply & ~friendly & (garrison < jnp.float32(0.0))
            garrison = jnp.abs(garrison)
            owner    = jnp.where(flipped, top_idx, owner)
            prev_owner = jnp.where(garrison > 0, owner, prev_owner)
            next_c_t = tick_rot_c * c_t - tick_rot_s * s_t
            next_s_t = tick_rot_s * c_t + tick_rot_c * s_t
            return (
                new_acc,
                jnp.where(t <= 5,  new_acc, acc_5),
                jnp.where(t <= 10, new_acc, acc_10),
                jnp.where(t <= 15, new_acc, acc_15),
                garrison,
                owner,
                prev_owner,
                jnp.where(t == 5,  owner, owner_5),
                jnp.where(t == 10, owner, owner_10),
                jnp.where(t == 15, owner, owner_15),
                next_c_t,
                next_s_t,
            ), None

        _zeros_pf = jnp.zeros((MAX_PLANETS, MAX_FLEETS), dtype=jnp.bool_)
        (inbound_all, inbound_5_all, inbound_10_all, inbound_15_all, _, owner_final, _, owner_5_all, owner_10_all, owner_15_all, _, _), _ = jax.lax.scan(
            _sim_tick,
            (
                _zeros_pf,
                _zeros_pf,
                _zeros_pf,
                _zeros_pf,
                _garrison_init,
                _owner_init,
                _prev_owner_init,
                _owner_init,
                _owner_init,
                _owner_init,
                tick_rot_c,
                tick_rot_s,
            ),
            jnp.arange(1, _T_INBOUND + 1, dtype=jnp.int32),
        )  # [MAX_PLANETS, MAX_FLEETS] / [MAX_PLANETS]

        inbound_sim    = inbound_all[safe_slots]                           # [MAX_SRC, K, MAX_FLEETS]
        inbound_5_sim  = inbound_5_all[safe_slots]                        # [MAX_SRC, K, MAX_FLEETS]
        inbound_10_sim = inbound_10_all[safe_slots]                       # [MAX_SRC, K, MAX_FLEETS]
        inbound_15_sim = inbound_15_all[safe_slots]                       # [MAX_SRC, K, MAX_FLEETS]

        f_alive_3d  = state.f_alive[None, None, :]
        is_my_3d    = (state.f_owner == player)[None, None, :]
        is_opp_3d   = (state.f_owner != player)[None, None, :]
        inbound     = inbound_sim & f_alive_3d

        f_ships_clipped = jnp.clip(state.f_ships, jnp.float32(0.0), max_ships)[None, None, :]
        my_inbound_ships  = ((inbound & is_my_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        opp_inbound_ships = ((inbound & is_opp_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships

        inbound_5  = inbound_5_sim  & f_alive_3d
        inbound_10 = inbound_10_sim & f_alive_3d
        inbound_15 = inbound_15_sim & f_alive_3d
        my_inbound_5_ships   = ((inbound_5  & is_my_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        opp_inbound_5_ships  = ((inbound_5  & is_opp_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        my_inbound_10_ships  = ((inbound_10 & is_my_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        opp_inbound_10_ships = ((inbound_10 & is_opp_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        my_inbound_15_ships  = ((inbound_15 & is_my_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        opp_inbound_15_ships = ((inbound_15 & is_opp_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships

        would_be_mine_5  = (owner_5_all[safe_slots]  == player).astype(jnp.float32)    # [MAX_SRC, K]
        would_be_mine_10 = (owner_10_all[safe_slots] == player).astype(jnp.float32)    # [MAX_SRC, K]
        would_be_mine_15 = (owner_15_all[safe_slots] == player).astype(jnp.float32)    # [MAX_SRC, K]
        would_be_mine_25 = (owner_final[safe_slots]  == player).astype(jnp.float32)    # [MAX_SRC, K]

        # ── Nearby-planet reachability features ─────────────────────────────
        # For each candidate target, sum garrison ships from my/enemy planets
        # whose estimated travel time ceil(dist / fleet_speed(garrison)) is ≤ 5
        # or ≤ 10 ticks. Gives the model a signal about how "contested" or
        # "reinforceable" the target area is in the very near term.
        speed_p      = _fleet_speed(state.p_ships.astype(jnp.float32))        # [MAX_PLANETS]
        dist_p2c     = jnp.hypot(
            tgt_x[:, :, None] - state.p_x[None, None, :],                     # [MAX_SRC, K, MAX_PLANETS]
            tgt_y[:, :, None] - state.p_y[None, None, :],
        )
        eta_p2c      = jnp.ceil(dist_p2c / jnp.maximum(speed_p[None, None, :], jnp.float32(1e-6)))

        p_alive_3d    = state.p_alive[None, None, :]                          # [1, 1, MAX_PLANETS]
        is_my_p3d     = (state.p_owner == player)[None, None, :]
        is_enemy_p3d  = ((state.p_owner != player) & (state.p_owner != -1))[None, None, :]
        p_ships_3d    = jnp.clip(state.p_ships, jnp.float32(0.0), max_ships)[None, None, :]

        can_reach_5   = (eta_p2c <= 5)  & p_alive_3d
        can_reach_10  = (eta_p2c <= 10) & p_alive_3d
        can_reach_15  = (eta_p2c <= 15) & p_alive_3d
        my_nearby_5_ships     = ((can_reach_5  & is_my_p3d).astype(jnp.float32)    * p_ships_3d).sum(axis=2) / max_ships
        enemy_nearby_5_ships  = ((can_reach_5  & is_enemy_p3d).astype(jnp.float32) * p_ships_3d).sum(axis=2) / max_ships
        my_nearby_10_ships    = ((can_reach_10 & is_my_p3d).astype(jnp.float32)    * p_ships_3d).sum(axis=2) / max_ships
        enemy_nearby_10_ships = ((can_reach_10 & is_enemy_p3d).astype(jnp.float32) * p_ships_3d).sum(axis=2) / max_ships
        my_nearby_15_ships    = ((can_reach_15 & is_my_p3d).astype(jnp.float32)    * p_ships_3d).sum(axis=2) / max_ships
        enemy_nearby_15_ships = ((can_reach_15 & is_enemy_p3d).astype(jnp.float32) * p_ships_3d).sum(axis=2) / max_ships

        # ── Self features  [MAX_SRC, 25] ────────────────────────────────
        # Inbound fleets for each source planet (gathered from simulation above).
        inbound_src    = inbound_all[src_slots]                                # [MAX_SRC, MAX_FLEETS]
        inbound_5_src  = inbound_5_all[src_slots]
        inbound_10_src = inbound_10_all[src_slots]
        inbound_15_src = inbound_15_all[src_slots]

        f_alive_s = state.f_alive[None, :]                                     # [1, MAX_FLEETS]
        is_my_fs  = (state.f_owner == player)[None, :]
        is_opp_fs = (state.f_owner != player)[None, :]
        f_ships_s = jnp.clip(state.f_ships, jnp.float32(0.0), max_ships)[None, :]

        self_my_inb_5   = ((inbound_5_src  & f_alive_s & is_my_fs).astype(jnp.float32)  * f_ships_s).sum(axis=1) / max_ships
        self_opp_inb_5  = ((inbound_5_src  & f_alive_s & is_opp_fs).astype(jnp.float32) * f_ships_s).sum(axis=1) / max_ships
        self_my_inb_10  = ((inbound_10_src & f_alive_s & is_my_fs).astype(jnp.float32)  * f_ships_s).sum(axis=1) / max_ships
        self_opp_inb_10 = ((inbound_10_src & f_alive_s & is_opp_fs).astype(jnp.float32) * f_ships_s).sum(axis=1) / max_ships
        self_my_inb_15  = ((inbound_15_src & f_alive_s & is_my_fs).astype(jnp.float32)  * f_ships_s).sum(axis=1) / max_ships
        self_opp_inb_15 = ((inbound_15_src & f_alive_s & is_opp_fs).astype(jnp.float32) * f_ships_s).sum(axis=1) / max_ships
        self_my_inb_25  = ((inbound_src    & f_alive_s & is_my_fs).astype(jnp.float32)  * f_ships_s).sum(axis=1) / max_ships
        self_opp_inb_25 = ((inbound_src    & f_alive_s & is_opp_fs).astype(jnp.float32) * f_ships_s).sum(axis=1) / max_ships

        # Nearby planets for each source planet.
        dist_p2src  = jnp.hypot(
            src_x[:, None] - state.p_x[None, :],                              # [MAX_SRC, MAX_PLANETS]
            src_y[:, None] - state.p_y[None, :],
        )
        eta_p2src   = jnp.ceil(dist_p2src / jnp.maximum(speed_p[None, :], jnp.float32(1e-6)))
        p_alive_s   = state.p_alive[None, :]
        is_my_ps    = (state.p_owner == player)[None, :]
        is_enemy_ps = ((state.p_owner != player) & (state.p_owner != -1))[None, :]
        p_ships_s   = jnp.clip(state.p_ships, jnp.float32(0.0), max_ships)[None, :]
        src_reach_5   = (eta_p2src <= 5)  & p_alive_s
        src_reach_10  = (eta_p2src <= 10) & p_alive_s
        src_reach_15  = (eta_p2src <= 15) & p_alive_s
        self_my_near5     = ((src_reach_5  & is_my_ps).astype(jnp.float32)    * p_ships_s).sum(axis=1) / max_ships
        self_enemy_near5  = ((src_reach_5  & is_enemy_ps).astype(jnp.float32) * p_ships_s).sum(axis=1) / max_ships
        self_my_near10    = ((src_reach_10 & is_my_ps).astype(jnp.float32)    * p_ships_s).sum(axis=1) / max_ships
        self_enemy_near10 = ((src_reach_10 & is_enemy_ps).astype(jnp.float32) * p_ships_s).sum(axis=1) / max_ships
        self_my_near15    = ((src_reach_15 & is_my_ps).astype(jnp.float32)    * p_ships_s).sum(axis=1) / max_ships
        self_enemy_near15 = ((src_reach_15 & is_enemy_ps).astype(jnp.float32) * p_ships_s).sum(axis=1) / max_ships

        self_feats = jnp.stack([
            # ── static ──────────────────────────────────────────────────────
            jnp.ones(MAX_BASE_PLANETS, jnp.float32),                          # [0]  bias
            src_x / board_size,                                                # [1]  x position
            src_y / board_size,                                                # [2]  y position
            src_orbit_r,                                                       # [3]  orbit radius
            src_rot_flag,                                                      # [4]  is rotating
            jnp.clip(state.p_ships[src_slots], 0.0, max_ships) / max_ships,  # [5]  garrison
            state.p_production[src_slots] / max_production,                   # [6]  production rate
            # ── inbound fleets ──────────────────────────────────────────────
            self_my_inb_5,                                                     # [7]  friendly inbound (5-tick)
            self_opp_inb_5,                                                    # [8]  enemy inbound (5-tick)
            self_my_inb_10,                                                    # [9]  friendly inbound (10-tick)
            self_opp_inb_10,                                                   # [10] enemy inbound (10-tick)
            self_my_inb_15,                                                    # [11] friendly inbound (15-tick)
            self_opp_inb_15,                                                   # [12] enemy inbound (15-tick)
            self_my_inb_25,                                                    # [13] friendly inbound (25-tick)
            self_opp_inb_25,                                                   # [14] enemy inbound (25-tick)
            # ── nearby planets ──────────────────────────────────────────────
            self_my_near5,                                                     # [15] my garrison reachable in 5 ticks
            self_enemy_near5,                                                  # [16] enemy garrison reachable in 5 ticks
            self_my_near10,                                                    # [17] my garrison reachable in 10 ticks
            self_enemy_near10,                                                 # [18] enemy garrison reachable in 10 ticks
            self_my_near15,                                                    # [19] my garrison reachable in 15 ticks
            self_enemy_near15,                                                 # [20] enemy garrison reachable in 15 ticks
            # ── prediction ──────────────────────────────────────────────────
            (owner_5_all[src_slots]  == player).astype(jnp.float32),          # [21] will still be mine after 5 ticks
            (owner_10_all[src_slots] == player).astype(jnp.float32),          # [22] will still be mine after 10 ticks
            (owner_15_all[src_slots] == player).astype(jnp.float32),          # [23] will still be mine after 15 ticks
            (owner_final[src_slots]  == player).astype(jnp.float32),          # [24] will still be mine after 25 ticks
        ], axis=1)  # [MAX_SRC, 25]

        # ── Candidate features  [MAX_SRC, K, 30] ────────────────────────
        cand_feats = jnp.stack([
            # ── identity ────────────────────────────────────────────────────
            jnp.ones((MAX_BASE_PLANETS, K), jnp.float32),                     # [0]  bias
            is_neutral,                                                        # [1]  neutral target
            is_friendly,                                                       # [2]  friendly target
            is_enemy,                                                          # [3]  enemy target
            # ── geometry ────────────────────────────────────────────────────
            delta_x / board_size,                                              # [4]  x offset from source
            delta_y / board_size,                                              # [5]  y offset from source
            tgt_dist / board_size,                                             # [6]  Euclidean distance
            eta_ticks / episode_steps,                                             # [7]  normalized ETA
            # ── target static properties ────────────────────────────────────
            jnp.clip(tgt_ships_2d, 0.0, max_ships) / max_ships,              # [8]  garrison
            state.p_production[safe_slots] / max_production,                  # [9]  production rate
            tgt_orbit_r,                                                       # [10] orbit radius
            tgt_rot_flag,                                                      # [11] is rotating
            # ── inbound fleets ──────────────────────────────────────────────
            my_inbound_5_ships,                                                # [12] friendly inbound (5-tick)
            opp_inbound_5_ships,                                               # [13] enemy inbound (5-tick)
            my_inbound_10_ships,                                               # [14] friendly inbound (10-tick)
            opp_inbound_10_ships,                                              # [15] enemy inbound (10-tick)
            my_inbound_15_ships,                                               # [16] friendly inbound (15-tick)
            opp_inbound_15_ships,                                              # [17] enemy inbound (15-tick)
            my_inbound_ships,                                                  # [18] friendly inbound (25-tick)
            opp_inbound_ships,                                                 # [19] enemy inbound (25-tick)
            # ── nearby planets ──────────────────────────────────────────────
            my_nearby_5_ships,                                                 # [20] my garrison reachable in 5 ticks
            enemy_nearby_5_ships,                                              # [21] enemy garrison reachable in 5 ticks
            my_nearby_10_ships,                                                # [22] my garrison reachable in 10 ticks
            enemy_nearby_10_ships,                                             # [23] enemy garrison reachable in 10 ticks
            my_nearby_15_ships,                                                # [24] my garrison reachable in 15 ticks
            enemy_nearby_15_ships,                                             # [25] enemy garrison reachable in 15 ticks
            # ── prediction ──────────────────────────────────────────────────
            would_be_mine_5,                                                   # [26] will be mine after 5-tick sim
            would_be_mine_10,                                                  # [27] will be mine after 10-tick sim
            would_be_mine_15,                                                  # [28] will be mine after 15-tick sim
            would_be_mine_25,                                                  # [29] will be mine after 25-tick sim
        ], axis=2).at[:, 0, :].set(0.0)  # [MAX_SRC, K, 30]; no-op slot = zeros

        return _EncodedTurn(
            src_valid       = src_valid,
            src_ids         = state.p_id[src_slots],
            src_ships       = src_ships,
            self_feats      = self_feats,
            src_slots       = src_slots,
            cand_slots_safe = safe_slots,
            cand_ids        = cand_ids,
            cand_mask       = cand_mask,
            angles          = angles,
            cand_feats      = cand_feats,
            global_feat     = global_feat,
            step            = state.step,
        )

    return jax.jit(jax.vmap(_encode_single))
