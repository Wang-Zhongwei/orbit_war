"""
JAX-accelerated feature encoding, operating directly on EnvState.

Encodes observations for all envs in one JIT+vmap call:
  - Reads EnvState arrays directly (no dict conversion).
  - Game-design rule: only the sun can block fleets. Planets and comets are
    treated as transparent to fleet trajectories. This makes the cand_mask
    check closed-form (no iteration) and decouples it from the expensive
    intercept solver.
  - cand_mask: cheap sun-segment test using first-order extrapolation of
    rotating-planet position with a per-source speed estimate
    (max(src_ships//2, MIN_SOURCE_SHIPS) ships).
  - The accurate intercept angle (and the second sun check) is computed
    once per dispatched fleet in rollout.py with the policy's actual ships
    choice — not in the encoder.
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
_BUFFER_TICKS: int = 3 # buffer ticks for will_be_mine calculation is target is enemy

# Conservative buffer beyond SUN_RADIUS in cand_mask. Absorbs the fact that
# the encoder's speed estimate (src_ships//2 ships) can differ from what the
# ships head eventually picks — bigger ships → faster fleet → shorter t_hat →
# predicted target closer to current position. Margin biases the mask toward
# false-positive rejection (small action-space loss) rather than false-negative
# acceptance (which reintroduces gradient-noise on rejected dispatches).
_SUN_BLOCK_MARGIN: float = 1.01


def self_feature_dim() -> int:
    return 11


def candidate_feature_dim() -> int:
    return 14


def global_feature_dim() -> int:
    return 12


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
    tgt_slot: jnp.ndarray,      # int32 — to exclude from blockers
    ships: jnp.ndarray,         # scalar — for speed
    angular_velocity: jnp.ndarray,
    all_p_x: jnp.ndarray,       # [MAX_PLANETS] — for blocker check
    all_p_y: jnp.ndarray,
    all_p_r: jnp.ndarray,
    all_p_alive: jnp.ndarray,
    all_p_is_comet: jnp.ndarray,  # [MAX_PLANETS] bool — comets excluded as blockers
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (angle float32, is_blocked bool) for this (src, tgt) pair.

    Called once per dispatched fleet (rollout.py / jax_agent.py) with the
    policy's actual ship count. Iteratively solves the lead-angle for rotating
    targets, then checks the chord src → intercept-point against sun and static
    (non-rotating, non-comet) planet blockers.
    """
    spd = _fleet_speed(ships)

    # --- Rotating target: binary-search intercept via fixed scan ---
    # Use current position for init_ang so ang_t = init_ang + av*t predicts
    # where the planet will be t ticks from NOW, not t ticks from game start.
    dx0 = tgt_x - CENTER
    dy0 = tgt_y - CENTER
    orb_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    rotating = orb_r + tgt_radius < ROTATION_RADIUS_LIMIT
    init_ang = jnp.arctan2(dy0, dx0)

    def scan_body(carry, tick):
        best_angle, best_tx, best_ty, found = carry
        t = tick.astype(jnp.float32)
        ang_t = init_ang + angular_velocity * t
        tx = CENTER + orb_r * jnp.cos(ang_t)
        ty = CENTER + orb_r * jnp.sin(ang_t)
        dx = tx - src_x
        dy = ty - src_y
        gap = spd * t - (jnp.sqrt(dx * dx + dy * dy) - src_radius)
        valid = (~found) & (jnp.abs(gap) < tgt_radius)
        angle_t = jnp.arctan2(dy, dx)
        return (
            jnp.where(valid, angle_t, best_angle),
            jnp.where(valid, tx, best_tx),
            jnp.where(valid, ty, best_ty),
            found | valid,
        ), None

    (rot_angle, rot_tx, rot_ty, rot_found), _ = jax.lax.scan(
        scan_body,
        (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.bool_(False)),
        jnp.arange(1, _MAX_TICKS + 1, dtype=jnp.int32),
    )

    # --- Static / fallback angle ---
    dx_s = tgt_x - src_x
    dy_s = tgt_y - src_y
    static_angle = jnp.arctan2(dy_s, dx_s)

    no_intercept = rotating & ~rot_found
    use_rot = rotating & rot_found
    angle = jnp.where(use_rot, rot_angle, static_angle)

    # --- Blocking: segment from src centre to intercept point ---
    # For rotating targets the fleet aims at the predicted future position, so
    # the collision segment must use that endpoint.
    end_x = jnp.where(use_rot, rot_tx, tgt_x)
    end_y = jnp.where(use_rot, rot_ty, tgt_y)
    seg_dx = end_x - src_x
    seg_dy = end_y - src_y
    seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy + 1e-30

    # Sun
    t_sun = jnp.clip(
        ((CENTER - src_x) * seg_dx + (CENTER - src_y) * seg_dy) / seg_l2, 0.0, 1.0
    )
    proj_x = src_x + t_sun * seg_dx
    proj_y = src_y + t_sun * seg_dy
    sun_blocked = (proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2 <= SUN_RADIUS ** 2

    # Static (non-rotating, non-comet) planets only. Moving planets are ignored —
    # their per-tick positions can't be cheaply evaluated against the straight-line
    # segment, and a current-snapshot check produces false positives/negatives.
    planet_slots = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    p_orb_r = jnp.hypot(all_p_x - CENTER, all_p_y - CENTER)
    is_static = (p_orb_r + all_p_r >= ROTATION_RADIUS_LIMIT) & ~all_p_is_comet
    blocker_mask = (
        all_p_alive
        & is_static
        & (planet_slots != src_slot)
        & (planet_slots != tgt_slot)
    )
    t_b = jnp.clip(
        ((all_p_x - src_x) * seg_dx + (all_p_y - src_y) * seg_dy) / seg_l2, 0.0, 1.0
    )
    proj_bx = src_x + t_b * seg_dx
    proj_by = src_y + t_b * seg_dy
    dist_sq = (all_p_x - proj_bx) ** 2 + (all_p_y - proj_by) ** 2
    planet_blocked = jnp.any(blocker_mask & (dist_sq <= all_p_r ** 2))

    is_blocked = sun_blocked | planet_blocked | no_intercept
    return angle, is_blocked


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
        return _find_angle_single(
            p_x[src_slot], p_y[src_slot], p_r[src_slot], src_slot,
            p_x[tgt_slot], p_y[tgt_slot], p_r[tgt_slot], tgt_slot,
            n_ships.astype(jnp.float32), angular_velocity,
            p_x, p_y, p_r, p_alive, p_is_comet,
        )
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
    self_feats:      jnp.ndarray  # [MAX_SRC, 11]
    src_slots:       jnp.ndarray  # [MAX_SRC]       int32 — planet slot for each source
    cand_slots_safe: jnp.ndarray  # [MAX_SRC, K]    int32 (no-op clamped to 0)
    cand_ids:        jnp.ndarray  # [MAX_SRC, K]    int32 (-1 for no-op/invalid)
    cand_mask:       jnp.ndarray  # [MAX_SRC, K]    bool
    angles:          jnp.ndarray  # [MAX_SRC, K]    float32
    cand_feats:      jnp.ndarray  # [MAX_SRC, K, 14]
    global_feat:     jnp.ndarray  # [12]
    # fleet_feats:     jnp.ndarray  # [MAX_FLEETS, 6]
    # fleet_mask:      jnp.ndarray  # [MAX_FLEETS]    bool
    step:            jnp.ndarray  # []              int32


def _make_env_encoder(K: int, env_cfg):
    """Return a JIT+vmap encoder: (batched_state, players[N int32]) -> _EncodedTurn[N,...].

    Call once at JaxVecEnv init; reuse every tick.
    """
    from .constants import MAX_ANGULAR_VELOCITY, BOARD_SIZE, MAX_PRODUCTION

    K_enemy    = K // 3
    K_neutral  = K // 3
    K_friendly = K - 1 - K_enemy - K_neutral   # friendly quota; total non-noop slots = K-1

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

        # ── Ship counts (per-candidate offensive estimate, unused for blocking) ──
        tgt_ships_2d  = state.p_ships[safe_slots]                             # [MAX_SRC, K]

        # ── Source-level state needed by the cand_mask geometry ──────────
        src_ships        = jnp.round(state.p_ships[src_slots]).astype(jnp.int32)  # [MAX_SRC]
        has_enough_ships = src_ships[:, None] >= MIN_SOURCE_SHIPS                 # [MAX_SRC, K]

        # Estimated fleet speed for the sun-block check. Use src_ships//2 (the
        # median over uniform ships-buckets) clamped to MIN_SOURCE_SHIPS — this
        # matches what the policy is most likely to dispatch on average. The
        # margin on SUN_RADIUS below absorbs the residual variance.
        n_ships_est = jnp.maximum(src_ships // 2, MIN_SOURCE_SHIPS)               # [MAX_SRC]
        v_est_src   = _fleet_speed(n_ships_est.astype(jnp.float32))               # [MAX_SRC]

        # ── Closed-form sun-block check + aim angle ──────────────────────
        # Predicts where the candidate will be when a fleet arrives, then tests
        # whether the chord src → predicted_tgt passes inside SUN_RADIUS.
        # Game rule: only the sun blocks fleets; planets/comets are transparent.
        src_x_b = state.p_x[src_slots][:, None]                                   # [MAX_SRC, 1]
        src_y_b = state.p_y[src_slots][:, None]
        tgt_x_b = state.p_x[safe_slots]                                            # [MAX_SRC, K]
        tgt_y_b = state.p_y[safe_slots]

        # First-order arrival time estimate using current src→tgt distance.
        dx_now = tgt_x_b - src_x_b
        dy_now = tgt_y_b - src_y_b
        d_now  = jnp.sqrt(dx_now * dx_now + dy_now * dy_now + 1e-12)
        t_hat  = d_now / v_est_src[:, None]                                        # [MAX_SRC, K]

        # Comets are excluded from candidates upstream, so this only needs the
        # static inner-orbit criterion for planets that rotate around the sun.
        tgt_orb_dx = tgt_x_b - CENTER
        tgt_orb_dy = tgt_y_b - CENTER
        tgt_orb_r  = jnp.sqrt(tgt_orb_dx * tgt_orb_dx + tgt_orb_dy * tgt_orb_dy)
        is_rotating = tgt_orb_r + state.p_radius[safe_slots] < ROTATION_RADIUS_LIMIT

        # Exact rotation (not Taylor) — same FLOPs, no radial drift bug.
        # For non-rotating targets angle_step = 0 → predicted = current.
        angle_step = jnp.where(is_rotating, state.angular_velocity * t_hat, jnp.float32(0.0))
        c_a = jnp.cos(angle_step)
        s_a = jnp.sin(angle_step)
        pred_dx = c_a * tgt_orb_dx - s_a * tgt_orb_dy
        pred_dy = s_a * tgt_orb_dx + c_a * tgt_orb_dy
        pred_tx = CENTER + pred_dx                                                 # [MAX_SRC, K]
        pred_ty = CENTER + pred_dy

        # Aim angle the BC label extractor matches against expert launch angles.
        # Static targets: pred = current, so angle is just arctan2(tgt - src).
        # Rotating targets: lead-corrected via first-order prediction.
        angles = jnp.arctan2(pred_ty - src_y_b, pred_tx - src_x_b)                # [MAX_SRC, K]

        # Sun chord-test: point-segment distance from CENTER to src→pred_tgt.
        seg_dx = pred_tx - src_x_b
        seg_dy = pred_ty - src_y_b
        seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy + 1e-12
        t_proj = jnp.clip(
            ((CENTER - src_x_b) * seg_dx + (CENTER - src_y_b) * seg_dy) / seg_l2,
            0.0, 1.0,
        )
        proj_x = src_x_b + t_proj * seg_dx
        proj_y = src_y_b + t_proj * seg_dy
        sun_dist_sq = (proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2
        blocked = sun_dist_sq < (SUN_RADIUS * _SUN_BLOCK_MARGIN) ** 2              # [MAX_SRC, K]                        # no-op sends 0 ships

        cand_mask = (all_cand_valid & ~blocked & has_enough_ships).at[:, 0].set(True)  # no-op always valid

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

        src_per_planet_feats = jnp.stack([
            jnp.ones(MAX_BASE_PLANETS, jnp.float32),                              # [0] bias term
            src_x / board_size,                                                    # [1] normalized board-x position
            src_y / board_size,                                                    # [2] normalized board-y position
            jnp.clip(state.p_ships[src_slots], 0.0, max_ships) / max_ships,      # [3] current garrison
            state.p_production[src_slots] / max_production,                       # [4] income rate
        ], axis=1)  # [MAX_SRC, 5]

        src_global_stats = jnp.stack([
            n_my_planets / max_planets,     # [6] friendly planet count
            n_enemy_planets / max_planets,  # [7] enemy planet count
            my_planet_ships / ship_norm,    # [8] total friendly fleet strength
            enemy_planet_ships / ship_norm, # [9] total enemy fleet strength
        ])
        src_orbit_r  = jnp.hypot(src_x - CENTER, src_y - CENTER) / (board_size * jnp.float32(0.5))
        src_rot_flag = src_rotating.astype(jnp.float32)

        self_feats = jnp.concatenate([
            src_per_planet_feats,
            jnp.broadcast_to(src_global_stats[None, :], (MAX_BASE_PLANETS, 4)),
            src_orbit_r[:, None],    # [9]  orbit radius — inner vs outer ring
            src_rot_flag[:, None],   # [10] 1 if rotating planet, 0 if static/comet
        ], axis=1)  # [MAX_SRC, 11]

        # ── Candidate features  [MAX_SRC, K, 14] ────────────────────────
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
        tgt_rot_flag = tgt_rotating.astype(jnp.float32)

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

        def _sim_tick(acc, t):
            t_f  = t.astype(jnp.float32)
            fx_t = state.f_x + t_f * f_vx_1d                              # [MAX_FLEETS]
            fy_t = state.f_y + t_f * f_vy_1d
            ang  = state.angular_velocity * t_f
            c, s = jnp.cos(ang), jnp.sin(ang)
            px_t = CENTER + jnp.where(all_rotating, c * all_orb_dx - s * all_orb_dy, all_orb_dx)
            py_t = CENTER + jnp.where(all_rotating, s * all_orb_dx + c * all_orb_dy, all_orb_dy)
            dx   = px_t[:, None] - fx_t[None, :]                          # [MAX_PLANETS, MAX_FLEETS]
            dy   = py_t[:, None] - fy_t[None, :]
            hit  = (dx ** 2 + dy ** 2) < all_r_sq[:, None]
            return acc | hit, None

        inbound_all, _ = jax.lax.scan(
            _sim_tick,
            jnp.zeros((MAX_PLANETS, MAX_FLEETS), dtype=jnp.bool_),
            jnp.arange(1, _T_INBOUND + 1, dtype=jnp.int32),
        )  # [MAX_PLANETS, MAX_FLEETS]

        inbound_sim = inbound_all[safe_slots]                              # [MAX_SRC, K, MAX_FLEETS]

        f_alive_3d  = state.f_alive[None, None, :]
        is_my_3d    = (state.f_owner == player)[None, None, :]
        is_opp_3d   = (state.f_owner != player)[None, None, :]
        inbound     = inbound_sim & f_alive_3d

        f_ships_clipped = jnp.clip(state.f_ships, jnp.float32(0.0), max_ships)[None, None, :]
        my_inbound_ships  = ((inbound & is_my_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships
        opp_inbound_ships = ((inbound & is_opp_3d).astype(jnp.float32) * f_ships_clipped).sum(axis=2) / max_ships

        # ── can_capture ───────────────────────────────────────────────────
        # True when source ships > projected garrison at arrival:
        #   garrison = current_ships + production*eta (enemy-owned only)
        #              + enemy_inbound_before_eta − my_inbound_before_eta
        # ETA per fleet uses first-order distance to current target position.
        f_x_3d      = state.f_x[None, None, :]                            # [1, 1, MAX_FLEETS]
        f_y_3d      = state.f_y[None, None, :]
        dx_f2t      = tgt_x[:, :, None] - f_x_3d                          # [MAX_SRC, K, MAX_FLEETS]
        dy_f2t      = tgt_y[:, :, None] - f_y_3d
        dist_f2t    = jnp.sqrt(dx_f2t ** 2 + dy_f2t ** 2 + jnp.float32(1e-12))
        v_src_full  = _fleet_speed(state.p_ships[src_slots][:, None].astype(jnp.float32))
        eta_mine    = tgt_dist / v_src_full                                # [MAX_SRC, K]
        eta_fleet   = dist_f2t / f_spd_1d[None, None, :]                  # [MAX_SRC, K, MAX_FLEETS]
        early       = eta_fleet < eta_mine[:, :, None]
        f_ships_raw = state.f_ships[None, None, :].astype(jnp.float32)
        my_early    = ((inbound & is_my_3d & early).astype(jnp.float32) * f_ships_raw).sum(axis=2)
        opp_early   = ((inbound & is_opp_3d & early).astype(jnp.float32) * f_ships_raw).sum(axis=2)
        prod_at_eta  = state.p_production[safe_slots] * eta_mine             # [MAX_SRC, K]
        buffer_prod  = _BUFFER_TICKS * state.p_production[safe_slots]
        tgt_ships_f  = tgt_ships_2d.astype(jnp.float32)
        # Our side: friendly garrison + production (if we own it) + inbound allies
        our_force    = is_friendly * (tgt_ships_f + prod_at_eta) + my_early
        # Their side: enemy garrison + production (if they own it) + inbound enemies
        their_force  = is_enemy * (tgt_ships_f + prod_at_eta + buffer_prod) + is_neutral * tgt_ships_f + opp_early
        will_be_mine = (our_force > their_force).astype(jnp.float32)

        cand_feats = jnp.stack([
            jnp.ones((MAX_BASE_PLANETS, K), jnp.float32),             # [0]  bias
            is_neutral,                                                 # [1]  neutral target
            is_friendly,                                                # [2]  friendly target
            is_enemy,                                                   # [3]  enemy target
            delta_x / board_size,                                       # [4]  x-offset
            delta_y / board_size,                                       # [5]  y-offset
            tgt_dist / board_size,                                      # [6]  Euclidean distance
            jnp.clip(tgt_ships_2d, 0.0, max_ships) / max_ships,       # [7]  target garrison
            state.p_production[safe_slots] / max_production,           # [8]  target income rate
            tgt_orbit_r,                                                # [9]  orbit radius
            tgt_rot_flag,                                               # [10] is rotating
            my_inbound_ships,                                           # [11] friendly inbound ships
            opp_inbound_ships,                                          # [12] enemy inbound ships
            will_be_mine,                                               # [13] planet projected mine at eta (no action)
        ], axis=2).at[:, 0, :].set(0.0)  # [MAX_SRC, K, 14]; no-op slot = zeros

        # ── Fleet features  [MAX_FLEETS, 6] ──────────────────────────────
        # Velocity (not pure heading) so the cross-attention can extrapolate
        # "fleet_pos + t·vel" geometrically without re-learning _fleet_speed.
        # f_is_enemy = (state.f_owner != player).astype(jnp.float32)
        # f_spd      = _fleet_speed(state.f_ships)                              # [MAX_FLEETS]
        # fleet_feats = jnp.stack([
        #     f_is_enemy,                                                       # [0] 1 if enemy fleet, else 0
        #     jnp.clip(state.f_ships, 0.0, max_ships) / max_ships,             # [1] fleet size — damage potential
        #     state.f_x / board_size,                                           # [2] normalized x position
        #     state.f_y / board_size,                                           # [3] normalized y position
        #     f_spd * jnp.cos(state.f_angle) / board_size,                      # [4] vx per tick, board-normalized
        #     f_spd * jnp.sin(state.f_angle) / board_size,                      # [5] vy per tick, board-normalized
        # ], axis=1).astype(jnp.float32)                                        # [MAX_FLEETS, 6]
        # fleet_mask = state.f_alive                                            # [MAX_FLEETS]

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
            # fleet_feats     = fleet_feats,
            # fleet_mask      = fleet_mask,
            step            = state.step,
        )

    return jax.jit(jax.vmap(_encode_single))
