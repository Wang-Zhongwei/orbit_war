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

import os

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import (
    EnvState, MAX_PLANETS, MAX_BASE_PLANETS, MAX_FLEETS,
    N_COMET_GROUPS, COMET_PER_GROUP, COMET_SLOT_BASE, MAX_PATH_LEN,
    _fleet_speed,
)
from .constants import (
    CENTER, COMET_RADIUS, COMET_SPAWN_STEPS, LAUNCH_SURFACE_EPS,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS,
)

MIN_SOURCE_SHIPS: int = 10
# Horizons widened to match the measured flight-time distribution (test/flight_time_by_owner.py):
#   * Rotating-intercept scan was truncating reachable far targets — neutral p95=65,
#     enemy p95=47 flight ticks, but the scan only searched 30. Raised to 60 so far
#     rotating targets are no longer falsely marked blocked (un-hides expansion/attack candidates).
#   * Inbound / would-be-mine outcome sim: enemy->capture p95=32, p99=63, so a 30-tick
#     outcome horizon missed the slow-capture tail. Raised to 45 (the final/"30" buckets
#     now look 45 ticks ahead) to cover ~98% of captures.
# Validated by BC A/B vs sub_v52 on June data: OUT=60/IN=45 = 67.2% (43/64) win-rate,
# vs OUT=60/IN=30 = 50.0% (32/64) — the IN=45 slow-capture coverage is worth its cost.
# MAX_TICKS_OUT widening is ~free; MAX_TICKS_IN dominates encoder time (1.49x at IN=45).
MAX_TICKS_OUT: int = 60
MAX_TICKS_IN: int = 45

# ---------------------------------------------------------------------------
# Time-varying per-owner candidate quota (fractions of the K-1 real candidate
# slots). Linearly interpolated across a short transition window of game ticks
# [T0, T1]: step <= T0 uses the *_T0 mix, step >= T1 uses the *_T1 mix (held for
# the rest of the game). Heuristic rationale, confirmed on real games by
# test/target_owner_by_tick.py:
#   * NEUTRAL  dominates very early (the expert sends ~90% of fleets at neutral
#              planets in the first 10 ticks — pure land-grab) then collapses to
#              ~3-5% by tick ~50 as the reachable neutrals get claimed.
#   * ENEMY    grows once expansion is done (mid/late game is mostly combat).
#   * FRIENDLY grows (reinforcement/consolidation) and takes the remainder so the
#              three always sum to K-1.
# The transition window is intentionally short (T1=50) because the measured shift
# is fast: neutral targeting is essentially over by tick 50, the first comet step.
# ---------------------------------------------------------------------------
_Q_NEUTRAL_T0, _Q_NEUTRAL_T1 = 0.8, 0.2
_Q_ENEMY_T0,   _Q_ENEMY_T1   = 0.1, 0.3
T0, T1 = 0, 50    # quota transition window, in game ticks (then held at *_T1)
#  friendly = remainder  (T0: 0.1 -> T1: 0.5)


def self_feature_dim() -> int:
    return 25


def candidate_feature_dim() -> int:
    return 31


def global_feature_dim() -> int:
    return 12



# ---------------------------------------------------------------------------
# Comet future positions (for the moving-blocker scan)
# ---------------------------------------------------------------------------

N_COMET_SLOTS: int = N_COMET_GROUPS * COMET_PER_GROUP


def _comet_future_positions(
    state: EnvState, n_ticks: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact per-tick future comet positions over the next n_ticks.

    Comets advance one waypoint per tick along pre-computed paths, so the
    position at future tick t is just comet_paths[gi, sj, idx + t] — no
    approximation. Never-spawned groups are included too (spawn steps are
    deterministic): they enter at waypoint 0 on their spawn step, so a fleet
    whose flight crosses a path a comet group will occupy mid-flight is
    flagged. A comet is inactive outside [0, path_len) or once dead.

    Row 0 is the *current* tick (t=0), so consecutive rows give the
    waypoint-to-waypoint segment a comet sweeps during each future tick.

    Returns (fut_x, fut_y, fut_active), each [n_ticks + 1, N_COMET_SLOTS].
    """
    ticks = jnp.arange(0, n_ticks + 1, dtype=jnp.int32)             # [T+1]
    idx_g = state.comet_path_idx                                     # [N_CG]
    spawn = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)          # [N_CG]
    step  = state.step.astype(jnp.int32)
    # Future path index per group: spawned groups advance from their current
    # idx; never-spawned (-2) reach idx 0 on the step where step+t == spawn.
    base  = jnp.where(idx_g >= -1, idx_g, step - spawn)              # [N_CG]
    idx_t = base[None, :] + ticks[:, None]                           # [T, N_CG]

    xs, ys, acts = [], [], []
    for gi in range(N_COMET_GROUPS):
        it   = idx_t[:, gi]                                          # [T]
        it_c = jnp.clip(it, 0, MAX_PATH_LEN - 1)
        # The whole group is deactivated once its *shortest* member's path is
        # exhausted (_deactivate_comet_slots), so every member is active for
        # idx in [0, min_len) — not its own path length.
        g_act = (it >= 0) & (it < jnp.min(state.comet_path_lens[gi]))
        for sj in range(COMET_PER_GROUP):
            slot = COMET_SLOT_BASE + gi * COMET_PER_GROUP + sj
            # Spawned comets may already be dead (fleet combat); unspawned
            # ones must NOT be gated on p_alive (still False pre-spawn).
            act = g_act & jnp.where(idx_g[gi] >= -1, state.p_alive[slot], True)
            xs.append(state.comet_paths[gi, sj, it_c, 0])
            ys.append(state.comet_paths[gi, sj, it_c, 1])
            acts.append(act)
    return jnp.stack(xs, axis=1), jnp.stack(ys, axis=1), jnp.stack(acts, axis=1)


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
    moving_blockers: bool = False,  # static arg: also treat rotating planets as time-varying blockers
    comet_fut: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    # ^ optional (fut_x, fut_y, fut_active), each [MAX_TICKS_OUT+1, N_COMET_SLOTS]
    #   (row 0 = now) from _comet_future_positions — comets join the
    #   moving-blocker scan as swept waypoint-to-waypoint segments
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
        gap = spd * t - (jnp.sqrt(dx * dx + dy * dy) - (src_radius + LAUNCH_SURFACE_EPS))
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
        jnp.arange(1, MAX_TICKS_OUT + 1, dtype=jnp.int32),
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

    # --- Rotating case: single blocking check at predicted intercept point ---
    rot_end_x = jnp.where(use_rot, rot_tx, tgt_x)
    rot_end_y = jnp.where(use_rot, rot_ty, tgt_y)
    rot_blk   = _seg_blocked(rot_end_x, rot_end_y)

    # --- Optional: time-varying (rotating planet + comet) blocking -------
    # _seg_blocked above is exact for the sun and static planets over the
    # *whole* chord (flights can far outlast the scan horizon), so those stay
    # on the chord check. Movers get the engine's swept-segment test
    # (_swept_pair_hit_batch math), fleet segment vs blocker segment per tick,
    # evaluated for ALL candidate rays — the per-angle results merge into
    # blk_c/l/r and rot_blk BEFORE angle selection, so a mover blocking the
    # center ray no longer hides a clear tangent ray.
    # O(MAX_TICKS_OUT * MAX_PLANETS) per pair.
    if moving_blockers:
        spd_safe = jnp.maximum(spd, jnp.float32(1e-6))
        # 4 candidate rays: center / left / right tangents + rotating intercept.
        ang4 = jnp.stack([ang_c, ang_l, ang_r,
                          jnp.where(use_rot, rot_angle, center_ang)])      # [4]
        c_a4 = jnp.cos(ang4)
        s_a4 = jnp.sin(ang4)
        # Per-ray flight time in ticks (distance to first contact / speed).
        eta4 = jnp.stack([
            jnp.hypot(ec_x - src_x, ec_y - src_y) / spd_safe,
            jnp.hypot(el_x - src_x, el_y - src_y) / spd_safe,
            jnp.hypot(er_x - src_x, er_y - src_y) / spd_safe,
            jnp.where(use_rot, rot_tick, d_s / spd_safe),
        ])                                                                  # [4]
        launch = src_radius + LAUNCH_SURFACE_EPS

        # rotating, alive, non-comet planets, excluding src/tgt themselves.
        # (Comets are covered by the comet_fut waypoint lookup instead.)
        mv_mask = (
            all_p_alive
            & ~all_p_is_comet
            & ~is_static_p
            & (planet_slots != src_slot)
            & (planet_slots != tgt_slot)
        )
        r_sq    = all_p_r ** 2
        p_c     = jnp.cos(angular_velocity)
        p_s     = jnp.sin(angular_velocity)
        comet_r_sq = jnp.float32(COMET_RADIUS * COMET_RADIUS)

        def _swept_hit(fx0, fy0, fx1, fy1, pax, pay, pbx, pby, r2, mask):
            """[4] bool: min distance² between the fleet segment (per ray) and
            each blocker's segment over this tick is within the radius —
            same quadratic as the engine's _swept_pair_hit_batch."""
            d0x = fx0 - pax
            d0y = fy0 - pay
            dvx = (fx1 - fx0) - (pbx - pax)
            dvy = (fy1 - fy0) - (pby - pay)
            a = dvx * dvx + dvy * dvy
            b = 2.0 * (d0x * dvx + d0y * dvy)
            c = d0x * d0x + d0y * d0y
            t = jnp.clip(-b / (2.0 * a + 1e-30), 0.0, 1.0)
            min_d2 = c + (b + a * t) * t
            return jnp.any(mask & (min_d2 <= r2), axis=-1)

        def _mv_body(carry, xs):
            blk4, px, py = carry
            if comet_fut is not None:
                t, pcx, pcy, pca, ccx, ccy, cca = xs
            else:
                t = xs
            # advance planet positions one angular step about CENTER
            ddx = px - CENTER
            ddy = py - CENTER
            px2 = CENTER + p_c * ddx - p_s * ddy
            py2 = CENTER + p_s * ddx + p_c * ddy
            t_f = t.astype(jnp.float32)
            # fleet segment over this tick, per ray, from the launch surface
            d0  = launch + spd * (t_f - 1.0)
            d1  = launch + spd * t_f
            fx0 = (src_x + d0 * c_a4)[:, None]                       # [4, 1]
            fy0 = (src_y + d0 * s_a4)[:, None]
            fx1 = (src_x + d1 * c_a4)[:, None]
            fy1 = (src_y + d1 * s_a4)[:, None]
            # include the partial landing tick: segment t-1 → t flies iff
            # the fleet hasn't landed before this tick started
            active4 = (t_f - 1.0) < eta4                              # [4]
            hit4 = _swept_hit(fx0, fy0, fx1, fy1,
                              px[None, :], py[None, :],
                              px2[None, :], py2[None, :],
                              r_sq[None, :], mv_mask[None, :])
            if comet_fut is not None:
                # comet segment = waypoint(t-1) → waypoint(t); requiring BOTH
                # ends active skips the spawn tick, mirroring the engine's
                # check_col=False on first placement.
                hit4 = hit4 | _swept_hit(fx0, fy0, fx1, fy1,
                                         pcx[None, :], pcy[None, :],
                                         ccx[None, :], ccy[None, :],
                                         comet_r_sq, (pca & cca)[None, :])
            return (blk4 | (active4 & hit4), px2, py2), None

        ticks = jnp.arange(1, MAX_TICKS_OUT + 1, dtype=jnp.int32)
        if comet_fut is not None:
            cf_x, cf_y, cf_act = comet_fut  # [MAX_TICKS_OUT + 1, NC], row 0 = now
            scan_xs = (ticks,
                       cf_x[:-1], cf_y[:-1], cf_act[:-1],
                       cf_x[1:],  cf_y[1:],  cf_act[1:])
        else:
            scan_xs = ticks
        (mv_blk4, _, _), _ = jax.lax.scan(
            _mv_body,
            (jnp.zeros(4, dtype=jnp.bool_), all_p_x, all_p_y),
            scan_xs,
        )
        blk_c   = blk_c   | mv_blk4[0]
        blk_l   = blk_l   | mv_blk4[1]
        blk_r   = blk_r   | mv_blk4[2]
        rot_blk = rot_blk | mv_blk4[3]

    # Prefer center; fall back to left then right. Blocked iff all three fail.
    static_ang = jnp.where(~blk_c, ang_c, jnp.where(~blk_l, ang_l, ang_r))
    static_blk = blk_c & blk_l & blk_r

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
    cand_mask:       jnp.ndarray  # [MAX_SRC, K]    bool  (valid & ~blocked; slot 0 True)
    angles:          jnp.ndarray  # [MAX_SRC, K]    float32
    cand_feats:      jnp.ndarray  # [MAX_SRC, K, 30]
    # K-pool reconstruction metadata (lets a K_max shard be re-sliced to any K<=K_max).
    cand_cat:        jnp.ndarray  # [MAX_SRC, K]    int8  (0=enemy,1=neutral,2=friendly; -1=no-op/empty)
    cand_dist:       jnp.ndarray  # [MAX_SRC, K]    float32 (source→candidate distance, board units)
    global_feat:     jnp.ndarray  # [12]
    step:            jnp.ndarray  # []              int32


def _select_sources_and_candidates(state: EnvState, player, K: int, cand_select: str):
    """Source + K-candidate slot selection for a single game state (no features).

    Returns:
        src_slots      [MAX_SRC]      planet slots chosen as launch sources
        src_valid      [MAX_SRC]      which of those are real (own, alive, >=MIN ships)
        all_cand_slots [MAX_SRC, K]   candidate planet slots per source (slot 0 = -1 no-op)
        all_cand_valid [MAX_SRC, K]   pre-blocking pool membership (slot 0 = True)

    This is the only K-dependent part of the encoder. _make_env_encoder calls it and
    then additionally AND-masks candidates whose flight path is blocked; lightweight
    analysis tools call it directly to measure pure pool coverage. Sharing it keeps
    the two from drifting.
    """
    # Per-owner quota over the K-1 real candidate slots (slot 0 is the no-op),
    # linearly scheduled by episode progress (see _Q_* constants at module top):
    # neutral high early then decaying, enemy/friendly growing. Fractions are
    # rounded to integer slot counts; friendly takes the remainder so they always
    # sum to K-1. Quotas are soft — a category short of planets has its deficit
    # backfilled by the nearest planets of any owner (the overflow pool below).
    n_real = K - 1
    t_frac = jnp.clip((state.step.astype(jnp.float32) - T0) / jnp.float32(T1 - T0), 0.0, 1.0)
    f_neutral  = _Q_NEUTRAL_T0 + (_Q_NEUTRAL_T1 - _Q_NEUTRAL_T0) * t_frac
    f_enemy    = _Q_ENEMY_T0   + (_Q_ENEMY_T1   - _Q_ENEMY_T0)   * t_frac
    k_neutral  = jnp.maximum(jnp.round(f_neutral * n_real), 1)
    k_enemy    = jnp.maximum(jnp.round(f_enemy   * n_real), 1)
    k_friendly = jnp.maximum(n_real - k_neutral - k_enemy, 1)   # remainder
    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    INF = jnp.float32(1e30)
    OVERFLOW_BIAS = jnp.float32(1e6)   # >> board diagonal: in-quota sort before overflow

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
    # Rank-based so the quota can vary per tick: dynamic slice sizes ([:k] with a
    # traced k) aren't allowed in JAX, so instead of "take the k nearest of each
    # category" we mark a planet in-quota when its within-category distance rank
    # is below that category's (tick-dependent) quota. In-quota planets sort ahead
    # of everything else; the nearest leftovers backfill any deficit — identical
    # semantics to the old quota+overflow, but with a tick-scheduled k.
    def _select_cands(src_slot):
        src_x       = state.p_x[src_slot]
        src_y       = state.p_y[src_slot]
        dist_to_src = jnp.hypot(state.p_x - src_x, state.p_y - src_y)
        # Exclude comets: the static-angle fallback aims at the comet's
        # current position, which it will have left by arrival time, so
        # fleets typically miss and may continue into the sun.
        excl_mask   = state.p_alive & (planet_idx != src_slot) & ~state.p_is_comet

        is_enemy_planet    = excl_mask & (state.p_owner != -1) & (state.p_owner != player)
        is_neutral_planet  = excl_mask & (state.p_owner == -1)
        is_friendly_planet = excl_mask & (state.p_owner == player)

        def cat_rank(cat_mask):
            # rank[p] = number of same-category planets strictly nearer than p.
            closer = cat_mask[None, :] & (dist_to_src[None, :] < dist_to_src[:, None])
            return jnp.sum(closer, axis=1)            # [MAX_PLANETS] int

        in_quota = (
            (is_enemy_planet    & (cat_rank(is_enemy_planet)    < k_enemy))
            | (is_neutral_planet  & (cat_rank(is_neutral_planet)  < k_neutral))
            | (is_friendly_planet & (cat_rank(is_friendly_planet) < k_friendly))
        )

        # Sort key: in-quota planets first (by distance), then the nearest
        # out-of-quota planets (overflow backfill), then ineligible (INF).
        key = jnp.where(
            excl_mask,
            jnp.where(in_quota, dist_to_src, OVERFLOW_BIAS + dist_to_src),
            INF,
        )
        final_slots = jnp.argsort(key)[:K - 1]        # [K-1]
        final_valid = key[final_slots] < INF

        cand_slots = jnp.concatenate([jnp.array([-1], jnp.int32), final_slots])
        cand_valid = jnp.concatenate([jnp.array([True], bool),     final_valid])
        return cand_slots, cand_valid

    def _select_cands_knn(src_slot):
        """Plain K-nearest-neighbours: the K-1 closest planets of any owner.

        No per-owner quotas — just distance. Slot 0 is the no-op; comets and
        the source planet are still excluded (same reasons as the category
        selector). Empty trailing slots (fewer than K-1 planets) are invalid.
        """
        src_x       = state.p_x[src_slot]
        src_y       = state.p_y[src_slot]
        dist_to_src = jnp.hypot(state.p_x - src_x, state.p_y - src_y)
        excl_mask   = state.p_alive & (planet_idx != src_slot) & ~state.p_is_comet

        sorted_dist = jnp.where(excl_mask, dist_to_src, INF)
        nn_slots    = jnp.argsort(sorted_dist)[:K - 1]          # [K-1]
        nn_valid    = sorted_dist[nn_slots] < INF

        cand_slots = jnp.concatenate([jnp.array([-1], jnp.int32), nn_slots])
        cand_valid = jnp.concatenate([jnp.array([True], bool),    nn_valid])
        return cand_slots, cand_valid

    _select = _select_cands_knn if cand_select == "knn" else _select_cands
    all_cand_slots, all_cand_valid = jax.vmap(_select)(src_slots)  # [MAX_SRC, K]
    return src_slots, src_valid, all_cand_slots, all_cand_valid


def _make_env_encoder(K: int, env_cfg):
    """Return a JIT+vmap encoder: (batched_state, players[N int32]) -> _EncodedTurn[N,...].

    Call once at JaxVecEnv init; reuse every tick.
    """
    from .constants import MAX_ANGULAR_VELOCITY, BOARD_SIZE, MAX_PRODUCTION

    # Candidate selection strategy (resolved once, at encoder-build time, so the
    # branch is a Python-level choice and adds no per-trace cost).
    cand_select = getattr(env_cfg, "cand_select", "category")
    if cand_select not in ("category", "knn"):
        raise ValueError(
            f"Unknown env.cand_select={cand_select!r}; expected 'category' or 'knn'"
        )

    # When True, _find_angle_single also treats rotating planets and comets
    # (exact future waypoints, incl. scheduled future spawns) as time-varying
    # blockers (more accurate, more expensive). Default False = static-only.
    # ORBIT_MOVING_BLOCKERS=1 enables it for configs that lack the field
    # (e.g. checkpoints trained before the flag existed) — A/B at inference.
    moving_blockers = bool(getattr(env_cfg, "moving_blockers", False)) or bool(
        int(os.environ.get("ORBIT_MOVING_BLOCKERS", "0"))
    )

    board_size           = jnp.float32(BOARD_SIZE)
    max_ships            = jnp.float32(env_cfg.max_ships)
    max_production       = jnp.float32(MAX_PRODUCTION)
    max_planets          = jnp.float32(env_cfg.max_planets)
    episode_steps        = jnp.float32(env_cfg.episode_steps)
    max_angular_velocity = jnp.float32(MAX_ANGULAR_VELOCITY)

    def _encode_single(state: EnvState, player: jnp.ndarray) -> _EncodedTurn:
        # ── Source + candidate selection (shared with analysis tools) ─────
        src_slots, src_valid, all_cand_slots, all_cand_valid = (
            _select_sources_and_candidates(state, player, K, cand_select)
        )
        safe_slots = jnp.clip(all_cand_slots, 0, MAX_PLANETS - 1)  # [MAX_SRC, K]

        # ── Ship counts (per-candidate offensive estimate, unused for blocking) ──
        tgt_ships_2d  = state.p_ships[safe_slots]                             # [MAX_SRC, K]

        # ── Source ships (actual full fleet size) ───────────────────────────
        src_ships = jnp.round(state.p_ships[src_slots]).astype(jnp.int32)    # [MAX_SRC]

        # ── Exact aim angles + blocking for all [MAX_SRC, K] pairs ─────────
        # Fleet size = full src_ships so speed is known exactly. The iterative
        # intercept solver checks sun + static-planet blocking — no re-aim
        # needed after policy output.
        # Comet future positions are state-only (shared by every pair), so
        # compute them once here and close over them in the vmapped kernel.
        comet_fut = (
            _comet_future_positions(state, MAX_TICKS_OUT)
            if moving_blockers else None
        )

        def _row(src_slot, tgt_slots_row, n_ships):
            def _pair(tgt_slot):
                return _find_angle_single(
                    state.p_x[src_slot], state.p_y[src_slot], state.p_radius[src_slot], src_slot,
                    state.p_x[tgt_slot], state.p_y[tgt_slot], state.p_radius[tgt_slot], tgt_slot,
                    n_ships.astype(jnp.float32), state.angular_velocity,
                    state.p_x, state.p_y, state.p_radius, state.p_alive, state.p_is_comet,
                    moving_blockers,
                    comet_fut,
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

        # Per-candidate category + distance for K-pool reconstruction. cat marks
        # empties / no-op as -1 (slot 0 is always the no-op) so a K_max shard can be
        # re-sliced to a smaller K by re-running the quota+overflow selection offline.
        cand_cat = jnp.where(tgt_owner == -1, jnp.int8(1),
                     jnp.where(tgt_owner == player, jnp.int8(2), jnp.int8(0)))
        cand_cat = jnp.where(all_cand_valid, cand_cat, jnp.int8(-1)).at[:, 0].set(jnp.int8(-1))
        cand_dist = (tgt_dist / board_size).astype(jnp.float32).at[:, 0].set(jnp.float32(0.0))

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
            ), combat_ships  # ys: per-tick arrival totals by player [MAX_PLANETS, 4]

        _zeros_pf = jnp.zeros((MAX_PLANETS, MAX_FLEETS), dtype=jnp.bool_)
        (inbound_all, inbound_5_all, inbound_10_all, inbound_15_all, _, owner_final, _, owner_5_all, owner_10_all, owner_15_all, _, _), combat_ticks = jax.lax.scan(
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
            jnp.arange(1, MAX_TICKS_IN + 1, dtype=jnp.int32),
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
        would_be_mine_final = (owner_final[safe_slots]  == player).astype(jnp.float32)    # [MAX_SRC, K]

        # ── will_be_mine: exact counterfactual capture outcome ───────────────
        # If this source launches its full garrison now (landing at ceil(eta)),
        # is the target mine at the end of the MAX_TICKS_IN outcome sim? Replays
        # each candidate planet's per-tick recurrence (production + combat,
        # identical math to _sim_tick) with the hypothetical fleet added to its
        # arrival-tick combat. Exact w.r.t. the sim: planets only interact via
        # fleets already in flight, so the injection only perturbs the target's
        # own trajectory; arrivals (combat_ticks ys) are unchanged. Differs from
        # would_be_mine_*, which only simulates fleets already in flight.
        # Validated by test/test_will_be_mine.py against an independent numpy
        # reference. BC A/B vs sub_v52 (1-day stream, 64 rounds): 29.7% vs 12.5%
        # baseline; encoder cost ~1.00x GPU / 1.10x CPU.
        src_ships_f = src_ships.astype(jnp.float32)                           # [MAX_SRC]
        eta_safe  = jnp.where(jnp.isfinite(eta_ticks), eta_ticks, jnp.float32(MAX_TICKS_IN))
        arrival_t = jnp.clip(jnp.ceil(eta_safe), 1.0, jnp.float32(MAX_TICKS_IN))  # [MAX_SRC, K]
        tgt_prod  = state.p_production[safe_slots]                            # [MAX_SRC, K]
        player_oh = (jnp.arange(4, dtype=jnp.int32) == player).astype(jnp.float32)  # [4]
        # Hoist the per-pair gather out of the scan: one gather over the whole
        # time axis instead of one [MAX_PLANETS,4]→[MAX_SRC,K,4] gather per tick
        # (the in-loop gather dominated the scan cost).
        combat_pairs = combat_ticks[:, safe_slots]                            # [T, MAX_SRC, K, 4]

        def _cf_tick(carry, xs):
            g, o, po = carry                                                  # each [MAX_SRC, K]
            t, cs = xs                                                        # cs [MAX_SRC, K, 4]
            inject = (t.astype(jnp.float32) == arrival_t)
            cs = cs + jnp.where(inject, src_ships_f[:, None], 0.0)[..., None] * player_oh
            # Mirror _sim_tick: production, then combat resolution.
            prod_owner = jnp.where(g > 0, o, po)
            g = g + tgt_prod * (prod_owner != -1).astype(jnp.float32)
            has_combat = jnp.any(cs > 0, axis=-1)
            top_idx    = jnp.argmax(cs, axis=-1).astype(jnp.int32)
            top_ships  = jnp.max(cs, axis=-1)
            # second-largest of 4 = max with the argmax slot masked out
            second     = jnp.max(
                jnp.where(jnp.arange(4) == top_idx[..., None], -jnp.inf, cs),
                axis=-1,
            )
            survivor   = top_ships - second
            apply      = has_combat & (survivor > jnp.float32(0.0))
            friendly   = (top_idx == o)
            g = jnp.where(apply, jnp.where(friendly, g + survivor, g - survivor), g)
            flipped = apply & ~friendly & (g < jnp.float32(0.0))
            g  = jnp.abs(g)
            o  = jnp.where(flipped, top_idx, o)
            po = jnp.where(g > 0, o, po)
            return (g, o, po), None

        (_, owner_cf, _), _ = jax.lax.scan(
            _cf_tick,
            (
                _garrison_init[safe_slots],
                _owner_init[safe_slots],
                _prev_owner_init[safe_slots],
            ),
            (jnp.arange(1, MAX_TICKS_IN + 1, dtype=jnp.int32), combat_pairs),
            unroll=5,  # tiny body — partial unroll (full unroll explodes XLA compile time)
        )
        will_be_mine = (owner_cf == player).astype(jnp.float32)               # [MAX_SRC, K]

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
        self_my_inb_30  = ((inbound_src    & f_alive_s & is_my_fs).astype(jnp.float32)  * f_ships_s).sum(axis=1) / max_ships
        self_opp_inb_30 = ((inbound_src    & f_alive_s & is_opp_fs).astype(jnp.float32) * f_ships_s).sum(axis=1) / max_ships

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
            self_my_inb_30,                                                    # [13] friendly inbound (30-tick)
            self_opp_inb_30,                                                   # [14] enemy inbound (30-tick)
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
            (owner_final[src_slots]  == player).astype(jnp.float32),          # [24] will still be mine after 30 ticks
        ], axis=1)  # [MAX_SRC, 25]

        # ── Candidate features  [MAX_SRC, K, 31] ────────────────────────
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
            my_inbound_ships,                                                  # [18] friendly inbound (MAX_TICKS_IN-tick)
            opp_inbound_ships,                                                 # [19] enemy inbound (MAX_TICKS_IN-tick)
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
            would_be_mine_final,                                                  # [29] will be mine after MAX_TICKS_IN
            will_be_mine,                                                      # [30] counterfactual capture if I send now
        ], axis=2).at[:, 0, :].set(0.0)  # [MAX_SRC, K, 31]; no-op slot = zeros

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
            cand_cat        = cand_cat,
            cand_dist       = cand_dist,
            global_feat     = global_feat,
            step            = state.step,
        )

    return jax.jit(jax.vmap(_encode_single))
