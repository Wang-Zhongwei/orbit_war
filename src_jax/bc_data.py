"""
Behavior-cloning data pipeline: episode JSON → flat (features, labels) arrays.

Shards use the **K-pool** format: the candidate pool is encoded once at POOL_K and
re-sliced to any training K <= POOL_K offline via assemble_for_k(). Each stored
sample = one source-planet decision at one timestep:
  self_feats   : [SELF_DIM]
  global_feats : [GLOB_DIM]
  pool_feats   : [POOL_K, CAND_DIM]   candidate features at the pool size
  pool_slot    : [POOL_K] int32       planet slot per candidate
  pool_cat     : [POOL_K] int8        owner category 0/1/2 (enemy/neutral/friendly); -1 = no-op/empty
  pool_dist    : [POOL_K] float32     source→candidate distance (board units)
  pool_mask    : [POOL_K] bool        valid & ~blocked (slot 0 = no-op = True)
  target_slot  : int32                planet slot the expert hit, or -1 (no-op) — K-independent

assemble_for_k(samples, K) materializes the legacy fixed-K training dict
(self_feats / cand_feats / global_feats / cand_mask / target_idx) by re-running the
encoder's category-quota + overflow selection over the pool and mapping target_slot
to a candidate index. Storing the K-independent target_slot (vs a K-specific index)
is what lets one shard serve k8 / k12 / any K <= POOL_K.

Label extraction (trajectory method)
------------------------------------
The expert's target is found by *replaying the physics of the launched fleet*:
the fleet flies a straight line at the constant speed the engine assigns it, while
every planet keeps rotating tick-by-tick.  Per tick we run the exact swept-circle
collision the engine uses (physics._phase_move_fleets / _swept_pair_hit_batch) and
report the first planet the fleet's path actually sweeps into — that planet *is*
the target, by construction.  A fleet that leaves the board, falls into the sun, or
never arrives within the horizon is labelled as the no-op slot 0.

This replaces the earlier nearest-launch-angle heuristic, which was ambiguous when
two candidates shared a bearing and relied on an approximate intercept solver.

The hot path uses `simulate_episode_targets_jax`: a jitted, vmapped-over-fleets,
scanned-over-ticks simulator that flies every launch in an episode in one call on
the accelerator.
"""

from __future__ import annotations

import functools
import json
import multiprocessing as mp
import os
import time
import zipfile
from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .constants import (
    BOARD_SIZE,
    CENTER,
    LAUNCH_SURFACE_EPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .physics import (
    EnvState,
    MAX_BASE_PLANETS,
    MAX_FLEETS,
    MAX_PATH_LEN,
    MAX_PLANETS,
    N_COMET_GROUPS,
    _fleet_speed,
)

# How many ticks forward to fly a fleet before giving up and calling it a no-op.
# Covers essentially all genuine arrivals (see test/flight_time_dist.py); beyond
# this a fleet has almost certainly left the board or missed.
MAX_FLY_TICKS: int = 30

# Candidate pool size baked into shards. Shards are encoded once at POOL_K and
# re-sliced offline to any training K <= POOL_K via assemble_for_k(). POOL_K is
# the encoder's *total* K (no-op slot included): per-category quotas are K//3
# enemy / K//3 friendly / remainder neutral — see features._make_env_encoder.
POOL_K: int = 12


def _pad_size_for(n: int) -> int:
    """Return the smallest _PAD_SIZES bucket ≥ n, or n itself for very long episodes."""
    for s in _PAD_SIZES:
        if n <= s:
            return s
    return n  # >512 states: unique shape, one extra compile


# Ordered pad sizes used to bucket episode batch sizes.  The smallest size that
# fits the episode is chosen so encode_fn compiles at most len(_PAD_SIZES) times
# instead of once per distinct episode length (each XLA recompilation costs 2-5s).
# From the 2026-05-24 data: p50=89, p95=247, p99=471, max=481 states/episode.
_PAD_SIZES: tuple[int, ...] = (256, 512)


# ---------------------------------------------------------------------------
# Parsed data format
# ---------------------------------------------------------------------------

class HitKind(IntEnum):
    """Outcome of flying a launched fleet forward through the physics."""
    NONE      = 0   # never resolved within the horizon -> no-op label
    PLANET    = 1   # swept into a planet -> that planet is the target
    SUN       = 2   # fell into the sun -> no-op label
    OUT       = 3   # left the board -> no-op label


# ---------------------------------------------------------------------------
# Observation → EnvState
# ---------------------------------------------------------------------------

def obs_to_env_state(obs: dict, num_agents: int = 2) -> EnvState:
    """Reconstruct a JAX EnvState from a Kaggle episode observation dict.

    Planet format: [id, owner, x, y, radius, ships, production]
    Fleet format:  [id, owner, x, y, angle, from_planet_id, ships]
    """
    planets_raw = obs["planets"]
    fleets_raw = obs.get("fleets", [])
    initial_raw = obs.get("initial_planets", obs["planets"])
    comet_ids_active: set[int] = set(obs.get("comet_planet_ids", []))

    # Slot 0..n_base-1 = base planets, slots 44..63 = comets.
    # comet IDs start at n_base (= len(initial_planets)).
    n_base: int = len(initial_raw)
    init_pos = {int(p[0]): (float(p[2]), float(p[3])) for p in initial_raw}

    p_id         = np.full(MAX_PLANETS, -1,  dtype=np.int32)
    p_owner      = np.full(MAX_PLANETS, -1,  dtype=np.int32)
    p_x          = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_y          = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_radius     = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_ships      = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_production = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_alive      = np.zeros(MAX_PLANETS,     dtype=bool)
    p_init_x     = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_init_y     = np.zeros(MAX_PLANETS,     dtype=np.float32)
    p_is_comet   = np.zeros(MAX_PLANETS,     dtype=bool)

    for row in planets_raw:
        pid       = int(row[0])
        is_comet  = pid in comet_ids_active
        if is_comet:
            slot = MAX_BASE_PLANETS + (pid - n_base)
            if not (0 <= slot < MAX_PLANETS):
                continue
        else:
            slot = pid
            if not (0 <= slot < MAX_BASE_PLANETS):
                continue

        p_id[slot]         = pid
        p_owner[slot]      = int(row[1])
        x, y               = float(row[2]), float(row[3])
        p_x[slot]          = x
        p_y[slot]          = y
        p_radius[slot]     = float(row[4])
        p_ships[slot]      = float(row[5])
        p_production[slot] = float(row[6])
        p_alive[slot]      = True
        p_is_comet[slot]   = is_comet

        if is_comet:
            p_init_x[slot] = x   # comets use current pos (p_is_comet disables rotation)
            p_init_y[slot] = y
        else:
            ix, iy = init_pos.get(pid, (x, y))
            p_init_x[slot] = ix
            p_init_y[slot] = iy

    f_owner = np.zeros(MAX_FLEETS, dtype=np.int32)
    f_x     = np.zeros(MAX_FLEETS, dtype=np.float32)
    f_y     = np.zeros(MAX_FLEETS, dtype=np.float32)
    f_angle = np.zeros(MAX_FLEETS, dtype=np.float32)
    f_ships = np.zeros(MAX_FLEETS, dtype=np.float32)
    f_alive = np.zeros(MAX_FLEETS, dtype=bool)

    for i, fl in enumerate(fleets_raw[:MAX_FLEETS]):
        f_owner[i] = int(fl[1])
        f_x[i]     = float(fl[2])
        f_y[i]     = float(fl[3])
        f_angle[i] = float(fl[4])
        f_ships[i] = float(fl[6])
        f_alive[i] = True

    # Comet path data is used only by jax_step, not by the encoder.
    dummy_paths     = np.zeros((N_COMET_GROUPS, 4, MAX_PATH_LEN, 2), dtype=np.float32)
    dummy_path_lens = np.zeros((N_COMET_GROUPS, 4), dtype=np.int32)
    dummy_ships_cg  = np.zeros(N_COMET_GROUPS, dtype=np.float32)
    dummy_path_idx  = np.full(N_COMET_GROUPS, -2, dtype=np.int32)

    # Keep as numpy — caller does one batch jnp.asarray() after stacking all states.
    # Avoids ~20 JAX dispatch calls per state (was the dominant bottleneck in bc_encode).
    return EnvState(
        p_id          = p_id,
        p_owner       = p_owner,
        p_x           = p_x,
        p_y           = p_y,
        p_radius      = p_radius,
        p_ships       = p_ships,
        p_production  = p_production,
        p_alive       = p_alive,
        p_init_x      = p_init_x,
        p_init_y      = p_init_y,
        p_is_comet    = p_is_comet,
        f_owner       = f_owner,
        f_x           = f_x,
        f_y           = f_y,
        f_angle       = f_angle,
        f_ships       = f_ships,
        f_alive       = f_alive,
        comet_paths     = dummy_paths,
        comet_path_lens = dummy_path_lens,
        comet_ships     = dummy_ships_cg,
        comet_path_idx  = dummy_path_idx,
        step             = np.int32(obs["step"]),
        angular_velocity = np.float32(obs["angular_velocity"]),
        num_agents       = np.int32(num_agents),
        comet_id_base    = np.int32(n_base),
    )


# ---------------------------------------------------------------------------
# Physics replica — vectorized / JIT'd batch simulator (hot path)
# ---------------------------------------------------------------------------

def _planet_pos_at_jax(step, p_init_x, p_init_y, p_x, p_y, rotating, ang_vel):
    """[MAX_P] planet positions at absolute `step` (jax mirror of numpy version)."""
    dx = p_init_x - CENTER
    dy = p_init_y - CENTER
    ang = ang_vel * step.astype(jnp.float32)
    c, s = jnp.cos(ang), jnp.sin(ang)
    rot_x = CENTER + c * dx - s * dy
    rot_y = CENTER + s * dx + c * dy
    return jnp.where(rotating, rot_x, p_x), jnp.where(rotating, rot_y, p_y)


def _swept_entry_time_jax(fa_x, fa_y, fb_x, fb_y, pa_x, pa_y, pb_x, pb_y, r):
    """[MAX_P] per-planet entry time in [0,1] for this tick, or +inf if no hit.

    Mirrors physics._swept_pair_hit_batch but returns the entry time t1 so the
    caller can pick the *earliest* arrival within a tick rather than the engine's
    slot-index tie-break (an implementation artefact, not the intended target).
    """
    d0x = fa_x - pa_x
    d0y = fa_y - pa_y
    dvx = (fb_x - fa_x) - (pb_x - pa_x)
    dvy = (fb_y - fa_y) - (pb_y - pa_y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    inv = 1.0 / (2.0 * a + 1e-12)
    t1 = (-b - sq) * inv
    t2 = (-b + sq) * inv
    moving = a >= 1e-12
    hit_moving = moving & (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_static = (~moving) & (c <= 0.0)
    t_entry = jnp.where(hit_static, 0.0, jnp.clip(t1, 0.0, 1.0))
    return jnp.where(hit_moving | hit_static, t_entry, jnp.inf)


@functools.partial(jax.jit, static_argnames=("max_ticks",))
def _sim_fleets_jax(
    src_slot,                       # [F] int32
    fx0, fy0, cos_a, sin_a, speed,  # [F] float32
    launch_step,                    # [F] int32
    ang_vel,                        # [F] float32
    p_init_x, p_init_y,             # [F, MAX_P]
    p_x, p_y, p_radius,             # [F, MAX_P]
    p_alive, is_comet,              # [F, MAX_P] bool
    max_ticks: int = MAX_FLY_TICKS,
):
    """Fly all fleets forward. Returns hit_kind/hit_slot/hit_tick, each [F].

    hit_kind uses the HitKind int codes (0 NONE, 1 PLANET, 2 SUN, 3 OUT).
    """
    F, MAX_P = p_x.shape

    def per_fleet(src_slot, fx0, fy0, cos_a, sin_a, speed, launch_step, ang_vel,
                  p_init_x, p_init_y, p_x, p_y, p_radius, p_alive, is_comet):
        dx = p_init_x - CENTER
        dy = p_init_y - CENTER
        orb_r = jnp.sqrt(dx * dx + dy * dy)
        rotating = (orb_r + p_radius < ROTATION_RADIUS_LIMIT) & p_alive & ~is_comet
        is_src = jnp.arange(MAX_P) == src_slot

        px0, py0 = _planet_pos_at_jax(launch_step, p_init_x, p_init_y, p_x, p_y,
                                      rotating, ang_vel)

        def body(carry, tick):
            fx, fy, px_old, py_old, done, hk, hslot, htick = carry
            nfx = fx + cos_a * speed
            nfy = fy + sin_a * speed
            px_new, py_new = _planet_pos_at_jax(launch_step + tick, p_init_x, p_init_y,
                                                p_x, p_y, rotating, ang_vel)

            seg_dx, seg_dy = nfx - fx, nfy - fy
            seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy
            t_sun = jnp.where(seg_l2 > 0.0,
                              jnp.clip(((CENTER - fx) * seg_dx + (CENTER - fy) * seg_dy)
                                       / (seg_l2 + 1e-30), 0.0, 1.0), 0.0)
            proj_x = fx + t_sun * seg_dx
            proj_y = fy + t_sun * seg_dy
            sun_hit = (proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2 < SUN_RADIUS ** 2

            t_entry = _swept_entry_time_jax(fx, fy, nfx, nfy, px_old, py_old,
                                            px_new, py_new, p_radius)
            valid = p_alive & ~(is_src & (tick == 1))
            t_entry = jnp.where(valid, t_entry, jnp.inf)
            best_slot = jnp.argmin(t_entry).astype(jnp.int32)
            best_t = t_entry[best_slot]
            planet_hit = jnp.isfinite(best_t)

            oob = ~((0.0 <= nfx) & (nfx <= BOARD_SIZE) & (0.0 <= nfy) & (nfy <= BOARD_SIZE))

            planet_first = planet_hit & (~sun_hit | (best_t <= t_sun))
            ev_kind = jnp.where(planet_first, HitKind.PLANET,
                       jnp.where(sun_hit, HitKind.SUN,
                        jnp.where(oob, HitKind.OUT, HitKind.NONE))).astype(jnp.int32)
            ev_slot = jnp.where(planet_first, best_slot, -1).astype(jnp.int32)
            event = ev_kind != HitKind.NONE

            fire = event & ~done
            hk = jnp.where(fire, ev_kind, hk)
            hslot = jnp.where(fire, ev_slot, hslot)
            htick = jnp.where(fire, tick, htick)
            done = done | event
            nfx = jnp.where(done, fx, nfx)
            nfy = jnp.where(done, fy, nfy)
            return (nfx, nfy, px_new, py_new, done, hk, hslot, htick), None

        init = (fx0, fy0, px0, py0, jnp.bool_(False),
                jnp.int32(HitKind.NONE), jnp.int32(-1), jnp.int32(-1))
        carry, _ = jax.lax.scan(body, init, jnp.arange(1, max_ticks + 1, dtype=jnp.int32))
        _, _, _, _, _, hk, hslot, htick = carry
        return hk, hslot, htick

    return jax.vmap(per_fleet)(
        src_slot, fx0, fy0, cos_a, sin_a, speed, launch_step, ang_vel,
        p_init_x, p_init_y, p_x, p_y, p_radius, p_alive, is_comet)


def simulate_episode_targets_jax(
    flat_actions: list[tuple[int, int, float, float]],
    *,
    p_x, p_y, p_init_x, p_init_y, p_radius, p_alive, p_is_comet,  # [T, MAX_P]
    angular_velocity, step,                                       # [T]
    max_ticks: int = MAX_FLY_TICKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-simulate every launch in an episode in one jitted call.

    flat_actions: list of (timestep, src_slot, act_angle, act_ships) — already
    resolved to planet slots by the caller. Returns parallel numpy arrays
    (hit_kind, hit_slot, hit_tick), one entry per action.
    """
    if not flat_actions:
        z = np.zeros(0, np.int32)
        return z, z, z
    ts = np.array([a[0] for a in flat_actions], np.int32)
    ss = np.array([a[1] for a in flat_actions], np.int32)
    ang = np.array([a[2] for a in flat_actions], np.float32)
    shp = np.array([a[3] for a in flat_actions], np.float32)

    g = lambda arr: jnp.asarray(np.asarray(arr)[ts])  # [F, MAX_P]
    pix, piy = g(p_init_x), g(p_init_y)
    px, py, pr = g(p_x), g(p_y), g(p_radius)
    pal, pc = g(p_alive), g(p_is_comet)
    src = jnp.asarray(ss)
    ang_j = jnp.asarray(ang)
    speed = _fleet_speed(jnp.asarray(shp))
    cos_a, sin_a = jnp.cos(ang_j), jnp.sin(ang_j)
    launch_step = jnp.asarray(np.asarray(step)[ts])
    ang_vel = jnp.asarray(np.asarray(angular_velocity)[ts])

    src_px = jnp.take_along_axis(px, src[:, None], axis=1)[:, 0]
    src_py = jnp.take_along_axis(py, src[:, None], axis=1)[:, 0]
    src_pr = jnp.take_along_axis(pr, src[:, None], axis=1)[:, 0]
    fx0 = src_px + cos_a * (src_pr + LAUNCH_SURFACE_EPS)
    fy0 = src_py + sin_a * (src_pr + LAUNCH_SURFACE_EPS)

    hk, hslot, htick = _sim_fleets_jax(
        src, fx0, fy0, cos_a, sin_a, speed, launch_step, ang_vel,
        pix, piy, px, py, pr, pal, pc, max_ticks=max_ticks)
    return np.asarray(hk), np.asarray(hslot), np.asarray(htick)


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _resolve_src_idx(src_ids: np.ndarray, src_valid: np.ndarray, from_pid: int) -> int:
    """Map a source planet id to its encoder source index, or -1 if not modelled."""
    for j in range(len(src_ids)):
        if src_valid[j] and src_ids[j] == from_pid:
            return j
    return -1




def _add_noop_labels(labels: list[tuple[int, int]], src_valid_i: np.ndarray,
                     noop: int = 0) -> None:
    """Append no-op labels for valid sources the expert chose to hold.

    Without these, the policy only sees "given expert decided to launch, where to?"
    and collapses toward always-launch at inference. ``noop`` is the hold label —
    candidate index 0 for the legacy format, planet slot -1 for the K-pool format.
    """
    acted_src = {src_idx for src_idx, _ in labels}
    for j in range(len(src_valid_i)):
        if bool(src_valid_i[j]) and j not in acted_src:
            labels.append((j, noop))


def _flatten_labels(
    labels_per_state: list[list[tuple[int, int]]],
    src_valid: np.ndarray,   # [N, MAX_SRC]
    N: int,
    noop: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-state labels (with no-op holds) into parallel row-index arrays.

    Returns (state_idx, src_idx, target), one entry per output sample, in the
    same order the old per-row append loop produced: each state's launches first,
    then its idle-source no-op holds. Callers gather feature rows with a single
    fancy-index op (self_feats[state_idx, src_idx]) instead of slicing per row.
    ``noop`` is the hold label appended for idle sources (see _add_noop_labels).
    """
    state_idx: list[int] = []
    src_idx: list[int] = []
    tgt: list[int] = []
    for i in range(N):  # only real states, not padding
        labels = labels_per_state[i]
        _add_noop_labels(labels, src_valid[i], noop)
        for s, c in labels:
            state_idx.append(i)
            src_idx.append(s)
            tgt.append(c)
    n = len(state_idx)
    return (np.fromiter(state_idx, np.intp, n),
            np.fromiter(src_idx, np.intp, n),
            np.fromiter(tgt, np.int32, n))


# ---------------------------------------------------------------------------
# Episode → flat sample arrays
# ---------------------------------------------------------------------------

def _collect_states_actions(episode: dict):
    """Shared state/action collection used by both sample functions.

    Replay timing: steps[i]['observation'] is the game state AFTER steps[i]['action']
    has been processed.  The agent chose steps[i]['action'] based on the state they
    could see, which was steps[i-1]['observation'].  We therefore pair:
        state   = steps[i-1][agent]['observation']
        action  = steps[i][agent]['action']
    """
    steps = episode.get("steps", [])
    num_agents = len(steps[0]) if steps else 2
    batch_states: list[EnvState] = []
    batch_players: list[int]     = []
    batch_actions: list[list]    = []

    for i in range(1, len(steps)):
        for agent_idx, agent_data in enumerate(steps[i]):
            action = agent_data.get("action", [])
            if not action:
                continue
            if agent_data.get("status") == "DONE":
                continue
            prev_obs = steps[i - 1][agent_idx]["observation"]
            try:
                state = obs_to_env_state(prev_obs, num_agents)
            except Exception:
                continue
            batch_states.append(state)
            batch_players.append(int(prev_obs["player"]))
            batch_actions.append(action)

    return batch_states, batch_players, batch_actions


def _episode_target_slots(
    *,
    src_ids: np.ndarray,        # [N, MAX_SRC]
    src_valid: np.ndarray,      # [N, MAX_SRC]
    src_slots: np.ndarray,      # [N, MAX_SRC]
    p_x: np.ndarray, p_y: np.ndarray,            # [pad, MAX_P]
    p_init_x: np.ndarray, p_init_y: np.ndarray,  # [pad, MAX_P]
    p_radius: np.ndarray,                         # [pad, MAX_P]
    p_alive: np.ndarray, p_is_comet: np.ndarray,  # [pad, MAX_P]
    angular_velocity: np.ndarray,                 # [pad]
    step: np.ndarray,                             # [pad]
    batch_actions: list[list],
    N: int,
) -> list[list[tuple[int, int]]]:
    """Trajectory-label every launch as (src_idx, target_slot).

    Returns labels_per_state[i] = list of (src_idx, target_slot) where target_slot
    is the planet slot the fleet sweeps into, or -1 for a non-PLANET outcome / no-op.
    K-independent: the K-pool shard stores these slots so they can be re-sliced to
    any K offline via assemble_for_k.
    """
    flat: list[tuple[int, int, float, float]] = []
    meta: list[tuple[int, int]] = []  # (timestep, src_idx)
    for i in range(N):
        for act in batch_actions[i]:
            src_idx = _resolve_src_idx(src_ids[i], src_valid[i], int(act[0]))
            if src_idx == -1:
                continue
            flat.append((i, int(src_slots[i][src_idx]), float(act[1]), float(act[2])))
            meta.append((i, src_idx))

    labels_per_state: list[list[tuple[int, int]]] = [[] for _ in range(N)]
    if not flat:
        return labels_per_state

    hit_kind, hit_slot, _ = simulate_episode_targets_jax(
        flat, p_x=p_x, p_y=p_y, p_init_x=p_init_x, p_init_y=p_init_y,
        p_radius=p_radius, p_alive=p_alive, p_is_comet=p_is_comet,
        angular_velocity=angular_velocity, step=step)
    tslot = np.where(np.asarray(hit_kind) == int(HitKind.PLANET),
                     np.asarray(hit_slot), -1).astype(np.int32)
    for n, (i, src_idx) in enumerate(meta):
        labels_per_state[i].append((src_idx, int(tslot[n])))
    return labels_per_state


def episode_to_samples(
    episode: dict,
    encode_fn,
) -> dict[str, np.ndarray] | None:
    """Convert one episode JSON to flat numpy arrays of BC training samples.

    Emits the **K-pool** shard format: the candidate features/metadata are stored
    at the encoder's POOL_K, and the target is a K-independent planet slot. Call
    assemble_for_k(samples, K) at train time to materialize cand_feats / cand_mask
    / target_idx for any K <= POOL_K. encode_fn must therefore be built at POOL_K.

    Returns None if no valid samples were found.
    """
    batch_states, batch_players, batch_actions = _collect_states_actions(episode)
    if not batch_states:
        return None

    N = len(batch_states)
    # Pad to the smallest _PAD_SIZES bucket that fits N so encode_fn reuses a
    # cached XLA kernel instead of recompiling for every distinct episode length.
    pad_target = _pad_size_for(N)
    pad_n = pad_target - N
    all_states  = batch_states  + [batch_states[-1]]  * pad_n
    all_players = batch_players + [batch_players[-1]] * pad_n

    # np.stack first (fast C loop), then single jnp.asarray per field — avoids
    # the ~8000 JAX dispatch calls that jnp.stack(list_of_jax_arrays) would cause.
    batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *all_states)
    players_arr = jnp.array(all_players, dtype=jnp.int32)
    enc = encode_fn(batched, players_arr)
    enc_np     = jax.tree_util.tree_map(np.asarray, enc)
    batched_np = jax.tree_util.tree_map(np.asarray, batched)

    labels_per_state = _episode_target_slots(
        src_ids=enc_np.src_ids, src_valid=enc_np.src_valid, src_slots=enc_np.src_slots,
        p_x=batched_np.p_x, p_y=batched_np.p_y,
        p_init_x=batched_np.p_init_x, p_init_y=batched_np.p_init_y,
        p_radius=batched_np.p_radius, p_alive=batched_np.p_alive,
        p_is_comet=batched_np.p_is_comet,
        angular_velocity=batched_np.angular_velocity, step=batched_np.step,
        batch_actions=batch_actions, N=N)

    si, sj, target_slot = _flatten_labels(labels_per_state, enc_np.src_valid, N, noop=-1)
    if si.size == 0:
        return None

    # Single fancy-index gather per field instead of per-row slice + np.stack.
    return {
        "self_feats":   enc_np.self_feats[si, sj].astype(np.float32),
        "global_feats": enc_np.global_feat[si].astype(np.float32),
        "pool_feats":   enc_np.cand_feats[si, sj].astype(np.float32),     # [n, POOL_K, CAND_DIM]
        "pool_slot":    enc_np.cand_slots_safe[si, sj].astype(np.int32),  # planet slot per cand
        "pool_cat":     enc_np.cand_cat[si, sj].astype(np.int8),          # 0/1/2 owner cat, -1 empty
        "pool_dist":    enc_np.cand_dist[si, sj].astype(np.float32),      # source→cand distance
        "pool_mask":    enc_np.cand_mask[si, sj],                          # valid & ~blocked
        "target_slot":  target_slot.astype(np.int32),                     # planet slot, -1 = no-op
    }


# ---------------------------------------------------------------------------
# Dataset loader — lazy iterator over episode files
# ---------------------------------------------------------------------------

def _count_episode_jsons(episodes_dir: Path) -> int:
    """Return the number of JSON episode files without loading them."""
    zips = sorted(episodes_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as zf:
            return sum(1 for n in zf.namelist() if n.endswith(".json"))
    return sum(1 for _ in episodes_dir.glob("*.json"))


def _iter_episode_jsons(episodes_dir: Path):
    """Yield (name, episode_dict) from either a zip archive or loose .json files."""
    zips = sorted(episodes_dir.glob("*.zip"))
    if zips:
        zf = zipfile.ZipFile(zips[0])
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        for name in names:
            try:
                with zf.open(name) as f:
                    yield name, json.load(f)
            except Exception:
                continue
    else:
        for path in sorted(episodes_dir.glob("*.json")):
            try:
                with open(path) as f:
                    yield path.name, json.load(f)
            except Exception:
                continue


def _filter_winner(episode: dict, winner: int) -> dict:
    """Return episode with only the winner's actions kept, loser gets action=[]."""
    import copy
    ep = copy.deepcopy(episode)
    for step in ep["steps"]:
        for agent_idx, agent_data in enumerate(step):
            if agent_idx != winner:
                agent_data["action"] = []
    return ep


# ---------------------------------------------------------------------------
# Concatenate sample dicts
# ---------------------------------------------------------------------------

def concat_samples(sample_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sample_list[0].keys()
    return {k: np.concatenate([s[k] for s in sample_list], axis=0) for k in keys}


# ---------------------------------------------------------------------------
# Shard encoding + discovery
# ---------------------------------------------------------------------------

SHARD_DIR_NAME = "shards"

# Module-level state populated by each worker process via Pool initializer.
_worker_encode_fn = None


def _worker_init(encode_fn_factory, factory_args) -> None:
    """Called once per worker process. Builds encode_fn locally (avoids pickling JAX fn)."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    global _worker_encode_fn
    _worker_encode_fn = encode_fn_factory(*factory_args)


def _worker_encode_episode(args: tuple) -> dict | None:
    """Top-level worker task: encode one episode dict → samples dict (numpy arrays)."""
    episode, winner_only = args
    if winner_only:
        rewards  = episode.get("rewards", [])
        statuses = episode.get("statuses", [])
        valid_rewards = [r for r in rewards if r is not None]
        if not valid_rewards:
            return None
        best = max(valid_rewards)
        winning_player = next(
            (i for i, (r, s) in enumerate(zip(rewards, statuses))
             if r == best and s == "DONE"),
            None,
        )
        if winning_player is None:
            return None
        episode = _filter_winner(episode, winning_player)
    return episode_to_samples(episode, _worker_encode_fn)


def encode_to_shards(
    episodes_dir: Path,
    encode_fn,
    *,
    episodes_per_shard: int = 100,
    num_shards: int | None = None,
    winner_only: bool = False,
    force: bool = False,
    workers: int = -1,
    encode_fn_factory=None,
    encode_fn_factory_args: tuple = (),
) -> Path:
    """Encode all episodes in episodes_dir and write sharded .npz files.

    Skips if shards already exist unless force=True. Returns the shard directory.
    num_shards limits how many shards are written (useful for quick tests).

    workers=-1 uses all available CPUs via multiprocessing. workers=1 (or 0) runs
    single-threaded. For workers>1 you must pass encode_fn_factory and
    encode_fn_factory_args so each worker can re-build the (non-picklable) JAX
    encode_fn locally.
    """
    shard_dir = episodes_dir / SHARD_DIR_NAME
    if not force and any(shard_dir.glob("*.npz")):
        print(f"Shards already exist in {shard_dir} — skipping (use --force to re-encode)")
        return shard_dir

    n_workers = os.cpu_count() if workers == -1 else workers
    use_mp = n_workers is not None and n_workers > 1 and encode_fn_factory is not None

    # Single-threaded path: build encode_fn locally if not provided.
    if not use_mp and encode_fn is None:
        if encode_fn_factory is None:
            raise ValueError("encode_fn or encode_fn_factory must be provided")
        encode_fn = encode_fn_factory(*encode_fn_factory_args)

    total_files = _count_episode_jsons(episodes_dir)
    print(f"Found {total_files} episode files in {episodes_dir}"
          + (f" | workers={n_workers}" if use_mp else ""))
    if winner_only:
        print("  winner_only=True: encoding only winning agent's actions")

    shard_dir.mkdir(exist_ok=True)
    buffer: list[dict[str, np.ndarray]] = []
    shard_idx = n_episodes = n_samples = n_skipped = 0
    t0 = time.time()

    def _maybe_flush(force_flush: bool = False) -> None:
        nonlocal buffer, shard_idx, n_samples
        if not buffer:
            return
        if force_flush or len(buffer) >= episodes_per_shard:
            _flush_shard(shard_dir, shard_idx, buffer)
            elapsed = time.time() - t0
            print(f"  shard {shard_idx:04d} written — {n_episodes}/{total_files} episodes, "
                  f"{n_samples} samples ({elapsed:.1f}s, "
                  f"{n_episodes / elapsed:.1f} ep/s)")
            buffer = []
            shard_idx += 1

    def _process_result(samples) -> bool:
        """Add samples to buffer; return True if num_shards limit hit."""
        nonlocal n_episodes, n_samples, n_skipped
        if samples is None:
            n_skipped += 1
            return False
        buffer.append(samples)
        n_episodes += 1
        n_samples  += len(samples["target_slot"])
        if len(buffer) >= episodes_per_shard:
            _maybe_flush()
            if num_shards is not None and shard_idx >= num_shards:
                return True
        elif n_episodes % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {n_episodes}/{total_files} episodes, {n_samples} samples "
                  f"({elapsed:.1f}s, {n_episodes / elapsed:.1f} ep/s)")
        return False

    if use_mp:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(encode_fn_factory, encode_fn_factory_args),
        )
        episodes_iter = (
            (ep, winner_only)
            for _name, ep in _iter_episode_jsons(episodes_dir)
            if isinstance(ep, dict) and "steps" in ep
        )
        done = False
        try:
            for samples in pool.imap_unordered(_worker_encode_episode, episodes_iter,
                                               chunksize=4):
                if _process_result(samples):
                    done = True
                    break
        finally:
            pool.terminate()
            pool.join()
        if not done and buffer and (num_shards is None or shard_idx < num_shards):
            _maybe_flush(force_flush=True)
    else:
        for _name, episode in _iter_episode_jsons(episodes_dir):
            if not isinstance(episode, dict) or "steps" not in episode:
                n_skipped += 1
                continue
            samples = episode_to_samples(episode, encode_fn)
            if _process_result(samples):
                break
        else:
            if buffer and (num_shards is None or shard_idx < num_shards):
                _maybe_flush(force_flush=True)

    elapsed = time.time() - t0
    avg_samples = n_samples / n_episodes if n_episodes else 0
    skipped_msg = f", {n_skipped} skipped" if n_skipped else ""
    print(
        f"Done: {n_episodes} episodes → {shard_idx} shards, "
        f"{n_samples} samples ({avg_samples:.1f} samples/ep{skipped_msg}) "
        f"in {elapsed:.1f}s ({n_episodes / elapsed:.1f} ep/s)"
    )
    print(f"Shards written to: {shard_dir}")
    return shard_dir


def _flush_shard(shard_dir: Path, idx: int, buffer: list[dict[str, np.ndarray]]) -> None:
    np.savez(shard_dir / f"shard_{idx:04d}.npz",
             n_episodes=np.array(len(buffer)),
             **concat_samples(buffer))


def find_shards(episodes_dirs: list[Path]) -> list[Path]:
    """Return all shard .npz paths across the given date directories."""
    paths: list[Path] = []
    for d in episodes_dirs:
        shard_dir = d / SHARD_DIR_NAME
        if not shard_dir.is_dir() or not any(shard_dir.glob("*.npz")):
            raise FileNotFoundError(
                f"No shards found in {shard_dir}. "
                f"Run: python -m src_jax.bc_encode --episodes_dir {d}"
            )
        paths.extend(sorted(shard_dir.glob("*.npz")))
    return paths


def shard_episode_counts(shard_paths: list[Path], episodes_per_shard_fallback: int = 100) -> list[int]:
    """Return the number of episodes in each shard.

    Reads the n_episodes scalar saved by _flush_shard. Falls back to
    episodes_per_shard_fallback for shards encoded before that field was added.
    """
    counts = []
    for p in shard_paths:
        d = np.load(p)
        if "n_episodes" in d:
            counts.append(int(d["n_episodes"]))
        else:
            counts.append(episodes_per_shard_fallback)
        d.close()
    return counts


# ---------------------------------------------------------------------------
# K-pool → fixed-K materialization (train-time)
# ---------------------------------------------------------------------------

def is_pool_shard(data) -> bool:
    """True if a loaded shard uses the K-pool format (vs the legacy fixed-K format)."""
    return "pool_feats" in data


def _k_quota(K: int) -> tuple[int, int, int]:
    """(K_enemy, K_neutral, K_friendly) for total K — mirrors _make_env_encoder."""
    K_e = K // 3
    K_f = K // 3
    K_n = K - 1 - K_e - K_f
    return K_e, K_n, K_f


def assemble_for_k(data, K: int) -> dict[str, np.ndarray]:
    """Re-slice a K-pool shard to a fixed candidate count K (<= POOL_K).

    Reproduces _make_env_encoder's category-quota + overflow candidate selection
    offline, over the stored pool, then maps the K-independent ``target_slot`` to a
    candidate index. Returns the legacy fixed-K training dict: self_feats /
    cand_feats / global_feats / cand_mask / target_idx.

    Correctness rests on set-monotonicity of the selection (the K-pool is a superset
    of any smaller-K pool); see test/test_kmax_pool_parity.py which verifies parity
    against a live _make_env_encoder(K) on real states.
    """
    pf    = data["pool_feats"]                       # [N, P, D]
    pslot = data["pool_slot"]                         # [N, P]
    pcat  = np.asarray(data["pool_cat"], np.int64)    # [N, P]
    pdist = np.asarray(data["pool_dist"], np.float32) # [N, P]
    pmask = np.asarray(data["pool_mask"], bool)       # [N, P]
    tslot = np.asarray(data["target_slot"])           # [N]
    N, P, _ = pf.shape
    if K > P:
        raise ValueError(f"assemble_for_k: K={K} exceeds stored pool size {P}")

    K_e, K_n, K_f = _k_quota(K)
    INF = np.float32(1e30)
    rows = np.arange(N)

    def _cat_pick(c: int, kc: int):
        """The kc nearest pool positions of category c (padded with junk; valid flags)."""
        if kc == 0:
            return np.zeros((N, 0), np.intp), np.zeros((N, 0), bool)
        key   = np.where(pcat == c, pdist, INF)               # [N, P]
        order = np.argsort(key, axis=1, kind="stable")        # nearest-first
        pos   = order[:, :kc].astype(np.intp)
        valid = np.take_along_axis(key, pos, axis=1) < INF
        return pos, valid

    pe, ve = _cat_pick(0, K_e)
    pn, vn = _cat_pick(1, K_n)
    pfr, vf = _cat_pick(2, K_f)
    per_pos = np.concatenate([pe, pn, pfr], axis=1)            # [N, K-1] (enemy,neutral,friendly)
    per_val = np.concatenate([ve, vn, vf], axis=1)
    Km1 = per_pos.shape[1]

    # Overflow: fill short quota slots with the nearest unclaimed real candidates,
    # mirroring the encoder's exclusive-prefix-sum slot assignment.
    claimed = np.zeros((N, P), bool)
    if Km1:
        ii = np.broadcast_to(rows[:, None], per_pos.shape)[per_val]
        jj = per_pos[per_val]
        claimed[ii, jj] = True
    real     = pcat >= 0
    ovfl_key = np.where(real & ~claimed, pdist, INF)
    ovfl_ord = np.argsort(ovfl_key, axis=1, kind="stable")    # [N, P] nearest unclaimed first
    ovfl_oks = np.take_along_axis(ovfl_key, ovfl_ord, axis=1) < INF
    inv      = (~per_val).astype(np.int64)
    ovfl_idx = np.minimum(np.cumsum(inv, axis=1) - inv, max(Km1 - 1, 0))
    ovfl_pos = np.take_along_axis(ovfl_ord, ovfl_idx, axis=1) if Km1 else per_pos
    ovfl_ok  = np.take_along_axis(ovfl_oks, ovfl_idx, axis=1) if Km1 else per_val
    final_pos = np.where(per_val, per_pos, ovfl_pos).astype(np.intp)
    final_ok  = np.where(per_val, True, ovfl_ok)

    # Prepend the no-op slot (pool position 0) and gather.
    pos = np.concatenate([np.zeros((N, 1), np.intp), final_pos], axis=1)   # [N, K]
    cand_feats = np.take_along_axis(pf, pos[:, :, None], axis=1).astype(np.float32)
    cand_mask  = np.take_along_axis(pmask, pos, axis=1)
    cand_mask &= np.concatenate([np.ones((N, 1), bool), final_ok], axis=1)
    cand_mask[:, 0] = True

    # Map the planet-slot target to a candidate index (require usable mask; k=0 = no-op).
    sel_slot = np.take_along_axis(pslot, pos, axis=1)          # [N, K]
    match = (sel_slot == tslot[:, None]) & cand_mask
    match[:, 0] = False
    has = match.any(axis=1)
    target_idx = np.where(has, match.argmax(axis=1), 0).astype(np.int32)

    return {
        "self_feats":   data["self_feats"],
        "cand_feats":   cand_feats,
        "global_feats": data["global_feats"],
        "cand_mask":    cand_mask,
        "target_idx":   target_idx,
    }
