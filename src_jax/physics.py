"""
JAX-accelerated Orbit Wars physics.

Public API
----------
JaxState          — fixed-size NamedTuple of JAX arrays (vmap-able)
init_jax_state    — Python init: seed → JaxState  (called once per episode)
jax_step          — JIT-compiled, vmap-able step function
jax_step_jit      — pre-JIT-compiled version

Action format for jax_step
---------------------------
    actions : float32[N_AGENTS, MAX_ACTIONS_PA, 3]
              [..., 0] = from_planet_id  (-1 = no action / padding)
              [..., 1] = angle (radians)
              [..., 2] = ships (int-valued float)

The function is vmappable over a leading batch axis on *both* state and actions.

Comet design note
-----------------
Comet groups spawn at steps 50/150/250/350/450.  Each group's path is
pre-computed at episode init because:
  - Paths depend only on seed, initial_planets, angular_velocity, spawn_step.
  - Each group expires ≤40 ticks after spawning, so by the next spawn tick
    the previous group is gone → comet_planet_ids is always empty at spawn
    time → the pre-computed paths are identical to the dynamic ones.
"""

from __future__ import annotations

import random as _stdlib_random
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    EPISODE_STEPS,
    MAX_ANGULAR_VELOCITY,
    MAX_SHIP_SPEED,
    MIN_ANGULAR_VELOCITY,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .generation import generate_comet_paths, generate_planets

# ---------------------------------------------------------------------------
# Array-size constants
# ---------------------------------------------------------------------------

MAX_BASE_PLANETS: int = 44    # game generates 20-40 base planets
N_COMET_GROUPS: int = 5
COMET_PER_GROUP: int = 4
MAX_PLANETS: int = MAX_BASE_PLANETS + N_COMET_GROUPS * COMET_PER_GROUP  # 64
MAX_FLEETS: int = 256
MAX_PATH_LEN: int = 42        # comet visible path ≤40 waypoints
MAX_ACTIONS_PA: int = 24      # max actions per agent per tick
N_AGENTS: int = 4             # maximum players supported (JIT array dimension)

# Comet group i occupies planet slots COMET_SLOT_BASE + i*4 .. +3
COMET_SLOT_BASE: int = MAX_BASE_PLANETS


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Fixed-size game state as JAX arrays.  All shapes written [dim] or [a,b,...]."""

    # ---- Planets [MAX_PLANETS] ----
    p_id:          jnp.ndarray   # int32   game planet ID
    p_owner:       jnp.ndarray   # int32   -1=neutral, 0..N-1=player
    p_x:           jnp.ndarray   # float32
    p_y:           jnp.ndarray   # float32
    p_radius:      jnp.ndarray   # float32
    p_ships:       jnp.ndarray   # float32
    p_production:  jnp.ndarray   # float32
    p_alive:       jnp.ndarray   # bool    slot occupied?
    p_init_x:      jnp.ndarray   # float32 x at step-0 (for rotation)
    p_init_y:      jnp.ndarray   # float32 y at step-0
    p_is_comet:    jnp.ndarray   # bool

    # ---- Fleets [MAX_FLEETS] ----
    f_owner:  jnp.ndarray   # int32
    f_x:      jnp.ndarray   # float32
    f_y:      jnp.ndarray   # float32
    f_angle:  jnp.ndarray   # float32
    f_ships:  jnp.ndarray   # float32
    f_alive:  jnp.ndarray   # bool

    # ---- Pre-computed comet data ----
    # paths[i, j, k, 0:2] = [x, y] of waypoint k for planet j of group i
    comet_paths:     jnp.ndarray   # float32 [N_CG, 4, MAX_PL, 2]
    comet_path_lens: jnp.ndarray   # int32   [N_CG, 4] actual visible length
    comet_ships:     jnp.ndarray   # float32 [N_CG]

    # ---- Dynamic comet state ----
    comet_path_idx: jnp.ndarray   # int32 [N_CG]  -2=never spawned, -1=spawned/pre-first-step, 0+=active

    # ---- Scalar episode state ----
    step:             jnp.ndarray   # int32
    angular_velocity: jnp.ndarray   # float32
    num_agents:       jnp.ndarray   # int32 (static value, kept for termination)
    comet_id_base:    jnp.ndarray   # int32: max_base_planet_id + 1; comets reuse these IDs


# ---------------------------------------------------------------------------
# Helpers — pure JAX math
# ---------------------------------------------------------------------------

def _fleet_speed(ships: jnp.ndarray) -> jnp.ndarray:
    s = jnp.maximum(ships, 1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * (jnp.log(s) / jnp.log(1000.0)) ** 1.5


def _swept_pair_hit_batch(
    fa_x: jnp.ndarray, fa_y: jnp.ndarray,   # [MAX_F]  fleet old pos
    fb_x: jnp.ndarray, fb_y: jnp.ndarray,   # [MAX_F]  fleet new pos
    pa_x: jnp.ndarray, pa_y: jnp.ndarray,   # [MAX_P]  planet old pos
    pb_x: jnp.ndarray, pb_y: jnp.ndarray,   # [MAX_P]  planet new pos
    p_radius: jnp.ndarray,                   # [MAX_P]
) -> jnp.ndarray:
    """Return [MAX_F, MAX_P] bool: True iff fleet i sweeps into planet j."""
    # Broadcast: fleets → [MAX_F,1], planets → [1,MAX_P]
    fa_x = fa_x[:, None];  fa_y = fa_y[:, None]
    fb_x = fb_x[:, None];  fb_y = fb_y[:, None]
    pa_x = pa_x[None, :];  pa_y = pa_y[None, :]
    pb_x = pb_x[None, :];  pb_y = pb_y[None, :]
    r    = p_radius[None, :]

    d0x = fa_x - pa_x
    d0y = fa_y - pa_y
    dvx = (fb_x - fa_x) - (pb_x - pa_x)
    dvy = (fb_y - fa_y) - (pb_y - pa_y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    # a ≈ 0 → relative velocity ≈ 0 → static check c ≤ 0
    disc = b * b - 4.0 * a * c
    sq   = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1   = (-b - sq) / (2.0 * a + 1e-12)
    t2   = (-b + sq) / (2.0 * a + 1e-12)

    moving = a >= 1e-12
    return jnp.where(
        moving,
        (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0),
        c <= 0.0,
    )

# ---------------------------------------------------------------------------
# Initialisation — Python side (called once per episode)
# ---------------------------------------------------------------------------

def init_jax_state(seed: int | None = None, num_agents: int = 2) -> EnvState:
    """Build a fresh JaxState from a seed.

    Calls Python generation code once, then converts everything to fixed-size
    JAX arrays.  The comet paths for all 5 spawn steps are pre-computed here.
    """
    if seed is None:
        seed = _stdlib_random.randrange(2 ** 31)
    rng = _stdlib_random.Random(seed)
    angular_velocity = rng.uniform(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
    raw_planets = generate_planets(rng)
    initial_planets = [p[:] for p in raw_planets]

    # Assign home planets
    num_groups = len(raw_planets) // 4
    if num_groups > 0:
        home_group = rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_agents == 2:
            raw_planets[base][1] = 0
            raw_planets[base][5] = 10
            raw_planets[base + 3][1] = 1
            raw_planets[base + 3][5] = 10
        elif num_agents == 4:
            for j in range(4):
                raw_planets[base + j][1] = j
                raw_planets[base + j][5] = 10

    n = len(raw_planets)
    assert n <= MAX_BASE_PLANETS, f"Too many base planets: {n} > {MAX_BASE_PLANETS}"

    # ---- Planet arrays ----
    pad = MAX_PLANETS - n
    p_id = jnp.asarray([p[0] for p in raw_planets] + [-1] * pad, dtype=jnp.int32)
    p_owner = jnp.asarray([p[1] for p in raw_planets] + [-1] * pad, dtype=jnp.int32)
    p_x = jnp.asarray([p[2] for p in raw_planets] + [0.0] * pad, dtype=jnp.float32)
    p_y = jnp.asarray([p[3] for p in raw_planets] + [0.0] * pad, dtype=jnp.float32)
    p_radius = jnp.asarray([p[4] for p in raw_planets] + [0.0] * pad, dtype=jnp.float32)
    p_ships = jnp.asarray([p[5] for p in raw_planets] + [0.0] * pad, dtype=jnp.float32)
    p_production = jnp.asarray([p[6] for p in raw_planets] + [0.0] * pad, dtype=jnp.float32)
    p_alive = jnp.asarray([True] * n + [False] * pad, dtype=jnp.bool_)
    p_init_x = jnp.asarray([p[2] for p in initial_planets] + [0.0] * pad, dtype=jnp.float32)
    p_init_y = jnp.asarray([p[3] for p in initial_planets] + [0.0] * pad, dtype=jnp.float32)
    p_is_comet = jnp.zeros(MAX_PLANETS, dtype=jnp.bool_)

    # ---- Fleet arrays (empty at start) ----
    f_owner = jnp.zeros(MAX_FLEETS, dtype=jnp.int32)
    f_x     = jnp.zeros(MAX_FLEETS, dtype=jnp.float32)
    f_y     = jnp.zeros(MAX_FLEETS, dtype=jnp.float32)
    f_angle = jnp.zeros(MAX_FLEETS, dtype=jnp.float32)
    f_ships = jnp.zeros(MAX_FLEETS, dtype=jnp.float32)
    f_alive = jnp.zeros(MAX_FLEETS, dtype=jnp.bool_)

    # ---- Pre-compute all comet paths ----
    cg_paths = [
        [
            [[0.0, 0.0] for _ in range(MAX_PATH_LEN)]
            for _ in range(COMET_PER_GROUP)
        ]
        for _ in range(N_COMET_GROUPS)
    ]
    cg_lens = [[0 for _ in range(COMET_PER_GROUP)] for _ in range(N_COMET_GROUPS)]
    cg_ships = [0.0 for _ in range(N_COMET_GROUPS)]
    cg_idx = [-2 for _ in range(N_COMET_GROUPS)]   # -2 = never spawned

    for gi, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = _stdlib_random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = generate_comet_paths(
            initial_planets, angular_velocity, spawn_step,
            comet_planet_ids=[],
            comet_speed=COMET_SPEED,
            rng=comet_rng,
        )
        ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        cg_ships[gi] = float(ships)
        if paths is not None:
            for j, path in enumerate(paths):
                L = min(len(path), MAX_PATH_LEN)
                cg_lens[gi][j] = L
                for k in range(L):
                    cg_paths[gi][j][k][0] = path[k][0]
                    cg_paths[gi][j][k][1] = path[k][1]

    return EnvState(
        p_id            = p_id,
        p_owner         = p_owner,
        p_x             = p_x,
        p_y             = p_y,
        p_radius        = p_radius,
        p_ships         = p_ships,
        p_production    = p_production,
        p_alive         = p_alive,
        p_init_x        = p_init_x,
        p_init_y        = p_init_y,
        p_is_comet      = p_is_comet,
        f_owner         = f_owner,
        f_x             = f_x,
        f_y             = f_y,
        f_angle         = f_angle,
        f_ships         = f_ships,
        f_alive         = f_alive,
        comet_paths     = jnp.asarray(cg_paths, dtype=jnp.float32),
        comet_path_lens = jnp.asarray(cg_lens, dtype=jnp.int32),
        comet_ships     = jnp.asarray(cg_ships, dtype=jnp.float32),
        comet_path_idx  = jnp.asarray(cg_idx, dtype=jnp.int32),
        step             = jnp.int32(0),
        angular_velocity = jnp.float32(angular_velocity),
        num_agents       = jnp.int32(num_agents),
        comet_id_base    = jnp.int32(max(p[0] for p in raw_planets) + 1 if raw_planets else 0),
    )


# ---------------------------------------------------------------------------
# Comet lifecycle helpers (pure JAX)
# ---------------------------------------------------------------------------

def _deactivate_comet_slots(
    p_alive: jnp.ndarray,
    p_is_comet: jnp.ndarray,
    p_ships: jnp.ndarray,
    p_owner: jnp.ndarray,
    comet_path_idx: jnp.ndarray,
    comet_path_lens: jnp.ndarray,
) -> tuple:
    """Mark comet planet slots inactive when their group's path is exhausted."""
    # For each comet group gi, the group is expired when path_idx >= min(path_lens[gi])
    # i.e. the shortest member has expired.  Use the minimum non-zero length.
    for gi in range(N_COMET_GROUPS):
        slot_base = COMET_SLOT_BASE + gi * COMET_PER_GROUP
        min_len = jnp.min(comet_path_lens[gi])   # shortest path in the group
        expired = (comet_path_idx[gi] >= -1) & (comet_path_idx[gi] >= min_len)
        for sj in range(COMET_PER_GROUP):
            slot = slot_base + sj
            p_alive    = p_alive.at[slot].set(jnp.where(expired, False, p_alive[slot]))
            p_is_comet = p_is_comet.at[slot].set(jnp.where(expired, False, p_is_comet[slot]))
    return p_alive, p_is_comet, p_ships, p_owner


def _spawn_comet_group(
    gi: int,
    state: EnvState,
) -> EnvState:
    """Activate comet group *gi* into its pre-assigned planet slots."""
    slot_base = COMET_SLOT_BASE + gi * COMET_PER_GROUP
    ships = state.comet_ships[gi]
    p_id         = state.p_id
    p_owner      = state.p_owner
    p_x          = state.p_x
    p_y          = state.p_y
    p_radius     = state.p_radius
    p_ships      = state.p_ships
    p_production = state.p_production
    p_alive      = state.p_alive
    p_init_x     = state.p_init_x
    p_init_y     = state.p_init_y
    p_is_comet   = state.p_is_comet
    comet_path_idx = state.comet_path_idx

    # All comet groups reuse the same game IDs: comet_id_base, +1, +2, +3.
    # This is correct because only one group can be active at a time (groups
    # expire within 40 ticks; next group spawns 100 ticks later).
    base_id = state.comet_id_base

    for j in range(COMET_PER_GROUP):
        slot = slot_base + j
        p_id         = p_id.at[slot].set(base_id + j)
        p_owner      = p_owner.at[slot].set(-1)
        p_x          = p_x.at[slot].set(-99.0)   # off-board sentinel
        p_y          = p_y.at[slot].set(-99.0)
        p_radius     = p_radius.at[slot].set(COMET_RADIUS)
        p_ships      = p_ships.at[slot].set(ships)
        p_production = p_production.at[slot].set(COMET_PRODUCTION)
        p_alive      = p_alive.at[slot].set(True)
        p_init_x     = p_init_x.at[slot].set(-99.0)
        p_init_y     = p_init_y.at[slot].set(-99.0)
        p_is_comet   = p_is_comet.at[slot].set(True)

    comet_path_idx = comet_path_idx.at[gi].set(-1)  # -1: incremented to 0 on first _compute_planet_new_positions call

    return state._replace(
        p_id=p_id, p_owner=p_owner, p_x=p_x, p_y=p_y,
        p_radius=p_radius, p_ships=p_ships, p_production=p_production,
        p_alive=p_alive, p_init_x=p_init_x, p_init_y=p_init_y,
        p_is_comet=p_is_comet, comet_path_idx=comet_path_idx,
    )


# ---------------------------------------------------------------------------
# Step sub-phases (pure JAX)
# ---------------------------------------------------------------------------

def _phase_remove_expired(state: EnvState) -> EnvState:
    """Deactivate comet groups whose path is exhausted."""
    p_alive, p_is_comet, p_ships, p_owner = _deactivate_comet_slots(
        state.p_alive, state.p_is_comet, state.p_ships, state.p_owner,
        state.comet_path_idx, state.comet_path_lens,
    )
    return state._replace(p_alive=p_alive, p_is_comet=p_is_comet,
                          p_ships=p_ships, p_owner=p_owner)


def _phase_spawn_comets(state: EnvState) -> EnvState:
    """Spawn a comet group if next completed step is a spawn tick."""
    next_step = state.step + 1
    for gi, spawn_step in enumerate(COMET_SPAWN_STEPS):
        should_spawn = (next_step == spawn_step) & (state.comet_path_idx[gi] < -1)
        # Only spawn if paths were successfully generated (comet_path_lens > 0)
        has_paths = jnp.any(state.comet_path_lens[gi] > 0)
        do_spawn  = should_spawn & has_paths
        # Compute what state would look like after spawning
        spawned = _spawn_comet_group(gi, state)
        # Conditionally apply
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_spawn, new, old),
            spawned,
            state,
        )
    return state


def _phase_process_moves(state: EnvState, actions: jnp.ndarray) -> EnvState:
    """Launch fleets for all agents.

    actions: float32[num_agents, MAX_ACTIONS_PA, 3]  (num_agents ≤ N_AGENTS)
             [..., 0]=from_planet_id  [..., 1]=angle  [..., 2]=ships
    """
    # Compact fleets first so alive fleets are at 0..num_alive-1
    order    = jnp.argsort(~state.f_alive)    # alive (False when negated) → front
    f_owner  = state.f_owner[order]
    f_x      = state.f_x[order]
    f_y      = state.f_y[order]
    f_angle  = state.f_angle[order]
    f_ships  = state.f_ships[order]
    f_alive  = state.f_alive[order]

    num_alive = jnp.sum(f_alive).astype(jnp.int32)

    # Build a planet-ID → slot lookup: p_id_to_slot[game_id] requires dynamic size.
    # Instead we search linearly inside the scan.
    p_id     = state.p_id
    p_owner_ = state.p_owner
    p_x_     = state.p_x
    p_y_     = state.p_y
    p_radius_= state.p_radius
    p_ships_ = state.p_ships
    p_alive_ = state.p_alive

    carry = (p_ships_, f_owner, f_x, f_y, f_angle, f_ships, f_alive, num_alive)

    def launch_one(carry, action_agent):
        """Scan body: process one (player_id, action) pair."""
        p_ships_, f_owner, f_x, f_y, f_angle, f_ships, f_alive, next_slot = carry
        player_id, from_id, angle, ships = action_agent

        from_id_i = from_id.astype(jnp.int32)
        ships_i   = ships.astype(jnp.float32)

        # Find planet slot matching from_id
        matches  = p_alive_ & (p_id == from_id_i)
        slot     = jnp.argmax(matches)           # first match
        found    = jnp.any(matches)

        valid = (
            found
            & (from_id_i >= 0)
            & (p_owner_[slot] == player_id.astype(jnp.int32))
            & (ships_i > 0.0)
            & (p_ships_[slot] >= ships_i)
        )

        # Deduct ships from source planet
        new_p_ships = p_ships_.at[slot].add(jnp.where(valid, -ships_i, 0.0))

        # Compute fleet launch position (just outside planet surface)
        cos_a   = jnp.cos(angle)
        sin_a   = jnp.sin(angle)
        lx      = p_x_[slot] + cos_a * (p_radius_[slot] + 0.1)
        ly      = p_y_[slot] + sin_a * (p_radius_[slot] + 0.1)

        # Write to next free fleet slot
        fslot = jnp.clip(next_slot, 0, MAX_FLEETS - 1)
        new_f_owner = f_owner.at[fslot].set(jnp.where(valid, player_id.astype(jnp.int32), f_owner[fslot]))
        new_f_x     = f_x.at[fslot].set(jnp.where(valid, lx,      f_x[fslot]))
        new_f_y     = f_y.at[fslot].set(jnp.where(valid, ly,      f_y[fslot]))
        new_f_angle = f_angle.at[fslot].set(jnp.where(valid, angle, f_angle[fslot]))
        new_f_ships = f_ships.at[fslot].set(jnp.where(valid, ships_i, f_ships[fslot]))
        new_f_alive = f_alive.at[fslot].set(jnp.where(valid, True,  f_alive[fslot]))
        new_next    = next_slot + jnp.where(valid & (next_slot < MAX_FLEETS), 1, 0)

        new_carry = (new_p_ships, new_f_owner, new_f_x, new_f_y, new_f_angle, new_f_ships, new_f_alive, new_next)
        return new_carry, None

    # Flatten actions: [n_agents, MAX_ACTIONS_PA, 3] → [n*MAX_A, 4] (player_id, from_id, angle, ships)
    # actions.shape[0] is the actual number of agents (may be < N_AGENTS)
    n_act = actions.shape[0]
    agent_ids = jnp.arange(n_act, dtype=jnp.float32)[:, None] * jnp.ones((1, MAX_ACTIONS_PA))
    actions_flat = jnp.concatenate([agent_ids[:, :, None], actions], axis=-1)  # [n, MAX_A, 4]
    actions_flat = actions_flat.reshape(-1, 4)  # [n*MAX_A, 4]

    (p_ships_, f_owner, f_x, f_y, f_angle, f_ships, f_alive, _), _ = jax.lax.scan(
        launch_one, carry, actions_flat
    )

    return state._replace(
        p_ships=p_ships_,
        f_owner=f_owner, f_x=f_x, f_y=f_y,
        f_angle=f_angle, f_ships=f_ships, f_alive=f_alive,
    )


def _phase_production(state: EnvState) -> EnvState:
    gain = jnp.where(state.p_alive & (state.p_owner >= 0), state.p_production, 0.0)
    return state._replace(p_ships=state.p_ships + gain)


def _compute_planet_new_positions(state: EnvState):
    """Compute (old_x, old_y, new_x, new_y, check_col, is_comet_first) per planet.

    Also advances comet_path_idx.  Returns updated state + position arrays.
    """
    cur_step = state.step
    av       = state.angular_velocity

    # -- Regular (non-comet) planets: orbital rotation --
    dx        = state.p_init_x - CENTER
    dy        = state.p_init_y - CENTER
    orb_r     = jnp.sqrt(dx * dx + dy * dy)
    rotating  = (orb_r + state.p_radius < ROTATION_RADIUS_LIMIT) & state.p_alive & ~state.p_is_comet
    init_ang  = jnp.arctan2(dy, dx)
    cur_ang   = init_ang + av * cur_step.astype(jnp.float32)
    rot_new_x = CENTER + orb_r * jnp.cos(cur_ang)
    rot_new_y = CENTER + orb_r * jnp.sin(cur_ang)

    new_x = jnp.where(rotating, rot_new_x, state.p_x)
    new_y = jnp.where(rotating, rot_new_y, state.p_y)

    # -- Comet planets: advance one step along pre-computed path --
    # Advance each group's path_idx by 1 if spawned (idx >= -1: -1 → 0 on first tick)
    new_cg_idx = state.comet_path_idx + jnp.where(state.comet_path_idx >= -1, 1, 0)

    # For each comet slot: look up its group and sub-index
    check_col = jnp.ones(MAX_PLANETS, dtype=bool)   # default: check collision

    for gi in range(N_COMET_GROUPS):
        slot_base = COMET_SLOT_BASE + gi * COMET_PER_GROUP
        idx       = new_cg_idx[gi]
        spawned   = new_cg_idx[gi] >= 0   # true after first increment (-1 → 0)
        for sj in range(COMET_PER_GROUP):
            slot    = slot_base + sj
            path_len = state.comet_path_lens[gi, sj]
            in_range = idx < path_len
            cx = jnp.where(spawned & in_range, state.comet_paths[gi, sj, jnp.clip(idx, 0, MAX_PATH_LEN - 1), 0], state.p_x[slot])
            cy = jnp.where(spawned & in_range, state.comet_paths[gi, sj, jnp.clip(idx, 0, MAX_PATH_LEN - 1), 1], state.p_y[slot])
            new_x = new_x.at[slot].set(jnp.where(state.p_alive[slot], cx, new_x[slot]))
            new_y = new_y.at[slot].set(jnp.where(state.p_alive[slot], cy, new_y[slot]))
            # First placement: old x = -99 → skip collision (off-board comet)
            first = state.p_x[slot] < 0.0
            check_col = check_col.at[slot].set(jnp.where(state.p_alive[slot], ~first, check_col[slot]))

    return state._replace(comet_path_idx=new_cg_idx), new_x, new_y, check_col


def _phase_move_fleets(
    state: EnvState,
    new_p_x: jnp.ndarray,
    new_p_y: jnp.ndarray,
    check_col: jnp.ndarray,
) -> tuple[EnvState, jnp.ndarray]:
    """Move all fleets; detect collisions.

    Returns (state with updated fleet positions and alive flags,
             fleet_hit_planet_slot [MAX_FLEETS] int32 — planet slot hit or -1)
    """
    speed = _fleet_speed(state.f_ships)  # [MAX_F]
    cos_a = jnp.cos(state.f_angle)
    sin_a = jnp.sin(state.f_angle)

    old_fx = state.f_x
    old_fy = state.f_y
    new_fx = old_fx + cos_a * speed
    new_fy = old_fy + sin_a * speed

    # ---- Swept planet collision [MAX_F, MAX_P] ----
    old_px = state.p_x
    old_py = state.p_y

    hits = _swept_pair_hit_batch(old_fx, old_fy, new_fx, new_fy,
                                 old_px, old_py, new_p_x, new_p_y,
                                 state.p_radius)   # [MAX_F, MAX_P]

    # Mask out inactive fleets, inactive/no-check planets
    fleet_mask  = state.f_alive[:, None]                    # [MAX_F, 1]
    planet_mask = (state.p_alive & check_col)[None, :]      # [1, MAX_P]
    hits = hits & fleet_mask & planet_mask                   # [MAX_F, MAX_P]

    # First hit planet per fleet (argmax gives first True column per row)
    any_hit   = jnp.any(hits, axis=1)                       # [MAX_F]
    first_hit = jnp.argmax(hits, axis=1)                    # [MAX_F] (0 if no hit)
    fleet_hit_planet = jnp.where(any_hit, first_hit, -1).astype(jnp.int32)

    # ---- Out-of-board check ----
    oob = ~((0.0 <= new_fx) & (new_fx <= BOARD_SIZE) &
            (0.0 <= new_fy) & (new_fy <= BOARD_SIZE))

    # ---- Sun collision (point-to-segment) ----
    # point_to_segment_distance for static point: just distance from sun to fleet segment
    # We need: min dist from (CENTER,CENTER) to segment (old_f, new_f)
    # Compute properly:
    seg_dx = new_fx - old_fx
    seg_dy = new_fy - old_fy
    seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy
    t_sun  = jnp.where(
        seg_l2 > 0.0,
        jnp.clip(((CENTER - old_fx) * seg_dx + (CENTER - old_fy) * seg_dy) / (seg_l2 + 1e-30), 0.0, 1.0),
        0.0,
    )
    proj_x = old_fx + t_sun * seg_dx
    proj_y = old_fy + t_sun * seg_dy
    sun_hit = ((proj_x - CENTER) ** 2 + (proj_y - CENTER) ** 2) < SUN_RADIUS ** 2

    # ---- Remove killed fleets ----
    killed = state.f_alive & (any_hit | oob | sun_hit)
    f_alive_new = state.f_alive & ~killed

    state = state._replace(f_x=new_fx, f_y=new_fy, f_alive=f_alive_new)
    return state, fleet_hit_planet


def _phase_resolve_combat(
    state: EnvState,
    fleet_hit_planet: jnp.ndarray,
) -> EnvState:
    """Resolve combat at each planet.

    fleet_hit_planet[i] = planet slot that fleet i hit (-1 = none).
    Dead fleets (f_alive=False before this phase) are still visible in
    fleet_hit_planet; we use state.f_ships/f_owner directly.
    """
    # For each fleet that hit a planet, sum ships by (planet_slot, owner)
    # combat_ships[MAX_P, N_AGENTS]: total fleet ships arriving from each agent
    # N_AGENTS is the array capacity; actual player count lives in state.num_agents
    hit_valid  = fleet_hit_planet >= 0    # [MAX_F] bool (hit something)
    p_slot     = jnp.clip(fleet_hit_planet, 0, MAX_PLANETS - 1)   # safe index
    owner_clp  = jnp.clip(state.f_owner, 0, N_AGENTS - 1)

    flat_idx   = p_slot * N_AGENTS + owner_clp  # [MAX_F]
    cs_flat    = jnp.zeros(MAX_PLANETS * N_AGENTS + 1)
    dummy_idx  = MAX_PLANETS * N_AGENTS          # dummy slot for non-hitting fleets
    safe_flat  = jnp.where(hit_valid, flat_idx, dummy_idx)

    cs_flat = cs_flat.at[safe_flat].add(
        jnp.where(hit_valid, state.f_ships, 0.0)
    )
    combat_ships = cs_flat[:MAX_PLANETS * N_AGENTS].reshape(MAX_PLANETS, N_AGENTS)
    # Mask out slots beyond actual player count so phantom agents don't win combat
    agent_mask = jnp.arange(N_AGENTS) < state.num_agents   # [N_AGENTS]
    combat_ships = combat_ships * agent_mask[None, :]
    # [MAX_P, N_AGENTS]: ships arriving from each agent at each planet

    has_combat  = jnp.any(combat_ships > 0.0, axis=1)         # [MAX_P]
    top_owner   = jnp.argmax(combat_ships, axis=1).astype(jnp.int32)  # [MAX_P]
    top_ships   = jnp.max(combat_ships, axis=1)                # [MAX_P]
    second_ships= jnp.sort(combat_ships, axis=1)[:, -2]        # [MAX_P]
    survivor    = top_ships - second_ships                      # [MAX_P]

    # survivor > 0: winner takes difference
    # survivor == 0 (tie): nothing changes
    friendly    = (top_owner == state.p_owner)                 # [MAX_P]
    new_garrison= jnp.where(friendly,
                            state.p_ships + survivor,
                            state.p_ships - survivor)
    conquest    = (~friendly) & (new_garrison < 0.0)
    new_owner   = jnp.where(conquest, top_owner, state.p_owner)
    new_ships   = jnp.where(conquest, -new_garrison,
                  jnp.where(friendly | (new_garrison >= 0.0),
                             jnp.maximum(new_garrison, 0.0),
                             new_garrison))  # hostile, positive garrison → remains

    apply = state.p_alive & has_combat & (survivor > 0.0)
    p_owner_new = jnp.where(apply, new_owner, state.p_owner)
    p_ships_new = jnp.where(apply, new_ships, state.p_ships)

    return state._replace(p_owner=p_owner_new, p_ships=p_ships_new)


def _phase_check_termination(
    state: EnvState,
    episode_steps: int = EPISODE_STEPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (statuses [N_AGENTS] int32, rewards [N_AGENTS] float32).

    statuses: 0=ACTIVE, 1=DONE
    rewards:  +1.0=win, -1.0=lose, 0.0=draw / not yet terminated
    Slots beyond state.num_agents always return DONE with reward 0.
    """
    time_up = state.step >= episode_steps - 1

    # Which agent indices are actually participating
    agent_active = jnp.arange(N_AGENTS) < state.num_agents   # [N_AGENTS]

    # Presence: owns a planet or has a fleet in flight
    p_owner_eq = jax.vmap(lambda a: jnp.any(state.p_alive & (state.p_owner == a)))(jnp.arange(N_AGENTS))
    f_owner_eq = jax.vmap(lambda a: jnp.any(state.f_alive & (state.f_owner == a)))(jnp.arange(N_AGENTS))
    has_presence = (p_owner_eq | f_owner_eq) & agent_active   # [N_AGENTS]
    num_alive_players = jnp.sum(has_presence).astype(jnp.int32)

    elimination = num_alive_players <= 1
    terminated  = time_up | elimination

    # Scores: total ships controlled (planets + fleets)
    p_scores = jax.vmap(
        lambda a: jnp.sum(jnp.where(state.p_alive & (state.p_owner == a), state.p_ships, 0.0))
    )(jnp.arange(N_AGENTS))
    f_scores = jax.vmap(
        lambda a: jnp.sum(jnp.where(state.f_alive & (state.f_owner == a), state.f_ships, 0.0))
    )(jnp.arange(N_AGENTS))
    scores = (p_scores + f_scores) * agent_active   # zero-out phantom agents

    max_score = jnp.max(scores)
    is_winner = (scores == max_score) & (max_score > 0.0) & agent_active

    rewards  = jnp.where(agent_active,
                  jnp.where(terminated, jnp.where(is_winner, 1.0, -1.0), 0.0),
                  0.0)
    statuses = jnp.where(agent_active,
                  jnp.where(terminated, 1, 0),
                  1).astype(jnp.int32)   # phantom agents are always "done"
    return statuses, rewards


# ---------------------------------------------------------------------------
# Main step function (JIT-compiled, vmap-able)
# ---------------------------------------------------------------------------

def jax_step(
    state: EnvState,
    actions: jnp.ndarray,
    episode_steps: int = EPISODE_STEPS,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Advance state by one tick.  Pure function — returns new state.

    Parameters
    ----------
    state   : JaxState
    actions : float32[N_AGENTS, MAX_ACTIONS_PA, 3]
              column 0 = planet_id, 1 = angle, 2 = ships
              Rows with planet_id=-1 are no-ops.
    episode_steps : static int

    Returns
    -------
    new_state   : JaxState
    statuses    : int32[N_AGENTS]   0=ACTIVE 1=DONE
    rewards     : float32[N_AGENTS] nonzero only when DONE
    """
    # Phase 0a: remove expired comets
    state = _phase_remove_expired(state)

    # Phase 0b: spawn new comet group if applicable
    state = _phase_spawn_comets(state)

    # Phase 1: fleet launch
    state = _phase_process_moves(state, actions)

    # Phase 2: production
    state = _phase_production(state)

    # Phase 3: compute new planet positions
    state, new_p_x, new_p_y, check_col = _compute_planet_new_positions(state)

    # Phase 4: move fleets + collision detection
    state, fleet_hit_planet = _phase_move_fleets(state, new_p_x, new_p_y, check_col)

    # Phase 5: apply planet position updates
    state = state._replace(p_x=new_p_x, p_y=new_p_y)

    # Phase 5b: remove freshly expired comets (path_idx advanced past path_len)
    state = _phase_remove_expired(state)

    # Phase 6: combat resolution
    state = _phase_resolve_combat(state, fleet_hit_planet)

    # Advance step counter
    state = state._replace(step=state.step + 1)

    # Phase 7: termination check
    statuses, rewards = _phase_check_termination(state, episode_steps)

    return state, statuses, rewards


# Pre-JIT for single-env use
jax_step_jit = jax.jit(jax_step, static_argnums=(2,))


def encode_actions(raw_actions: list[list], player: int) -> jnp.ndarray:
    """Convert a Python action list → float32[MAX_ACTIONS_PA, 3].

    raw_actions: [[from_planet_id, angle, ships], ...]
    Returns a padded array suitable for jax_step.
    """
    rows = [[-1.0, -1.0, -1.0] for _ in range(MAX_ACTIONS_PA)]
    for i, move in enumerate(raw_actions):
        if i >= MAX_ACTIONS_PA:
            break
        rows[i][0] = float(move[0])  # planet id
        rows[i][1] = float(move[1])  # angle
        rows[i][2] = float(move[2])  # ships
    return jnp.asarray(rows, dtype=jnp.float32)


def encode_joint_actions(actions_per_agent: list[list[list]]) -> jnp.ndarray:
    """Stack per-agent encoded actions → float32[num_agents, MAX_ACTIONS_PA, 3].

    num_agents is inferred from len(actions_per_agent); may be 2 or 4.
    """
    rows = []
    for player, acts in enumerate(actions_per_agent):
        rows.append(encode_actions(acts, player))
    return jnp.stack(rows, axis=0)  # [num_agents, MAX_ACTIONS_PA, 3]
