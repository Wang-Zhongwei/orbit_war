"""
Behavior-cloning data pipeline: episode JSON → flat (features, labels) arrays.

Each training sample = one source-planet decision at one timestep:
  self_feats   : [SELF_DIM]
  cand_feats   : [K, CAND_DIM]
  global_feats : [GLOB_DIM]
  cand_mask    : [K]
  fleet_feats  : [MAX_FLEETS, FLEET_DIM]
  fleet_mask   : [MAX_FLEETS]
  target_idx   : int in [0, K)   — which candidate the expert chose
  ships_bucket : int in [0, B)   — how many ships (as a fraction bucket)

Label extraction:
  - source planet: matched by planet_id in encoder's src_ids
  - target candidate: matched by recomputing the intercept angle to each candidate
    using the expert's ACTUAL ship count (act_ships), then finding the candidate
    whose recomputed angle is closest to the action's launch angle.
    Using actual ships (vs the encoder's estimated src_ships//2) matters for
    rotating targets whose lead angle depends on fleet speed.
  - ship bucket: bucket b sends floor((b+1)/B * src_ships) ships;
    pick the bucket whose ships_count is closest to the expert's choice
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
import zipfile
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from .constants import CENTER, MAX_SHIP_SPEED, ROTATION_RADIUS_LIMIT
from .physics import (
    EnvState,
    MAX_BASE_PLANETS,
    MAX_FLEETS,
    MAX_PATH_LEN,
    MAX_PLANETS,
    N_COMET_GROUPS,
)

# Maximum angular distance (radians) to accept an angle match.
# ~34 degrees; tight enough to avoid mis-labeling, loose enough for imperfect expert angles.
_ANGLE_MATCH_THRESHOLD: float = 0.6

_MAX_TICKS_PY: int = 25  # must match features.py's _MAX_TICKS


def _find_angles_for_action(
    src_x: float, src_y: float, src_radius: float,
    tgt_x: np.ndarray, tgt_y: np.ndarray, tgt_radius: np.ndarray,  # [K]
    ships: float, angular_velocity: float,
    valid_mask: np.ndarray,  # [K] bool — which candidates to process
) -> np.ndarray:
    """Compute intercept angles to all K candidates in one vectorized sweep.

    The 25-tick scan stays in a Python loop but operates on K-wide numpy arrays,
    replacing K separate Python calls with 25 numpy operations over K elements.
    """
    s = max(ships, 1.0)
    spd = 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(s) / math.log(1000.0)) ** 1.5

    dx0 = tgt_x - CENTER
    dy0 = tgt_y - CENTER
    orb_r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    rotating  = (orb_r + tgt_radius) < ROTATION_RADIUS_LIMIT  # [K]
    init_ang  = np.arctan2(dy0, dx0)                           # [K]

    # Static fallback — non-rotating targets and rotating ones with no intercept
    angles = np.arctan2(tgt_y - src_y, tgt_x - src_x)         # [K]
    found      = np.zeros(len(tgt_x), dtype=bool)
    needs_scan = rotating & valid_mask

    for tick in range(1, _MAX_TICKS_PY + 1):
        remaining = needs_scan & ~found
        if not remaining.any():
            break
        t    = float(tick)
        ang_t = init_ang + angular_velocity * t
        tx   = CENTER + orb_r * np.cos(ang_t)
        ty   = CENTER + orb_r * np.sin(ang_t)
        dx   = tx - src_x
        dy   = ty - src_y
        gap  = spd * t - (np.sqrt(dx * dx + dy * dy) - src_radius)
        hit  = remaining & (np.abs(gap) < tgt_radius)
        angles = np.where(hit, np.arctan2(dy, dx), angles)
        found |= hit

    return angles  # [K]


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

    return EnvState(
        p_id          = jnp.asarray(p_id),
        p_owner       = jnp.asarray(p_owner),
        p_x           = jnp.asarray(p_x),
        p_y           = jnp.asarray(p_y),
        p_radius      = jnp.asarray(p_radius),
        p_ships       = jnp.asarray(p_ships),
        p_production  = jnp.asarray(p_production),
        p_alive       = jnp.asarray(p_alive),
        p_init_x      = jnp.asarray(p_init_x),
        p_init_y      = jnp.asarray(p_init_y),
        p_is_comet    = jnp.asarray(p_is_comet),
        f_owner       = jnp.asarray(f_owner),
        f_x           = jnp.asarray(f_x),
        f_y           = jnp.asarray(f_y),
        f_angle       = jnp.asarray(f_angle),
        f_ships       = jnp.asarray(f_ships),
        f_alive       = jnp.asarray(f_alive),
        comet_paths     = jnp.asarray(dummy_paths),
        comet_path_lens = jnp.asarray(dummy_path_lens),
        comet_ships     = jnp.asarray(dummy_ships_cg),
        comet_path_idx  = jnp.asarray(dummy_path_idx),
        step             = jnp.int32(obs["step"]),
        angular_velocity = jnp.float32(obs["angular_velocity"]),
        num_agents       = jnp.int32(num_agents),
        comet_id_base    = jnp.int32(n_base),
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def _ships_to_bucket(act_ships: int, src_ships: int, B: int) -> int:
    """Return the bucket index whose decoded ship count is closest to act_ships."""
    bucket_ships = [max(1, int((b + 1) * src_ships // B)) for b in range(B)]
    return int(min(range(B), key=lambda b: abs(bucket_ships[b] - act_ships)))


def extract_labels(
    action: list,
    *,
    src_ids: np.ndarray,        # [MAX_BASE_PLANETS] int32
    src_valid: np.ndarray,      # [MAX_BASE_PLANETS] bool
    src_ships: np.ndarray,      # [MAX_BASE_PLANETS] int32
    cand_mask: np.ndarray,      # [MAX_BASE_PLANETS, K] bool
    B: int,
    # State geometry — needed to recompute angles with the expert's actual ship count
    p_x: np.ndarray,            # [MAX_PLANETS] float32
    p_y: np.ndarray,            # [MAX_PLANETS] float32
    p_radius: np.ndarray,       # [MAX_PLANETS] float32
    angular_velocity: float,
    src_slots: np.ndarray,      # [MAX_BASE_PLANETS] int32 — planet slot per source
    cand_slots: np.ndarray,     # [MAX_BASE_PLANETS, K] int32 — planet slot per candidate
) -> list[tuple[int, int, int]]:
    """Return (src_slot_idx, cand_idx, ship_bucket) for each matched action entry.

    If the expert's launch angle is beyond _ANGLE_MATCH_THRESHOLD from every candidate
    (i.e. the target was outside the candidate pool), the action is labelled as no-op
    (cand_idx=0, ships_bucket=0) rather than dropped — preserving the training signal
    that this state called for inaction within the modelled action space.
    """
    labels: list[tuple[int, int, int]] = []

    for act in action:
        from_pid  = int(act[0])
        act_angle = float(act[1])
        act_ships = int(act[2])

        # Find source slot
        src_idx = -1
        for j in range(len(src_ids)):
            if src_valid[j] and src_ids[j] == from_pid:
                src_idx = j
                break
        if src_idx == -1:
            continue  # planet not in encoder's source set (< MIN_SOURCE_SHIPS)

        S = int(src_ships[src_idx])
        if S <= 0:
            continue

        src_slot  = int(src_slots[src_idx])
        cslots    = cand_slots[src_idx]          # [K]

        # Exclude the no-op slot (k=0) from angle matching
        valid_mask = cand_mask[src_idx].copy()
        valid_mask[0] = False

        # Recompute intercept angles to all K candidates at once with actual act_ships.
        # This vectorises the per-candidate scan: 25 numpy ops over K instead of K×25 calls.
        angles_k = _find_angles_for_action(
            float(p_x[src_slot]), float(p_y[src_slot]), float(p_radius[src_slot]),
            p_x[cslots], p_y[cslots], p_radius[cslots],
            float(act_ships), angular_velocity,
            valid_mask,
        )

        # Vectorised angular distance; masked candidates get inf
        diff  = angles_k - act_angle
        dists = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
        dists = np.where(valid_mask, dists, np.inf)

        best_k    = int(np.argmin(dists))
        best_dist = float(dists[best_k])

        if best_dist > _ANGLE_MATCH_THRESHOLD:
            # Expert targeted a planet outside the candidate pool → label as no-op
            labels.append((src_idx, 0, 0))
            continue

        bucket = _ships_to_bucket(act_ships, S, B)
        labels.append((src_idx, best_k, bucket))

    return labels


# ---------------------------------------------------------------------------
# Episode → flat sample arrays
# ---------------------------------------------------------------------------

def episode_to_samples(
    episode: dict,
    encode_fn,
    ship_bucket_count: int,
) -> dict[str, np.ndarray] | None:
    """Convert one episode JSON to flat numpy arrays of BC training samples.

    encode_fn: the vmapped JAX encoder from _make_env_encoder.
    Returns None if no valid samples were found.

    Replay timing: steps[i]['observation'] is the game state AFTER steps[i]['action']
    has been processed.  The agent chose steps[i]['action'] based on the state they
    could see, which was steps[i-1]['observation'].  We therefore pair:
        state   = steps[i-1][agent]['observation']
        action  = steps[i][agent]['action']
    """
    batch_states, batch_players, batch_actions = _collect_states_actions(episode)
    if not batch_states:
        return None

    # Batch all states and run encoder once per episode
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batch_states)
    players_arr = jnp.array(batch_players, dtype=jnp.int32)
    enc = encode_fn(batched, players_arr)
    enc_np     = jax.tree_util.tree_map(np.asarray, enc)
    batched_np = jax.tree_util.tree_map(np.asarray, batched)

    rows_sf, rows_cf, rows_gf, rows_cm = [], [], [], []
    rows_ff, rows_fm, rows_ti, rows_sb = [], [], [], []

    for i in range(len(batch_states)):
        labels = extract_labels(
            batch_actions[i],
            src_ids          = enc_np.src_ids[i],
            src_valid        = enc_np.src_valid[i],
            src_ships        = enc_np.src_ships[i],
            cand_mask        = enc_np.cand_mask[i],
            B                = ship_bucket_count,
            p_x              = batched_np.p_x[i],
            p_y              = batched_np.p_y[i],
            p_radius         = batched_np.p_radius[i],
            angular_velocity = float(batched_np.angular_velocity[i]),
            src_slots        = enc_np.src_slots[i],
            cand_slots       = enc_np.cand_slots_safe[i],
        )
        # Emit explicit no-op labels for valid sources the expert chose to hold.
        # Without these, the policy only sees "given expert decided to launch,
        # where to?" and collapses toward always-launch at inference.
        acted_src = {src_idx for src_idx, _, _ in labels}
        src_valid_i = enc_np.src_valid[i]
        for j in range(len(src_valid_i)):
            if bool(src_valid_i[j]) and j not in acted_src:
                labels.append((j, 0, 0))
        for src_idx, cand_idx, ship_bucket in labels:
            rows_sf.append(enc_np.self_feats[i,   src_idx])
            rows_cf.append(enc_np.cand_feats[i,   src_idx])
            rows_gf.append(enc_np.global_feat[i])
            rows_cm.append(enc_np.cand_mask[i,    src_idx])
            # rows_ff.append(enc_np.fleet_feats[i])
            # rows_fm.append(enc_np.fleet_mask[i])
            rows_ti.append(cand_idx)
            rows_sb.append(ship_bucket)

    if not rows_sf:
        return None

    return {
        "self_feats":   np.stack(rows_sf).astype(np.float32),
        "cand_feats":   np.stack(rows_cf).astype(np.float32),
        "global_feats": np.stack(rows_gf).astype(np.float32),
        "cand_mask":    np.stack(rows_cm),
        # "fleet_feats":  np.stack(rows_ff).astype(np.float32),
        # "fleet_mask":   np.stack(rows_fm),
        "target_idx":   np.array(rows_ti, dtype=np.int32),
        "ships_bucket": np.array(rows_sb, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# On-device variant for streaming — avoids JAX→numpy→JAX round-trip
# ---------------------------------------------------------------------------

def _collect_states_actions(episode: dict):
    """Shared state/action collection used by both sample functions."""
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


def episode_to_samples_jax(
    episode: dict,
    encode_fn,
    ship_bucket_count: int,
) -> dict | None:
    """Streaming variant of episode_to_samples: feature arrays stay on device.

    Only the small index/mask arrays needed for label extraction (src_ids, cand_mask, etc.)
    are copied to CPU. The large feature tensors (self_feats, cand_feats, fleet_feats) are
    gathered on-device via JAX advanced indexing and returned as JAX arrays.

    Use in training loops where you want to avoid the JAX→numpy→JAX round-trip that
    episode_to_samples incurs when writing shards.
    """
    batch_states, batch_players, batch_actions = _collect_states_actions(episode)
    if not batch_states:
        return None

    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batch_states)
    players_arr = jnp.array(batch_players, dtype=jnp.int32)
    enc = encode_fn(batched, players_arr)

    # Transfer only the small arrays needed for label extraction
    src_ids_np    = np.asarray(enc.src_ids)
    src_valid_np  = np.asarray(enc.src_valid)
    src_ships_np  = np.asarray(enc.src_ships)
    cand_mask_np  = np.asarray(enc.cand_mask)
    src_slots_np  = np.asarray(enc.src_slots)
    cand_slots_np = np.asarray(enc.cand_slots_safe)
    p_x_np        = np.asarray(batched.p_x)
    p_y_np        = np.asarray(batched.p_y)
    p_radius_np   = np.asarray(batched.p_radius)
    ang_vel_np    = np.asarray(batched.angular_velocity)

    ts_list: list[int] = []
    si_list: list[int] = []
    ci_list: list[int] = []
    bi_list: list[int] = []

    for i in range(len(batch_states)):
        labels = extract_labels(
            batch_actions[i],
            src_ids          = src_ids_np[i],
            src_valid        = src_valid_np[i],
            src_ships        = src_ships_np[i],
            cand_mask        = cand_mask_np[i],
            B                = ship_bucket_count,
            p_x              = p_x_np[i],
            p_y              = p_y_np[i],
            p_radius         = p_radius_np[i],
            angular_velocity = float(ang_vel_np[i]),
            src_slots        = src_slots_np[i],
            cand_slots       = cand_slots_np[i],
        )
        # Emit explicit no-op labels for valid sources the expert chose to hold.
        # Without these, the policy only sees "given expert decided to launch,
        # where to?" and collapses toward always-launch at inference.
        acted_src = {src_idx for src_idx, _, _ in labels}
        src_valid_i = src_valid_np[i]
        for j in range(len(src_valid_i)):
            if bool(src_valid_i[j]) and j not in acted_src:
                labels.append((j, 0, 0))
        for src_idx, cand_idx, bucket in labels:
            ts_list.append(i)
            si_list.append(src_idx)
            ci_list.append(cand_idx)
            bi_list.append(bucket)

    if not ts_list:
        return None

    ts = np.array(ts_list, dtype=np.int32)
    si = np.array(si_list, dtype=np.int32)

    # Gather features on-device — large tensors never leave the accelerator
    return {
        "self_feats":   enc.self_feats[ts, si],
        "cand_feats":   enc.cand_feats[ts, si],
        "global_feats": enc.global_feat[ts],
        "cand_mask":    enc.cand_mask[ts, si],
        # "fleet_feats":  enc.fleet_feats[ts],
        # "fleet_mask":   enc.fleet_mask[ts],
        "target_idx":   jnp.array(ci_list, dtype=jnp.int32),
        "ships_bucket": jnp.array(bi_list, dtype=jnp.int32),
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

# ---------------------------------------------------------------------------
# Multiprocessing worker helpers
# ---------------------------------------------------------------------------

# Module-level state populated by each worker process via Pool initializer.
_worker_encode_fn = None
_worker_ship_bucket_count: int = 0


def _worker_init(encode_fn_factory, factory_args, ship_bucket_count: int) -> None:
    """Called once per worker process. Builds encode_fn locally (avoids pickling JAX fn)."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    global _worker_encode_fn, _worker_ship_bucket_count
    _worker_encode_fn = encode_fn_factory(*factory_args)
    _worker_ship_bucket_count = ship_bucket_count


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
    return episode_to_samples(episode, _worker_encode_fn, _worker_ship_bucket_count)


def encode_to_shards(
    episodes_dir: Path,
    encode_fn,
    ship_bucket_count: int,
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
        n_samples  += len(samples["target_idx"])
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
            initargs=(encode_fn_factory, encode_fn_factory_args, ship_bucket_count),
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
            samples = episode_to_samples(episode, encode_fn, ship_bucket_count)
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
