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
  - target candidate: matched by finding the encoder's precomputed angle
    (enc.angles[src_idx, k]) closest to the action's launch angle
  - ship bucket: bucket b sends floor((b+1)/B * src_ships) ships;
    pick the bucket whose ships_count is closest to the expert's choice
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from .physics import (
    EnvState,
    MAX_BASE_PLANETS,
    MAX_FLEETS,
    MAX_PATH_LEN,
    MAX_PLANETS,
    N_COMET_GROUPS,
)
from .features import _make_env_encoder

# Maximum angular distance (radians) to accept an angle match.
# ~34 degrees; tight enough to avoid mis-labeling, loose enough for imperfect expert angles.
_ANGLE_MATCH_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# Observation → EnvState
# ---------------------------------------------------------------------------

def obs_to_env_state(obs: dict) -> EnvState:
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
        num_agents       = jnp.int32(2),
        comet_id_base    = jnp.int32(n_base),
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return abs(math.atan2(math.sin(d), math.cos(d)))


def _ships_to_bucket(act_ships: int, src_ships: int, B: int) -> int:
    """Return the bucket index whose decoded ship count is closest to act_ships."""
    bucket_ships = [max(1, int((b + 1) * src_ships // B)) for b in range(B)]
    return int(min(range(B), key=lambda b: abs(bucket_ships[b] - act_ships)))


def extract_labels(
    action: list,
    *,
    src_ids: np.ndarray,    # [MAX_BASE_PLANETS] int32
    src_valid: np.ndarray,  # [MAX_BASE_PLANETS] bool
    src_ships: np.ndarray,  # [MAX_BASE_PLANETS] int32
    cand_mask: np.ndarray,  # [MAX_BASE_PLANETS, K] bool
    angles: np.ndarray,     # [MAX_BASE_PLANETS, K] float32
    B: int,
) -> list[tuple[int, int, int]]:
    """Return (src_slot_idx, cand_idx, ship_bucket) for each matched action entry."""
    K = cand_mask.shape[1]
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

        # Find closest candidate by angle — skip no-op (k=0)
        best_k    = -1
        best_dist = float("inf")
        for k in range(1, K):
            if not cand_mask[src_idx, k]:
                continue
            dist = _angle_diff(float(angles[src_idx, k]), act_angle)
            if dist < best_dist:
                best_dist = dist
                best_k    = k

        if best_k == -1 or best_dist > _ANGLE_MATCH_THRESHOLD:
            continue  # no valid candidate or angle too far from any candidate

        S = int(src_ships[src_idx])
        if S <= 0:
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
    steps = episode.get("steps", [])
    batch_states: list[EnvState] = []
    batch_players: list[int]     = []
    batch_actions: list[list]    = []

    for i in range(1, len(steps)):          # start at 1 so i-1 is valid
        for agent_idx, agent_data in enumerate(steps[i]):
            action = agent_data.get("action", [])
            if not action:
                continue
            if agent_data.get("status") == "DONE":
                continue
            # State the agent saw before acting: previous step's observation
            prev_obs = steps[i - 1][agent_idx]["observation"]
            try:
                state = obs_to_env_state(prev_obs)
            except Exception:
                continue
            batch_states.append(state)
            batch_players.append(int(prev_obs["player"]))
            batch_actions.append(action)

    if not batch_states:
        return None

    # Batch all states and run encoder once per episode
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batch_states)
    players_arr = jnp.array(batch_players, dtype=jnp.int32)
    enc = encode_fn(batched, players_arr)
    enc_np = jax.tree_util.tree_map(np.asarray, enc)

    rows_sf, rows_cf, rows_gf, rows_cm = [], [], [], []
    rows_ff, rows_fm, rows_ti, rows_sb = [], [], [], []

    N = len(batch_states)
    for i in range(N):
        labels = extract_labels(
            batch_actions[i],
            src_ids   = enc_np.src_ids[i],
            src_valid = enc_np.src_valid[i],
            src_ships = enc_np.src_ships[i],
            cand_mask = enc_np.cand_mask[i],
            angles    = enc_np.angles[i],
            B         = ship_bucket_count,
        )
        for src_idx, cand_idx, ship_bucket in labels:
            rows_sf.append(enc_np.self_feats[i,   src_idx])
            rows_cf.append(enc_np.cand_feats[i,   src_idx])
            rows_gf.append(enc_np.global_feat[i])
            rows_cm.append(enc_np.cand_mask[i,    src_idx])
            rows_ff.append(enc_np.fleet_feats[i])
            rows_fm.append(enc_np.fleet_mask[i])
            rows_ti.append(cand_idx)
            rows_sb.append(ship_bucket)

    if not rows_sf:
        return None

    return {
        "self_feats":   np.stack(rows_sf).astype(np.float32),
        "cand_feats":   np.stack(rows_cf).astype(np.float32),
        "global_feats": np.stack(rows_gf).astype(np.float32),
        "cand_mask":    np.stack(rows_cm),
        "fleet_feats":  np.stack(rows_ff).astype(np.float32),
        "fleet_mask":   np.stack(rows_fm),
        "target_idx":   np.array(rows_ti, dtype=np.int32),
        "ships_bucket": np.array(rows_sb, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Dataset loader — lazy iterator over episode files
# ---------------------------------------------------------------------------

def iter_episodes(
    episodes_dir: Path,
    encode_fn,
    ship_bucket_count: int,
    *,
    min_avg_score: float = 0.0,
    winner_only: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield sample dicts, one per episode, from all .json files in episodes_dir."""
    paths = sorted(episodes_dir.glob("*.json"))
    for path in paths:
        try:
            with open(path) as f:
                episode = json.load(f)
        except Exception:
            continue

        if not isinstance(episode, dict) or "steps" not in episode:
            continue  # old format (agent logs only)

        if winner_only:
            rewards = episode.get("rewards", [])
            statuses = episode.get("statuses", [])
            # Keep only actions from the player who won (+1 reward)
            winning_player = next(
                (i for i, (r, s) in enumerate(zip(rewards, statuses))
                 if r == max(rewards) and s == "DONE"),
                None,
            )
            if winning_player is None:
                continue
            # Mask out the losing agent's steps
            episode = _filter_winner(episode, winning_player)

        samples = episode_to_samples(episode, encode_fn, ship_bucket_count)
        if samples is not None:
            yield samples


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
