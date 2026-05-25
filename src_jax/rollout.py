"""
Fully JAX rollout via jax.lax.scan.

Everything from env-step → feature encoding → policy forward → action building
runs on-device.  The only Python involvement per training update is:
  1. Triggering the JIT-compiled collect() call.
  2. Computing returns (another scan, JIT-compiled).
  3. Logging scalar metrics.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .jax_physics import (
    JaxState,
    jax_step,
    MAX_BASE_PLANETS,
    MAX_ACTIONS_PA,
    N_AGENTS,
)
from .features import _make_env_encoder
from .ppo import TransitionBatch, sample_actions
from .env import select_boards

MAX_SRC = MAX_BASE_PLANETS   # alias for clarity
SELF_DIM = 11
CAND_DIM = 10
GLOB_DIM = 8


# ---------------------------------------------------------------------------
# Carry and step output types
# ---------------------------------------------------------------------------

class RolloutCarry(NamedTuple):
    state: JaxState               # [N, ...]
    board_indices: jnp.ndarray    # [N] int32
    ep_reward: jnp.ndarray        # [N] float32
    learner_players: jnp.ndarray  # [N] int32
    key: jnp.ndarray


class RolloutStep(NamedTuple):
    sf: jnp.ndarray        # [N, MAX_SRC, SELF_DIM]
    cf: jnp.ndarray        # [N, MAX_SRC, K, CAND_DIM]
    gf: jnp.ndarray        # [N, GLOB_DIM]
    cm: jnp.ndarray        # [N, MAX_SRC, K] bool
    ti: jnp.ndarray        # [N, MAX_SRC] int32
    sb: jnp.ndarray        # [N, MAX_SRC] int32
    lp: jnp.ndarray        # [N, MAX_SRC] float32
    val: jnp.ndarray       # [N, MAX_SRC] float32
    src_valid: jnp.ndarray # [N, MAX_SRC] bool
    reward: jnp.ndarray    # [N] float32
    done: jnp.ndarray      # [N] bool


# ---------------------------------------------------------------------------
# Pure-JAX action builder (one env, no batch dim)
# ---------------------------------------------------------------------------

def _build_actions_single(
    src_ids: jnp.ndarray,    # [MAX_SRC] int32
    src_ships: jnp.ndarray,  # [MAX_SRC] int32
    angles: jnp.ndarray,     # [MAX_SRC, K] float32
    cand_mask: jnp.ndarray,  # [MAX_SRC, K] bool
    src_valid: jnp.ndarray,  # [MAX_SRC] bool
    ti: jnp.ndarray,         # [MAX_SRC] int32
    sb: jnp.ndarray,         # [MAX_SRC] int32
    ship_bucket_count: int,
) -> jnp.ndarray:
    """Return [MAX_ACTIONS_PA, 3] action tensor. Pure JAX, no Python control flow."""
    row_idx = jnp.arange(MAX_SRC, dtype=jnp.int32)

    tgt_angles  = angles[row_idx, ti]                           # [MAX_SRC]
    tgt_ok      = (ti != 0) & cand_mask[row_idx, ti]
    action_ok   = src_valid & tgt_ok                            # [MAX_SRC]

    frac        = (sb.astype(jnp.float32) + 1.0) / jnp.float32(ship_bucket_count)
    ships_send  = jnp.maximum(1, jnp.floor(frac * src_ships.astype(jnp.float32)).astype(jnp.int32))

    n_act = min(MAX_SRC, MAX_ACTIONS_PA)
    rows  = jnp.stack([
        src_ids[:n_act].astype(jnp.float32),
        tgt_angles[:n_act],
        ships_send[:n_act].astype(jnp.float32),
    ], axis=1)                                                   # [n_act, 3]
    rows  = jnp.where(action_ok[:n_act, None], rows, jnp.full((3,), -1.0))

    if n_act < MAX_ACTIONS_PA:
        pad  = jnp.full((MAX_ACTIONS_PA - n_act, 3), -1.0, jnp.float32)
        rows = jnp.concatenate([rows, pad], axis=0)

    return rows  # [MAX_ACTIONS_PA, 3]


# ---------------------------------------------------------------------------
# Factory: returns a JIT-compiled collect() function
# ---------------------------------------------------------------------------

def make_rollout_fn(
    board_pool: JaxState,
    n_boards: int,
    cfg,
    module,
    rollout_steps: int,
):
    """
    Build and return a JIT-compiled rollout collector.

    Parameters
    ----------
    board_pool    : JaxState[n_boards, ...]
    n_boards      : int
    cfg           : TrainConfig
    module        : PlanetPolicy (Flax module)
    rollout_steps : int (static — baked into the JIT)

    Returns
    -------
    collect(params, opp_params, carry) -> (final_carry, RolloutStep[T, N, ...])
    """
    K                  = cfg.env.candidate_count
    N_ENVS             = cfg.ppo.num_envs
    ship_bucket_count  = cfg.env.ship_bucket_count
    episode_steps      = cfg.env.episode_steps
    alternate_sides    = cfg.alternate_player_sides
    opp_deterministic  = cfg.self_play_deterministic

    encode_batch = _make_env_encoder(K, cfg.env)
    _batched_step = jax.jit(jax.vmap(functools.partial(jax_step, episode_steps=episode_steps)))

    # ── helpers (closed over) ────────────────────────────────────────────────

    def _run_policy(params, enc, key, deterministic):
        sf = enc.self_feats   # [N, MAX_SRC, SELF_DIM]
        cf = enc.cand_feats   # [N, MAX_SRC, K, CAND_DIM]
        gf = jnp.broadcast_to(enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask    # [N, MAX_SRC, K]

        outputs = module.apply(
            params,
            sf.reshape(N_ENVS * MAX_SRC, SELF_DIM),
            cf.reshape(N_ENVS * MAX_SRC, K, CAND_DIM),
            gf.reshape(N_ENVS * MAX_SRC, GLOB_DIM),
            cm.reshape(N_ENVS * MAX_SRC, K),
        )
        key, subkey = jax.random.split(key)
        ti_f, sb_f, lp_f, _, _ = sample_actions(outputs, subkey, deterministic=deterministic)

        ti  = ti_f.reshape(N_ENVS, MAX_SRC)
        sb  = sb_f.reshape(N_ENVS, MAX_SRC)
        lp  = lp_f.reshape(N_ENVS, MAX_SRC)
        val = outputs.value.reshape(N_ENVS, MAX_SRC)
        return ti, sb, lp, val, key

    def _enc_to_actions(enc, ti, sb):
        def one_env(src_ids, src_ships, angles, cand_mask, valid, ti_i, sb_i):
            return _build_actions_single(
                src_ids, src_ships, angles, cand_mask, valid, ti_i, sb_i, ship_bucket_count
            )
        return jax.vmap(one_env)(
            enc.src_ids, enc.src_ships, enc.angles,
            enc.cand_mask, enc.src_valid, ti, sb,
        )  # [N, MAX_ACTIONS_PA, 3]

    def _joint_actions(learner_players, l_acts, o_acts):
        dummy = jnp.full((N_ENVS, N_AGENTS - 2, MAX_ACTIONS_PA, 3), -1.0)

        def one_env(lp_idx, l_act, o_act):
            p0 = jax.lax.cond(lp_idx == 0, lambda: l_act, lambda: o_act)
            p1 = jax.lax.cond(lp_idx == 0, lambda: o_act, lambda: l_act)
            return jnp.stack([p0, p1], axis=0)  # [2, MAX_ACTIONS_PA, 3]

        two = jax.vmap(one_env)(learner_players, l_acts, o_acts)  # [N, 2, ...]
        return jnp.concatenate([two, dummy], axis=1)               # [N, N_AGENTS, ...]

    def _reward_and_done(learner_players, statuses, rewards):
        def one_env(lp_idx, sts, rws):
            done_ = sts[lp_idx].astype(jnp.bool_)
            my_r  = rws[lp_idx]
            opp_r = rws[1 - lp_idx]
            r = jnp.where((my_r > 0.0) & (opp_r > 0.0), 0.0, my_r)
            return done_, jnp.where(done_, r, 0.0)
        return jax.vmap(one_env)(learner_players, statuses, rewards)

    # ── scan body ────────────────────────────────────────────────────────────

    def scan_fn(params, opp_params, carry, _):
        state, board_indices, ep_reward, learner_players, key = carry

        # Encode and run learner policy
        l_enc = encode_batch(state, learner_players)
        ti, sb, lp, val, key = _run_policy(params, l_enc, key, deterministic=False)
        l_acts = _enc_to_actions(l_enc, ti, sb)

        # Encode and run opponent policy
        opp_players = 1 - learner_players
        o_enc  = encode_batch(state, opp_players)
        oti, osb, _, _, key = _run_policy(opp_params, o_enc, key, deterministic=opp_deterministic)
        o_acts = _enc_to_actions(o_enc, oti, osb)

        # Physics step
        joint = _joint_actions(learner_players, l_acts, o_acts)
        new_state, statuses, rewards = _batched_step(state, joint)

        # Reward / done
        done, reward = _reward_and_done(learner_players, statuses, rewards)

        # Episode tracking
        ep_reward = ep_reward + reward

        # Reset done envs from pool
        next_boards = select_boards(board_pool, board_indices, n_boards)

        def maybe_reset(env_done, env_state, new_board, lp):
            ns  = jax.lax.cond(env_done, lambda: new_board, lambda: env_state)
            nlp = jax.lax.cond(
                env_done & alternate_sides,
                lambda: jnp.int32(1) - lp,
                lambda: lp,
            )
            return ns, nlp

        next_states, next_lp = jax.vmap(maybe_reset)(
            done, new_state, next_boards, learner_players
        )
        next_board_idx = jnp.where(done, board_indices + 1, board_indices)
        next_ep_reward = jnp.where(done, jnp.zeros_like(ep_reward), ep_reward)

        step = RolloutStep(
            sf=l_enc.self_feats,
            cf=l_enc.cand_feats,
            gf=l_enc.global_feat,
            cm=l_enc.cand_mask,
            ti=ti, sb=sb, lp=lp, val=val,
            src_valid=l_enc.src_valid,
            reward=reward, done=done,
        )
        new_carry = RolloutCarry(next_states, next_board_idx, next_ep_reward, next_lp, key)
        return new_carry, step

    @jax.jit
    def collect(params, opp_params, carry: RolloutCarry):
        def _fn(carry, x):
            return scan_fn(params, opp_params, carry, x)
        return jax.lax.scan(_fn, carry, None, length=rollout_steps)

    return collect


# ---------------------------------------------------------------------------
# Bootstrap value — called in Python after the scan
# ---------------------------------------------------------------------------

def compute_bootstrap_val(
    module,
    params,
    state: JaxState,          # [N, ...]
    learner_players: jnp.ndarray,  # [N]
    encode_batch,
    K: int,
    N_ENVS: int,
) -> jnp.ndarray:
    """Average critic value over valid source planets for each env. Returns [N]."""
    enc = encode_batch(state, learner_players)
    gf  = jnp.broadcast_to(enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM))
    outputs = module.apply(
        params,
        enc.self_feats.reshape(N_ENVS * MAX_SRC, SELF_DIM),
        enc.cand_feats.reshape(N_ENVS * MAX_SRC, K, CAND_DIM),
        gf.reshape(N_ENVS * MAX_SRC, GLOB_DIM),
        enc.cand_mask.reshape(N_ENVS * MAX_SRC, K),
    )
    val   = outputs.value.reshape(N_ENVS, MAX_SRC)
    valid = enc.src_valid.astype(jnp.float32)
    count = jnp.maximum(valid.sum(axis=1), 1.0)
    return (val * valid).sum(axis=1) / count  # [N]


# ---------------------------------------------------------------------------
# Returns and advantages — JIT-compiled reverse scan
# ---------------------------------------------------------------------------

@jax.jit
def compute_returns(
    reward: jnp.ndarray,  # [T, N]
    done: jnp.ndarray,    # [T, N] bool
    boot_val: jnp.ndarray,  # [N]
    gamma: float,
) -> jnp.ndarray:
    """Return step-level returns [T, N] via reverse discount scan."""
    def scan_fn(future_ret, step):
        r, d = step
        new_fut = r + gamma * future_ret * (1.0 - d.astype(jnp.float32))
        return new_fut, new_fut

    _, rets = jax.lax.scan(scan_fn, boot_val, (reward, done), reverse=True)
    return rets  # [T, N]


# ---------------------------------------------------------------------------
# Build TransitionBatch for PPO — keep rollout tensors as JAX arrays
# ---------------------------------------------------------------------------

def build_transition_batch(steps: RolloutStep, step_returns: jnp.ndarray) -> TransitionBatch:
    """
    Filter padded rows and pack into TransitionBatch.

    steps.* : [T, N, MAX_SRC, ...] (or [T, N] for scalars)
    step_returns : [T, N]  (from compute_returns)
    """
    T, N = step_returns.shape

    sf_all    = steps.sf          # [T, N, MAX_SRC, SELF_DIM]
    cf_all    = steps.cf          # [T, N, MAX_SRC, K, CAND_DIM]
    gf_env    = steps.gf          # [T, N, GLOB_DIM]
    cm_all    = steps.cm          # [T, N, MAX_SRC, K]
    ti_all    = steps.ti          # [T, N, MAX_SRC]
    sb_all    = steps.sb
    lp_all    = steps.lp
    val_all   = steps.val
    valid_all = steps.src_valid   # [T, N, MAX_SRC] bool
    rets_all  = step_returns      # [T, N]

    # Filter indices: (t, n, s) where src_valid
    valid_flat = valid_all.reshape(-1)
    idx = jnp.where(valid_flat)[0]

    sf_f  = sf_all.reshape(-1, sf_all.shape[-1])[idx]
    cf_f  = cf_all.reshape(-1, cf_all.shape[-2], cf_all.shape[-1])[idx]
    cm_f  = cm_all.reshape(-1, cm_all.shape[-1])[idx]
    ti_f  = ti_all.reshape(-1)[idx]
    sb_f  = sb_all.reshape(-1)[idx]
    lp_f  = lp_all.reshape(-1)[idx]
    val_f = val_all.reshape(-1)[idx]

    # For each valid (t, n, s), global feat is gf_env[t, n] and return is rets_all[t, n].
    # tn_idx maps flat decision index → flat (t*N + n) index.
    tn_idx = idx // sf_all.shape[2]              # [N_valid] in [0, T*N)
    gf_f   = gf_env.reshape(T * N, -1)[tn_idx]  # [N_valid, GLOB_DIM]
    ret_f  = rets_all.reshape(-1)[tn_idx]        # [N_valid]
    adv_f  = ret_f - val_f

    return TransitionBatch(
        self_features      = sf_f,
        candidate_features = cf_f,
        global_features    = gf_f,
        candidate_mask     = cm_f,
        target_index       = ti_f.astype(jnp.int32),
        ships_bucket       = sb_f.astype(jnp.int32),
        log_prob           = lp_f,
        returns            = ret_f,
        advantages         = adv_f,
    )


# ---------------------------------------------------------------------------
# Episode stats from scan output
# ---------------------------------------------------------------------------

def episode_stats(steps: RolloutStep) -> dict[str, float]:
    done   = steps.done    # [T, N]
    reward = steps.reward  # [T, N]

    def scan_fn(running, step):
        r, d = step
        running = running + r
        finished = jnp.where(d, running, 0.0)
        running = jnp.where(d, 0.0, running)
        return running, finished

    _, finished_rewards = jax.lax.scan(
        scan_fn,
        jnp.zeros(done.shape[1], dtype=jnp.float32),
        (reward, done),
    )
    episodes_finished = done.sum()
    reward_mean = jnp.where(
        episodes_finished > 0,
        finished_rewards.sum() / episodes_finished,
        0.0,
    )

    return {
        "episode_reward_mean": float(reward_mean),
        "episodes_finished":   float(episodes_finished),
        "samples":             float(steps.src_valid.sum()),
    }
