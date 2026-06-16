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

from .physics import (
    EnvState,
    jax_step,
    MAX_BASE_PLANETS,
    MAX_ACTIONS_PA,
    N_AGENTS,
)
from .features import (
    MIN_SOURCE_SHIPS, _make_env_encoder, global_feature_dim,
)
from .feature_versions import make_env_encoder
from .augment import apply_augmentation_batched
from .ppo import TrainingBatch, sample_actions
from .env import select_boards
from .heuristic import heuristic_actions

MAX_SRC  = MAX_BASE_PLANETS   # alias for clarity
GLOB_DIM = global_feature_dim()


# ---------------------------------------------------------------------------
# Carry and step output types
# ---------------------------------------------------------------------------

class RolloutCarry(NamedTuple):
    state: EnvState               # [N, ...]
    board_indices: jnp.ndarray    # [N] int32
    episode_rewards: jnp.ndarray        # [N] float32
    learner_players: jnp.ndarray  # [N] int32
    key: jnp.ndarray


class RolloutStep(NamedTuple):
    self_features: jnp.ndarray        # [N, MAX_SRC, SELF_DIM]
    candidate_features: jnp.ndarray   # [N, MAX_SRC, K, CAND_DIM]
    global_features: jnp.ndarray      # [N, GLOB_DIM]
    candidate_masks: jnp.ndarray      # [N, MAX_SRC, K] bool
    target_indices: jnp.ndarray       # [N, MAX_SRC] int32
    log_probs: jnp.ndarray        # [N, MAX_SRC] float32
    values: jnp.ndarray       # [N, MAX_SRC] float32
    src_valid: jnp.ndarray # [N, MAX_SRC] bool
    reward: jnp.ndarray    # [N] float32
    done: jnp.ndarray      # [N] bool
    ref_target_logits: jnp.ndarray  # [N, MAX_SRC, K] (KL) or [N, 1] placeholder


# ---------------------------------------------------------------------------
# Pure-JAX action builder (one env, no batch dim)
# ---------------------------------------------------------------------------


def _build_actions_single(
    src_ids: jnp.ndarray,         # [MAX_SRC] int32
    cand_mask: jnp.ndarray,       # [MAX_SRC, K] bool — already encodes blocking
    src_valid: jnp.ndarray,       # [MAX_SRC] bool
    ti: jnp.ndarray,              # [MAX_SRC] int32
    ships_send: jnp.ndarray,      # [MAX_SRC] int32
    tgt_angles: jnp.ndarray,      # [MAX_SRC] float32 — precomputed by encoder
) -> jnp.ndarray:
    """Assemble [MAX_ACTIONS_PA, 3] action tensor from precomputed ships + angles."""
    row_idx = jnp.arange(MAX_SRC, dtype=jnp.int32)

    tgt_ok    = (ti != 0) & cand_mask[row_idx, ti]
    action_ok = src_valid & tgt_ok                               # [MAX_SRC]

    n_act = min(MAX_SRC, MAX_ACTIONS_PA)
    rows  = jnp.stack([
        src_ids[:n_act].astype(jnp.float32),
        tgt_angles[:n_act],
        ships_send[:n_act].astype(jnp.float32),
    ], axis=1)                                                    # [n_act, 3]
    rows  = jnp.where(action_ok[:n_act, None], rows, jnp.full((3,), -1.0))

    if n_act < MAX_ACTIONS_PA:
        pad  = jnp.full((MAX_ACTIONS_PA - n_act, 3), -1.0, jnp.float32)
        rows = jnp.concatenate([rows, pad], axis=0)

    return rows  # [MAX_ACTIONS_PA, 3]


# ---------------------------------------------------------------------------
# Factory: returns a JIT-compiled collect() function
# ---------------------------------------------------------------------------

def make_rollout_fn(
    board_pool: EnvState,
    n_boards: int,
    cfg,
    module,
    rollout_steps: int,
    n_opponents: int = 1,
    heuristic_opp_mask: tuple[bool, ...] = (),
    augment: bool = False,
    opp_encode_batch=None,
    ref_params=None,
    ref_encode_batch=None,
    compute_ref_logits: bool = False,
):
    """
    Build and return a JIT-compiled rollout collector.

    Parameters
    ----------
    board_pool          : JaxState[n_boards, ...]
    n_boards            : int
    cfg                 : TrainConfig
    module              : PlanetPolicy (Flax module)
    rollout_steps       : int (static — baked into the JIT)
    n_opponents         : int (static) — opponent pool size; env i uses pool[i % n_opponents]
    heuristic_opp_mask  : tuple[bool] length n_opponents — True for heuristic slots.
                          Heuristic opponents bypass neural inference entirely.
                          opp_params slots for heuristic indices are ignored (pass dummies).

    opp_encode_batch    : encoder for neural opponents (cross-version play). None =
                          use the learner's encoder. opp_params are run through the
                          same dim-agnostic ``module`` on these features.
    ref_params          : frozen KL-reference params (run through ``module``). When
                          set with compute_ref_logits, the rollout precomputes the
                          ref's target logits on each visited state.
    ref_encode_batch    : encoder for the KL reference (its own feat_version). None =
                          use the learner's encoder.
    compute_ref_logits  : if True, store ref target logits [N, MAX_SRC, K] per step;
                          else store a [N, 1] placeholder.

    Returns
    -------
    collect(params, opp_params, carry) -> (final_carry, RolloutStep[T, N, ...])
    opp_params shape: [n_opponents, ...] stacked param tree
    """
    K                  = cfg.env.candidate_count
    N_ENVS             = cfg.ppo.num_envs
    episode_steps      = cfg.env.episode_steps
    alternate_sides    = cfg.alternate_player_sides
    opp_deterministic  = cfg.self_play_deterministic
    fleet_penalty      = jnp.float32(cfg.ppo.fleet_penalty)
    capture_bonus      = jnp.float32(cfg.ppo.capture_bonus)
    capture_time_weight = jnp.float32(cfg.ppo.capture_time_weight)

    # Players seated per game (2 = 1v1, 4 = FFA). The learner holds one seat and a
    # single shared opponent policy controls the other n_opp = num_agents - 1 seats.
    # num_agents is static (python int) so the opponent-slot loop and joint-action
    # scatter unroll at JIT time; N_AGENTS (=4) stays the physical array width.
    num_agents = int(cfg.env.num_agents)
    n_opp      = num_agents - 1
    assert num_agents in (2, 4), f"num_agents must be 2 or 4, got {num_agents}"
    assert num_agents <= N_AGENTS, f"num_agents {num_agents} exceeds physics N_AGENTS {N_AGENTS}"

    if not heuristic_opp_mask:
        heuristic_opp_mask = (False,) * n_opponents
    _any_heuristic = any(heuristic_opp_mask)
    _any_neural    = not all(heuristic_opp_mask)
    # Static bool array for per-env selection in the mixed case (baked into JIT).
    _heuristic_arr = jnp.array(heuristic_opp_mask, dtype=jnp.bool_)

    encode_batch = make_env_encoder(K, cfg.env, getattr(cfg.model, "feat_version", "v7"))
    # Opponent / reference encoders default to the learner's (same-version play).
    opp_encode_batch = opp_encode_batch or encode_batch
    ref_encode_batch = ref_encode_batch or encode_batch
    _compute_ref = bool(compute_ref_logits and ref_params is not None)
    _batched_step = jax.jit(jax.vmap(functools.partial(jax_step, episode_steps=episode_steps)))
    _vmap_heuristic = jax.vmap(heuristic_actions)

    # ── helpers (closed over) ────────────────────────────────────────────────

    def _run_policy(params, enc, key, deterministic):
        sf = enc.self_feats   # [N, MAX_SRC, SELF_DIM]
        cf = enc.cand_feats   # [N, MAX_SRC, K, CAND_DIM]
        gf = jnp.broadcast_to(enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask    # [N, MAX_SRC, K]
        sv = enc.src_valid    # [N, MAX_SRC] bool

        # vmap over envs so cross-source attention stays within each env's turn.
        out_per_env = jax.vmap(
            lambda sf_i, cf_i, gf_i, cm_i, sv_i: module.apply(params, sf_i, cf_i, gf_i, cm_i, sv_i)
        )(sf, cf, gf, cm, sv)                                               # [N, MAX_SRC, ...]
        outputs = jax.tree_util.tree_map(
            lambda x: x.reshape(N_ENVS * MAX_SRC, *x.shape[2:]), out_per_env
        )
        key, subkey = jax.random.split(key)
        ti_f, lp_f, _, _ = sample_actions(outputs, subkey, deterministic=deterministic)

        ti  = ti_f.reshape(N_ENVS, MAX_SRC)
        lp  = lp_f.reshape(N_ENVS, MAX_SRC)
        val = outputs.value.reshape(N_ENVS, MAX_SRC)
        return ti, lp, val, key

    def _run_opp_policy(stacked_opp_params, enc, key, deterministic):
        # stacked_opp_params: [n_opponents, ...] — select per env via round-robin
        opp_idx = jnp.arange(N_ENVS, dtype=jnp.int32) % n_opponents
        per_env_params = jax.tree_util.tree_map(lambda p: p[opp_idx], stacked_opp_params)

        sf = enc.self_feats  # [N, MAX_SRC, SELF_DIM]
        cf = enc.cand_feats  # [N, MAX_SRC, K, CAND_DIM]
        gf = jnp.broadcast_to(enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM))
        cm = enc.cand_mask   # [N, MAX_SRC, K]
        sv = enc.src_valid   # [N, MAX_SRC] bool

        outputs = jax.vmap(
            lambda p, sf_i, cf_i, gf_i, cm_i, sv_i: module.apply(p, sf_i, cf_i, gf_i, cm_i, sv_i)
        )(per_env_params, sf, cf, gf, cm, sv)                               # [N, MAX_SRC, ...]

        flat = jax.tree_util.tree_map(lambda x: x.reshape(N_ENVS * MAX_SRC, *x.shape[2:]), outputs)
        key, subkey = jax.random.split(key)
        ti_f, _, _, key = sample_actions(flat, subkey, deterministic=deterministic)
        return ti_f.reshape(N_ENVS, MAX_SRC), key

    def _enc_to_actions(enc, ti):
        def one_env(src_ids, src_ships, cand_mask, angles, valid, ti_i):
            row_idx    = jnp.arange(MAX_SRC, dtype=jnp.int32)
            ships_send = jnp.maximum(MIN_SOURCE_SHIPS, src_ships)
            tgt_angles = angles[row_idx, ti_i]
            return _build_actions_single(src_ids, cand_mask, valid, ti_i, ships_send, tgt_angles)
        return jax.vmap(one_env)(
            enc.src_ids, enc.src_ships, enc.cand_mask, enc.angles, enc.src_valid, ti,
        )  # [N, MAX_ACTIONS_PA, 3]

    def _opp_player_ids(lp):
        """Ordered ids of the n_opp seats that aren't the learner's, for one env.

        j in [0, n_opp) maps to player id ``j + (j >= lp)`` — i.e. arange(num_agents)
        with the learner's seat removed. e.g. num_agents=4, lp=2 -> [0, 1, 3].
        """
        j = jnp.arange(n_opp, dtype=jnp.int32)
        return j + (j >= lp).astype(jnp.int32)

    def _joint_actions(learner_players, l_acts, o_acts):
        # l_acts: [N, MAX_ACTIONS_PA, 3]; o_acts: [N, n_opp, MAX_ACTIONS_PA, 3]
        # Scatter the learner action into its seat and each opponent action into its
        # seat, leaving phantom seats (>= num_agents) as no-ops (-1).
        def one_env(lp, l_act, o_act):
            full = jnp.full((N_AGENTS, MAX_ACTIONS_PA, 3), -1.0, jnp.float32)
            full = full.at[lp].set(l_act)
            full = full.at[_opp_player_ids(lp)].set(o_act)
            return full  # [N_AGENTS, MAX_ACTIONS_PA, 3]
        return jax.vmap(one_env)(learner_players, l_acts, o_acts)  # [N, N_AGENTS, ...]

    def _reward_and_done(learner_players, statuses, rewards):
        # FFA-general: physics gives +1 to every co-winner. Mirror the 1v1 "a shared
        # win is a draw → 0 reward" rule by zeroing the learner's reward whenever it
        # is one of multiple winners. For num_agents=2 this is exactly the old
        # "both win → 0" behaviour (n_winners == 2).
        def one_env(lp_idx, sts, rws):
            done_     = sts[lp_idx].astype(jnp.bool_)
            my_r      = rws[lp_idx]
            n_winners = (rws > 0.0).sum()
            r = jnp.where((my_r > 0.0) & (n_winners > 1), 0.0, my_r)
            return done_, jnp.where(done_, r, 0.0)
        return jax.vmap(one_env)(learner_players, statuses, rewards)

    # ── scan body ────────────────────────────────────────────────────────────

    def scan_fn(params, opp_params, carry, _):
        state, board_indices, ep_reward, learner_players, key = carry

        # Encode and run learner policy
        l_enc = encode_batch(state, learner_players)
        if augment:
            key, aug_key = jax.random.split(key)
            aug_ids = jax.random.randint(aug_key, (N_ENVS,), 0, 8)
            sf_aug, cf_aug, gf_aug = apply_augmentation_batched(
                l_enc.self_feats, l_enc.cand_feats, l_enc.global_feat, aug_ids,
            )
            l_enc = l_enc._replace(self_feats=sf_aug, cand_feats=cf_aug, global_feat=gf_aug)
        ti, lp, val, key = _run_policy(params, l_enc, key, deterministic=False)
        l_acts = _enc_to_actions(l_enc, ti)

        # Frozen KL-reference target logits on the SAME visited state. The ref uses
        # its own-version encoder and (when augmenting) the SAME aug_ids as the
        # learner, so ref and learner logits describe the same (augmented) board and
        # are slot-aligned. Run through the dim-agnostic module with ref_params.
        if _compute_ref:
            ref_enc = ref_encode_batch(state, learner_players)
            if augment:
                rsf, rcf, rgf = apply_augmentation_batched(
                    ref_enc.self_feats, ref_enc.cand_feats, ref_enc.global_feat, aug_ids,
                )
                ref_enc = ref_enc._replace(self_feats=rsf, cand_feats=rcf, global_feat=rgf)
            ref_gf = jnp.broadcast_to(
                ref_enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM)
            )
            ref_out = jax.vmap(
                lambda sf_i, cf_i, gf_i, cm_i, sv_i: module.apply(
                    ref_params, sf_i, cf_i, gf_i, cm_i, sv_i
                )
            )(ref_enc.self_feats, ref_enc.cand_feats, ref_gf, ref_enc.cand_mask, ref_enc.src_valid)
            ref_tgt = ref_out.target_logits          # [N, MAX_SRC, K]
        else:
            ref_tgt = jnp.zeros((N_ENVS, 1), jnp.float32)  # placeholder (unused)

        # Opponent actions. Per env there are n_opp = num_agents - 1 opposing seats,
        # each controlled by the same (shared) opponent policy but encoded from its
        # own egocentric viewpoint. We unroll over the seats (n_opp is static) so the
        # heuristic/neural branch selection is resolved once at JIT time. Each seat j
        # of every env holds player id `j + (j >= learner_player)`.
        # opp_player_ids: [N, n_opp]
        opp_player_ids = jax.vmap(_opp_player_ids)(learner_players)

        def _opp_actions(opp_players_j, key):
            """Actions [N, MAX_ACTIONS_PA, 3] for one opponent seat across all envs."""
            if _any_heuristic:
                h_acts = _vmap_heuristic(state, opp_players_j)

            if _any_neural:
                o_enc = opp_encode_batch(state, opp_players_j)
                if n_opponents == 1:
                    # All envs share the same params — use the efficient batched forward.
                    single_opp = jax.tree_util.tree_map(lambda p: p[0], opp_params)
                    oti, _, _, key = _run_policy(single_opp, o_enc, key, deterministic=opp_deterministic)
                else:
                    oti, key = _run_opp_policy(opp_params, o_enc, key, deterministic=opp_deterministic)
                n_acts = _enc_to_actions(o_enc, oti)

            if _any_heuristic and _any_neural:
                # Mixed pool: select per-env based on which opponent slot is assigned.
                opp_slot = jnp.arange(N_ENVS, dtype=jnp.int32) % n_opponents
                env_is_heuristic = _heuristic_arr[opp_slot]   # [N]
                acts = jnp.where(env_is_heuristic[:, None, None], h_acts, n_acts)
            elif _any_heuristic:
                acts = h_acts
            else:
                acts = n_acts
            return acts, key

        o_acts_slots = []
        for j in range(n_opp):
            acts_j, key = _opp_actions(opp_player_ids[:, j], key)
            o_acts_slots.append(acts_j)
        o_acts = jnp.stack(o_acts_slots, axis=1)  # [N, n_opp, MAX_ACTIONS_PA, 3]

        # Physics step
        joint = _joint_actions(learner_players, l_acts, o_acts)
        new_state, statuses, rewards = _batched_step(state, joint)

        # Reward / done
        done, reward = _reward_and_done(learner_players, statuses, rewards)

        # Reward shaping (skipped when both coefficients are 0)
        def _shape_one(prev_s, next_s, lp, acts):
            # Fleet dispatch penalty: count non-no-op actions (planet_id >= 0)
            n_dispatches = (acts[:, 0] >= 0).astype(jnp.float32).sum()
            penalty = -fleet_penalty * n_dispatches

            # Capture bonus: planets that flipped to mine this tick
            was_not_mine = (prev_s.p_owner != lp) & prev_s.p_alive
            now_mine     = (next_s.p_owner == lp)
            n_captures   = (was_not_mine & now_mine).astype(jnp.float32).sum()
            t_frac       = prev_s.step.astype(jnp.float32) / jnp.float32(episode_steps)
            time_mult    = jnp.float32(1.0) + capture_time_weight * (jnp.float32(1.0) - t_frac)
            bonus = capture_bonus * n_captures * time_mult

            return penalty + bonus

        shaping = jax.vmap(_shape_one)(state, new_state, learner_players, l_acts)
        reward  = reward + jnp.where(done, jnp.zeros_like(shaping), shaping)

        # Episode tracking
        ep_reward = ep_reward + reward

        # Reset done envs from pool
        next_boards = select_boards(board_pool, board_indices, n_boards)

        def maybe_reset(env_done, env_state, new_board, lp):
            ns  = jax.lax.cond(env_done, lambda: new_board, lambda: env_state)
            # Rotate the learner's seat on reset so every seat is trained equally.
            # Use the NEW board's own num_agents so a mixed 2p/4p pool always lands
            # on a valid (non-phantom) seat. For uniform num_agents=2 this is the
            # original 0<->1 swap.
            nlp = jax.lax.cond(
                env_done & alternate_sides,
                lambda: (lp + 1) % new_board.num_agents,
                lambda: lp,
            )
            return ns, nlp

        next_states, next_lp = jax.vmap(maybe_reset)(
            done, new_state, next_boards, learner_players
        )
        next_board_idx = jnp.where(done, board_indices + 1, board_indices)
        next_ep_reward = jnp.where(done, jnp.zeros_like(ep_reward), ep_reward)

        step = RolloutStep(
            self_features=l_enc.self_feats,
            candidate_features=l_enc.cand_feats,
            global_features=l_enc.global_feat,
            candidate_masks=l_enc.cand_mask,
            target_indices=ti, log_probs=lp, values=val,
            src_valid=l_enc.src_valid,
            reward=reward, done=done,
            ref_target_logits=ref_tgt,
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
    policy,
    params,
    state: EnvState,          # [N, ...]
    learner_players: jnp.ndarray,  # [N]
    encode_batch,
    K: int,
    N_ENVS: int,
) -> jnp.ndarray:
    """Average critic value over valid source planets for each env. Returns [N]."""
    enc = encode_batch(state, learner_players)
    gf  = jnp.broadcast_to(enc.global_feat[:, None, :], (N_ENVS, MAX_SRC, GLOB_DIM))
    sv  = enc.src_valid   # [N_ENVS, MAX_SRC] bool
    out_per_env = jax.vmap(
        lambda sf_i, cf_i, gf_i, cm_i, sv_i: policy.apply(params, sf_i, cf_i, gf_i, cm_i, sv_i)
    )(enc.self_feats, enc.cand_feats, gf, enc.cand_mask, sv)
    val   = out_per_env.value                                               # [N_ENVS, MAX_SRC]
    valid = sv.astype(jnp.float32)
    count = jnp.maximum(valid.sum(axis=1), 1.0)
    return (val * valid).sum(axis=1) / count  # [N]


# ---------------------------------------------------------------------------
# Returns and advantages — JIT-compiled reverse scan
# ---------------------------------------------------------------------------

@jax.jit
def compute_returns(
    reward: jnp.ndarray,    # [T, N]
    done: jnp.ndarray,      # [T, N] bool
    values: jnp.ndarray,    # [T, N, MAX_SRC] per-source critic values
    src_valid: jnp.ndarray, # [T, N, MAX_SRC] bool
    boot_val: jnp.ndarray,  # [N] env-level bootstrap value
    gamma: float,
    gae_lambda: float,
) -> jnp.ndarray:
    """Step-level λ-returns [T, N] via reverse GAE scan.

    Computes the TD(λ) / GAE return target
        R_t^λ = r_t + γ·[(1−λ)·V_{t+1} + λ·R_{t+1}^λ]·(1−done_t)
    so the value-function target blends the (low-variance) one-step bootstrap
    with the (low-bias) Monte-Carlo return. λ=1 recovers the pure Monte-Carlo
    return (the previous behaviour); λ<1 leans on the critic to cut variance,
    which matters most with short rollout horizons where the full return is
    truncated early. The per-source advantage downstream stays R_t − V_source,
    so this is a drop-in change to the return target only.

    V_t is the env-level critic value = mean over valid source planets (same
    aggregation as ``boot_val`` in ``compute_bootstrap_val``).
    """
    valid = src_valid.astype(jnp.float32)
    count = jnp.maximum(valid.sum(axis=-1), 1.0)
    v_env = (values * valid).sum(axis=-1) / count   # [T, N]

    def scan_fn(carry, step):
        next_v, next_ret = carry            # V_{t+1}, R_{t+1}^λ
        r, d, v_t = step
        nonterminal = 1.0 - d.astype(jnp.float32)
        blended = (1.0 - gae_lambda) * next_v + gae_lambda * next_ret
        new_ret = r + gamma * blended * nonterminal
        return (v_t, new_ret), new_ret

    init = (boot_val, boot_val)
    _, rets = jax.lax.scan(scan_fn, init, (reward, done, v_env), reverse=True)
    return rets  # [T, N]


# ---------------------------------------------------------------------------
# Build TrainingBatch for PPO — keep rollout tensors as JAX arrays
# ---------------------------------------------------------------------------

def build_training_batch(steps: RolloutStep, step_returns: jnp.ndarray) -> TrainingBatch:
    """
    Pack rollout steps into a TrainingBatch, preserving per-env grouping.

    Keeping [B, MAX_SRC, ...] structure (B = T*N) ensures the cross-source
    attention in _minibatch_update operates within each env's planets rather
    than across unrelated decisions from different environments.

    steps.* : [T, N, MAX_SRC, ...] (or [T, N] for env-level fields)
    step_returns : [T, N]
    """
    T, N = step_returns.shape
    B    = T * N

    def _r(x):
        return x.reshape(B, *x.shape[2:])

    sf  = _r(steps.self_features)       # [B, MAX_SRC, self_dim]
    cf  = _r(steps.candidate_features)  # [B, MAX_SRC, K, cand_dim]
    gf  = _r(steps.global_features)     # [B, glob_dim]
    cm  = _r(steps.candidate_masks)     # [B, MAX_SRC, K]
    ti  = _r(steps.target_indices)      # [B, MAX_SRC]
    lp  = _r(steps.log_probs)           # [B, MAX_SRC]
    val = _r(steps.values)              # [B, MAX_SRC]
    sv  = _r(steps.src_valid)           # [B, MAX_SRC]
    rtl = _r(steps.ref_target_logits)   # [B, MAX_SRC, K] (KL) or [B, 1] placeholder
    ret = step_returns.reshape(B)       # [B]

    return TrainingBatch(
        self_features      = sf,
        candidate_features = cf,
        global_features    = gf,
        candidate_mask     = cm,
        target_index       = ti.astype(jnp.int32),
        log_prob           = lp,
        returns            = ret,
        advantages         = ret[:, None] - val,  # [B, MAX_SRC]
        src_valid          = sv,
        ref_target_logits  = rtl,
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
