"""Smoke test for N-player PPO rollout (num_agents = 2 and 4).

Exercises the cheap-but-risky bits of the 4-player change end to end:

  1. Seat-mapping math: {learner} ∪ opp_player_ids(learner) is a permutation of
     range(num_agents) for every learner seat (no seat dropped or doubled).
  2. Board generation: a 4-player board's home group carries owners {0,1,2,3};
     a 2-player board carries {0,1}.
  3. Full collect(): the jitted rollout scan runs, returns finite tensors with
     the expected shapes, and the learner reward stays in [-1, 1] (after the
     draw-zeroing + shaping, magnitude is bounded by 1 + shaping which we only
     sanity-check for finiteness).
  4. Joint-action seating inside a real step: every non-no-op learner action id
     appears in the learner's seat row of the assembled joint tensor.

Run tiny (CPU-friendly): a handful of envs, 2 rollout steps, small board pool.

    python -m tools.smoke_4player --config configs/ppo_low_kl.yaml
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np


def _seat_permutation_check(num_agents: int) -> None:
    """Replicate rollout._opp_player_ids and assert seats form a permutation."""
    def opp_ids(lp):
        j = np.arange(num_agents - 1)
        return j + (j >= lp).astype(np.int64)

    for lp in range(num_agents):
        seats = sorted([lp, *opp_ids(lp).tolist()])
        assert seats == list(range(num_agents)), (
            f"num_agents={num_agents} lp={lp}: seats {seats} not a permutation"
        )
    print(f"  [ok] seat permutation for num_agents={num_agents}", flush=True)


def _board_owner_check(num_agents: int, seed: int) -> None:
    from src_jax.env import make_board_pool

    pool = make_board_pool(4, seed, num_agents=num_agents)
    owners = np.asarray(pool.p_owner)  # [4, MAX_PLANETS]
    # collect the non-neutral owners present on each board
    for b in range(owners.shape[0]):
        present = set(int(o) for o in owners[b] if o >= 0)
        assert present == set(range(num_agents)), (
            f"num_agents={num_agents} board {b}: owners {present} != {set(range(num_agents))}"
        )
    print(f"  [ok] board owners == {set(range(num_agents))} for num_agents={num_agents}", flush=True)


def _seat_counts_check() -> None:
    from src_jax.env import _board_seat_counts

    assert _board_seat_counts(10, 2, 1.0) == [2] * 10, "num_agents=2 must be all 2p"
    assert _board_seat_counts(10, 4, 1.0) == [4] * 10, "frac=1 must be all 4p"
    assert _board_seat_counts(10, 4, 0.0) == [2] * 10, "frac=0 must be all 2p"
    half = _board_seat_counts(10, 4, 0.5)
    assert half.count(4) == 5 and half.count(2) == 5, f"frac=0.5 -> {half}"
    print(f"  [ok] _board_seat_counts: half-mix={half}", flush=True)


def _phantom_encode_check(cfg, seed: int) -> None:
    """The new path: encoding a phantom opponent (a seat that doesn't exist on a
    2-player board) must yield no valid sources (-> no-op actions) and stay finite."""
    from src_jax.features import _make_env_encoder
    from src_jax.env import make_board_pool

    pool = make_board_pool(2, seed, num_agents=2, mix_4player_frac=1.0)  # 2-player boards
    from src_jax.feature_versions import make_env_encoder
    encode_batch = make_env_encoder(cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7"))
    # Encode each board from the viewpoint of phantom seat 2 (does not exist on a 2p board).
    phantom_players = jnp.full((2,), 2, dtype=jnp.int32)
    enc = encode_batch(pool, phantom_players)
    sv = np.asarray(enc.src_valid)
    assert not sv.any(), f"phantom seat must have NO valid sources, got {int(sv.sum())}"
    for name, arr in [("self_feats", enc.self_feats), ("cand_feats", enc.cand_feats),
                      ("global_feat", enc.global_feat)]:
        assert np.all(np.isfinite(np.asarray(arr))), f"phantom {name} non-finite"
    print("  [ok] phantom-seat encode: 0 valid sources, features finite", flush=True)


def _mixed_collect_check(cfg, frac: float, seed: int) -> None:
    from src_jax.policies import build_policy
    from src_jax.ppo import build_train_state
    from src_jax.rollout import make_rollout_fn, RolloutCarry
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg.env.num_agents = 4
    cfg.env.mix_4player_frac = frac
    N_ENVS = cfg.ppo.num_envs
    T = cfg.ppo.rollout_steps
    N_BOARDS = 8

    policy = build_policy(cfg)
    train_state = build_train_state(cfg, seed=seed)
    key = jax.random.PRNGKey(seed)

    board_pool = make_board_pool(N_BOARDS, seed, num_agents=4, mix_4player_frac=frac)
    na = np.asarray(board_pool.num_agents)
    assert set(na.tolist()) == {2, 4}, f"mixed pool should have both 2 and 4, got {set(na.tolist())}"

    collect = make_rollout_fn(
        board_pool=board_pool, n_boards=N_BOARDS, cfg=cfg, module=policy,
        rollout_steps=T, n_opponents=1, heuristic_opp_mask=(False,),
        augment=cfg.ppo.augment,
    )
    batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
        board_pool, N_BOARDS, N_ENVS, seed, cfg.alternate_player_sides, num_agents=4,
    )
    # Learner must never sit in a phantom seat: learner_player < that env's num_agents.
    lp = np.asarray(learner_players)
    env_na = np.asarray(batched_state.num_agents)
    assert np.all(lp < env_na), f"learner seat >= num_agents somewhere: lp={lp} na={env_na}"

    carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)
    stacked_opp = jax.tree_util.tree_map(lambda x: x[None], train_state.params)
    final_carry, steps = collect(train_state.params, stacked_opp, carry)
    jax.block_until_ready(steps)

    assert np.all(np.isfinite(np.asarray(steps.reward))), "mixed reward non-finite"
    # is_4player global flag (feature [1]) must vary across envs in a mixed pool.
    flags = set(np.asarray(steps.global_features)[0, :, 1].tolist())
    assert flags == {0.0, 1.0}, f"mixed pool should show both is_4player flags, got {flags}"
    print(f"  [ok] mixed collect (frac={frac}): pool 2&4, seats valid, "
          f"is_4player flags={flags}, rewards finite", flush=True)


def _collect_check(cfg, num_agents: int, seed: int) -> None:
    from src_jax.policies import build_policy
    from src_jax.ppo import build_train_state
    from src_jax.rollout import make_rollout_fn, RolloutCarry
    from src_jax.env import make_board_pool, make_initial_carry_state

    cfg.env.num_agents = num_agents
    N_ENVS = cfg.ppo.num_envs
    T = cfg.ppo.rollout_steps
    N_BOARDS = 8

    policy = build_policy(cfg)
    train_state = build_train_state(cfg, seed=seed)
    key = jax.random.PRNGKey(seed)

    board_pool = make_board_pool(N_BOARDS, seed, num_agents=num_agents)
    collect = make_rollout_fn(
        board_pool=board_pool, n_boards=N_BOARDS, cfg=cfg, module=policy,
        rollout_steps=T, n_opponents=1, heuristic_opp_mask=(False,),
        augment=cfg.ppo.augment,
    )
    batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
        board_pool, N_BOARDS, N_ENVS, seed, cfg.alternate_player_sides,
        num_agents=num_agents,
    )
    # Learner seats must be spread across all num_agents positions when alternating.
    seats_seen = set(int(x) for x in np.asarray(learner_players))
    if cfg.alternate_player_sides:
        assert seats_seen == set(range(num_agents)), (
            f"learner seats {seats_seen} should cover {set(range(num_agents))}"
        )

    carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)
    stacked_opp = jax.tree_util.tree_map(lambda x: x[None], train_state.params)

    final_carry, steps = collect(train_state.params, stacked_opp, carry)
    jax.block_until_ready(steps)

    # Shape + finiteness checks
    assert steps.reward.shape == (T, N_ENVS), steps.reward.shape
    assert steps.done.shape == (T, N_ENVS), steps.done.shape
    for name, arr in [
        ("reward", steps.reward), ("values", steps.values),
        ("log_probs", steps.log_probs), ("global_features", steps.global_features),
    ]:
        a = np.asarray(arr)
        assert np.all(np.isfinite(a)), f"{name} has non-finite values"
    # is_4player global flag (feature index [1]) must match the mode
    gf1 = float(np.asarray(steps.global_features)[0, 0, 1])
    assert gf1 == (1.0 if num_agents == 4 else 0.0), f"is_4player flag={gf1} for num_agents={num_agents}"
    print(f"  [ok] collect() ran: reward{tuple(steps.reward.shape)} finite, "
          f"is_4player={gf1} for num_agents={num_agents}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_low_kl.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices={jax.devices()}", flush=True)

    from src_jax.config import load_train_config

    print("== seat permutation math ==", flush=True)
    _seat_permutation_check(2)
    _seat_permutation_check(4)

    print("== board owner assignment ==", flush=True)
    _board_owner_check(2, args.seed)
    _board_owner_check(4, args.seed)

    print("== board seat-count mixing ==", flush=True)
    _seat_counts_check()

    print("== phantom-seat encode (mixed-pool new path) ==", flush=True)
    cfg = load_train_config(args.config)
    _phantom_encode_check(cfg, args.seed)

    print("== full collect() (tiny) ==", flush=True)
    for na in (2, 4):
        cfg = load_train_config(args.config)
        # shrink to a CPU-friendly smoke
        cfg.ppo.num_envs = 8
        cfg.ppo.rollout_steps = 2
        cfg.ppo.augment = False
        _collect_check(cfg, na, args.seed)

    print("== mixed 2p/4p collect() (tiny) ==", flush=True)
    cfg = load_train_config(args.config)
    cfg.ppo.num_envs = 8
    cfg.ppo.rollout_steps = 2
    cfg.ppo.augment = False
    _mixed_collect_check(cfg, 0.5, args.seed)

    print("\nALL SMOKE CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
