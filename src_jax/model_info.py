"""
Usage:
    python -m src_jax.model_info --config configs/ppo_default.yaml
"""
from __future__ import annotations

import argparse

import jax

from .config import load_train_config
from .policies import build_policy, init_params
from .features import self_feature_dim, candidate_feature_dim, global_feature_dim
from .physics import MAX_BASE_PLANETS


_SELF_DIM  = self_feature_dim()
_CAND_DIM  = candidate_feature_dim()
_GLOB_DIM  = global_feature_dim()
_MAX_SRC   = MAX_BASE_PLANETS


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _rollout_buffer_bytes(T: int, N: int, K: int) -> int:
    per_env = (
        _MAX_SRC * _SELF_DIM * 4          # self_features      float32
        + _MAX_SRC * K * _CAND_DIM * 4    # candidate_features float32
        + _GLOB_DIM * 4                    # global_features    float32
        + _MAX_SRC * K                     # candidate_masks    bool (1 byte)
        + _MAX_SRC * 4                     # target_indices     int32
        + _MAX_SRC * 4                     # log_probs          float32
        + _MAX_SRC * 4                     # values             float32
        + _MAX_SRC                         # src_valid          bool
        + 4                                # reward             float32
        + 1                                # done               bool
    )
    return T * N * per_env


def _minibatch_bytes(mb: int, K: int, H: int) -> tuple[int, int]:
    """Returns (input_bytes, activation_bytes) for one minibatch forward+backward."""
    # Exact: feature slices passed into _minibatch_update
    input_bytes = mb * (
        _SELF_DIM * 4          # sf
        + K * _CAND_DIM * 4    # cf
        + _GLOB_DIM * 4        # gf
        + K                    # cm  bool
        + 4 * 4                # old_lp, ti, ret, adv  (int32/float32)
    )
    # Approximate: activations stored for backprop across both policy + value heads.
    # Each head has ~2 MLP encoders + 1 transformer block (LN, QKV, FFN) + output head.
    # Dominant cost is K×H per sample across ~12 effective activation tensors per head.
    # ×2 heads, ×2 for backprop pass. XLA fusion reduces this; treat as upper bound.
    activation_bytes = mb * K * H * 12 * 2 * 4  # float32
    return input_bytes, activation_bytes


def model_info(cfg) -> dict[str, object]:
    policy = build_policy(cfg)
    params = init_params(policy, seed=0)

    n_params    = _count_params(params)
    param_bytes = n_params * 4  # float32

    # Adam keeps two moment arrays (same shape as params); gradients add one more copy.
    optimizer_bytes = 3 * param_bytes  # m1 + m2 + grad

    T  = cfg.ppo.rollout_steps
    N  = cfg.ppo.num_envs
    K  = cfg.env.candidate_count
    H  = cfg.model.hidden_size
    mb = cfg.ppo.minibatch_size

    rollout_bytes          = _rollout_buffer_bytes(T, N, K)
    n_opponents            = max(len(cfg.opponents), 1)
    opp_bytes              = n_opponents * param_bytes
    mb_input_bytes, mb_act_bytes = _minibatch_bytes(mb, K, H)

    # Peak occurs during the backward pass of a minibatch update:
    # full rollout buffer + model + optimizer state + minibatch inputs + activations
    total_bytes = (param_bytes + optimizer_bytes + rollout_bytes
                   + opp_bytes + mb_input_bytes + mb_act_bytes)

    def _mb_size(b: int) -> float:
        return b / (1024 ** 2)

    return {
        "n_params":            n_params,
        "param_mb":            _mb_size(param_bytes),
        "optimizer_mb":        _mb_size(optimizer_bytes),
        "rollout_mb":          _mb_size(rollout_bytes),
        "opponent_pool_mb":    _mb_size(opp_bytes),
        "minibatch_input_mb":  _mb_size(mb_input_bytes),
        "minibatch_act_mb":    _mb_size(mb_act_bytes),
        "total_mb":            _mb_size(total_bytes),
    }


def print_model_info(cfg) -> None:
    info = model_info(cfg)
    H  = cfg.model.hidden_size
    K  = cfg.env.candidate_count
    T  = cfg.ppo.rollout_steps
    N  = cfg.ppo.num_envs
    mb = cfg.ppo.minibatch_size

    print(f"Model config:  hidden={H}  K={K}  T={T}  N={N}  mb={mb}  max_src={_MAX_SRC}")
    print(f"  Parameters:            {info['n_params']:>10,}  ({info['param_mb']:.2f} MB)")
    print(f"  Optimizer (Adam):      {'':<10}  ({info['optimizer_mb']:.2f} MB  [grad+m1+m2])")
    print(f"  Rollout buffer:        {'':<10}  ({info['rollout_mb']:.2f} MB)")
    print(f"  Opponent pool:         {'':<10}  ({info['opponent_pool_mb']:.2f} MB  [{len(cfg.opponents)} opponent(s)])")
    print(f"  Minibatch inputs:      {'':<10}  ({info['minibatch_input_mb']:.2f} MB)")
    print(f"  Minibatch activations: {'':<10}  ({info['minibatch_act_mb']:.2f} MB  [approx, upper bound])")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Estimated peak total:  {'':<10}  ({info['total_mb']:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg  = load_train_config(args.config)
    print_model_info(cfg)
