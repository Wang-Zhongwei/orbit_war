from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .features import self_feature_dim, candidate_feature_dim, global_feature_dim


# Finite floor used to mask invalid candidates in the target logits. Must be
# very negative (prob≈0) but finite so entropy/log_softmax stay NaN-free.
MASK_LOGIT = -1e9


class PolicyOutput(NamedTuple):
    target_logits: jax.Array   # [N, K]
    ships_logits: jax.Array    # [N, K, B]
    value: jax.Array           # [N]


class _PolicyHead(nn.Module):
    """Policy network: target_logits and ships_logits. No shared weights with _ValueNet."""
    hidden_size: int
    candidate_count: int
    ship_bucket_count: int
    n_heads: int

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, cand_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
    ) -> tuple[jax.Array, jax.Array]:
        H, K = self.hidden_size, self.candidate_count
        N = self_features.shape[0]

        # 2-layer GELU MLP encoders (GELU per RLIAYN)
        self_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(self_features))))
        global_h = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(global_features))))
        cand_flat = candidate_features.reshape(N * K, candidate_features.shape[-1])
        cand_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

        # Inject source + global context into candidate tokens before attention.
        ctx = (
            nn.Dense(H)(jnp.broadcast_to(self_h[:, None, :],   (N, K, H)))
            + nn.Dense(H)(jnp.broadcast_to(global_h[:, None, :], (N, K, H)))
        )
        cand_h = cand_h + ctx                                                   # [N, K, H]

        # Full transformer block: Pre-LN attention + Pre-LN FFN.
        # Pre-LN (LN before sub-layer) is more stable than post-LN during
        # early training when gradients can be large.
        attn_mask = candidate_mask[:, None, None, :]                            # [N, 1, 1, K]
        normed    = nn.LayerNorm()(cand_h)
        attn      = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, out_features=H,
        )(normed, mask=attn_mask)
        cand_h    = cand_h + attn                                               # attention residual

        normed2   = nn.LayerNorm()(cand_h)
        ffn       = nn.Dense(H)(nn.gelu(nn.Dense(4 * H)(normed2)))
        cand_h    = cand_h + ffn                                                # FFN residual [N, K, H]

        # Score each candidate using the full joint context.
        joint = jnp.concatenate([
            jnp.broadcast_to(self_h[:, None, :],   (N, K, H)),
            jnp.broadcast_to(global_h[:, None, :], (N, K, H)),
            cand_h,
        ], axis=-1)                                                             # [N, K, 3H]

        target_logits = nn.Dense(1)(nn.gelu(nn.Dense(H)(joint))).squeeze(-1)   # [N, K]
        # Mask invalid candidates with a large *finite* negative instead of -inf.
        # -inf makes softmax prob exactly 0, but then the entropy term
        # exp(log_p) * log_p = 0 * -inf = NaN, which poisons the PPO loss and
        # every gradient after it. A finite floor gives prob≈0 with finite log_p.
        target_logits = jnp.where(candidate_mask, target_logits, MASK_LOGIT)
        ships_logits  = nn.Dense(self.ship_bucket_count)(
            nn.gelu(nn.Dense(H)(joint))
        )                                                                       # [N, K, B]

        return target_logits, ships_logits


class _ValueNet(nn.Module):
    """Value network: fully separate parameters from _PolicyHead.
    Receives stop_gradient inputs so value gradients cannot reach policy weights."""
    hidden_size: int
    candidate_count: int
    n_heads: int

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, cand_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
    ) -> jax.Array:
        H, K = self.hidden_size, self.candidate_count
        N = self_features.shape[0]

        self_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(self_features))))
        global_h = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(global_features))))
        cand_flat = candidate_features.reshape(N * K, candidate_features.shape[-1])
        cand_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

        ctx = (
            nn.Dense(H)(jnp.broadcast_to(self_h[:, None, :],   (N, K, H)))
            + nn.Dense(H)(jnp.broadcast_to(global_h[:, None, :], (N, K, H)))
        )
        cand_h = cand_h + ctx

        attn_mask = candidate_mask[:, None, None, :]
        normed    = nn.LayerNorm()(cand_h)
        attn      = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, out_features=H,
        )(normed, mask=attn_mask)
        cand_h    = cand_h + attn

        normed2   = nn.LayerNorm()(cand_h)
        ffn       = nn.Dense(H)(nn.gelu(nn.Dense(4 * H)(normed2)))
        cand_h    = cand_h + ffn                                                # [N, K, H]

        pooled   = cand_h.mean(axis=1)                                          # [N, H]
        value_in = jnp.concatenate([self_h, global_h, pooled], axis=-1)        # [N, 3H]
        return nn.Dense(1)(nn.gelu(nn.Dense(H)(value_in))).squeeze(-1)          # [N]


class PlanetPolicy(nn.Module):
    self_dim: int
    candidate_dim: int
    global_dim: int
    candidate_count: int
    ship_bucket_count: int = 8
    hidden_size: int = 128
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, candidate_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
    ) -> PolicyOutput:
        target_logits, ships_logits = _PolicyHead(
            hidden_size      = self.hidden_size,
            candidate_count  = self.candidate_count,
            ship_bucket_count= self.ship_bucket_count,
            n_heads          = self.n_heads,
        )(self_features, candidate_features, global_features, candidate_mask)

        # stop_gradient ensures value-loss gradients cannot reach policy weights
        # even if a future refactor accidentally shares a sub-module.
        value = _ValueNet(
            hidden_size     = self.hidden_size,
            candidate_count = self.candidate_count,
            n_heads         = self.n_heads,
        )(
            jax.lax.stop_gradient(self_features),
            jax.lax.stop_gradient(candidate_features),
            jax.lax.stop_gradient(global_features),
            candidate_mask,
        )

        return PolicyOutput(
            target_logits = target_logits,
            ships_logits  = ships_logits,
            value         = value,
        )


def build_policy(cfg) -> PlanetPolicy:
    return PlanetPolicy(
        self_dim         = self_feature_dim(),
        candidate_dim    = candidate_feature_dim(),
        global_dim       = global_feature_dim(),
        candidate_count  = cfg.env.candidate_count,
        ship_bucket_count= cfg.env.ship_bucket_count,
        hidden_size      = cfg.model.hidden_size,
        n_heads          = getattr(cfg.model, 'n_heads', 4),
    )


def init_params(module: PlanetPolicy, seed: int = 0) -> dict:
    K = module.candidate_count
    dummy = (
        jnp.zeros((1, module.self_dim)),
        jnp.zeros((1, K, module.candidate_dim)),
        jnp.zeros((1, module.global_dim)),
        jnp.ones((1, K), dtype=bool),
    )
    return module.init(jax.random.PRNGKey(seed), *dummy)
