"""Cross-source candidate-scoring policy (formerly ``PlanetPolicy``).

Per source planet, encodes the K candidate targets with a cross-source
coordination attention block (so sources can form coalitions / avoid
double-committing) plus a candidate transformer block, then scores each
candidate. Separate value net with no shared weights.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..features import self_feature_dim, candidate_feature_dim, global_feature_dim
from .base import PolicyOutput, MASK_LOGIT


class _PolicyHead(nn.Module):
    """Policy network: target_logits. No shared weights with _ValueNet."""
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
        src_valid: jax.Array,           # [N] bool
    ) -> jax.Array:
        H, K = self.hidden_size, self.candidate_count
        N = self_features.shape[0]

        # 2-layer GELU MLP encoders (GELU per RLIAYN) 
        self_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(self_features))))
        global_h = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(global_features))))
        cand_flat = candidate_features.reshape(N * K, candidate_features.shape[-1])
        cand_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

        # Within-turn cross-source coordination: each source attends to all valid
        # sibling sources so it can learn coalition attacks and avoid double-sending.
        # Mask [1,1,1,N] broadcasts to [N, n_heads, N_q, N_k]; invalid sources are
        # masked as keys so padding doesn't pollute the coalition signal.
        cs_mask    = src_valid[None, None, None, :]
        cs_normed  = nn.LayerNorm()(self_h)
        self_h     = (self_h + nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, out_features=H,
        )(cs_normed, mask=cs_mask)).reshape(N, H)
        cs_normed2 = nn.LayerNorm()(self_h)
        self_h     = self_h + nn.Dense(H)(nn.gelu(nn.Dense(4 * H)(cs_normed2)))

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

        return target_logits


class _ValueNet(nn.Module):
    """Value network: fully separate parameters from _PolicyHead."""
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
        src_valid: jax.Array,           # [N] bool
    ) -> jax.Array:
        H, K = self.hidden_size, self.candidate_count
        N = self_features.shape[0]

        self_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(self_features))))
        global_h = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(global_features))))
        cand_flat = candidate_features.reshape(N * K, candidate_features.shape[-1])
        cand_h   = nn.gelu(nn.Dense(H)(nn.gelu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

        cs_mask    = src_valid[None, None, None, :]
        cs_normed  = nn.LayerNorm()(self_h)
        self_h     = (self_h + nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, out_features=H,
        )(cs_normed, mask=cs_mask)).reshape(N, H)
        cs_normed2 = nn.LayerNorm()(self_h)
        self_h     = self_h + nn.Dense(H)(nn.gelu(nn.Dense(4 * H)(cs_normed2)))

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


class CrossSourcePolicy(nn.Module):
    self_dim: int
    candidate_dim: int
    global_dim: int
    candidate_count: int
    hidden_size: int = 128
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, candidate_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
        src_valid: jax.Array,           # [N] bool
    ) -> PolicyOutput:
        target_logits = _PolicyHead(
            hidden_size     = self.hidden_size,
            candidate_count = self.candidate_count,
            n_heads         = self.n_heads,
        )(self_features, candidate_features, global_features, candidate_mask, src_valid)

        value = _ValueNet(
            hidden_size     = self.hidden_size,
            candidate_count = self.candidate_count,
            n_heads         = self.n_heads,
        )(self_features, candidate_features, global_features, candidate_mask, src_valid)

        return PolicyOutput(
            target_logits = target_logits,
            value         = value,
        )


# Back-compat alias: older code / pickles referenced ``PlanetPolicy``.
PlanetPolicy = CrossSourcePolicy


def build(cfg) -> CrossSourcePolicy:
    """Construct a ``CrossSourcePolicy`` from a TrainConfig.

    Architecture dispatch (``cfg.model.arch``) lives in ``policies/__init__.py``;
    this just builds the net. ``init_params`` is the package-level generic.
    """
    if getattr(cfg.model, "shared_backbone", False):
        raise NotImplementedError(
            "model.shared_backbone is only supported by the entity_* arches "
            "(the cross_source net fuses its backbone into each head). "
            "Set arch=entity_mlp/entity_deep_mlp or shared_backbone=false."
        )
    return CrossSourcePolicy(
        self_dim        = self_feature_dim(),
        candidate_dim   = candidate_feature_dim(),
        global_dim      = global_feature_dim(),
        candidate_count = cfg.env.candidate_count,
        hidden_size     = cfg.model.hidden_size,
        n_heads         = getattr(cfg.model, 'n_heads', 4),
    )
