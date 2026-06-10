"""Entity-transformer policy family.

All variants share the :mod:`~src_jax.policies.backbone` EntityTransformer
encoder + cross-source mixer + multi-query value head, and differ only in the
candidate-scoring head:

  * :class:`EntityMLPPolicy`     (arch ``"entity_mlp"``)      — single-hidden-layer
    joint MLP over ``[self, global, cand]`` (the original entity head).
  * :class:`EntityDeepMLPPolicy` (arch ``"entity_deep_mlp"``) — deeper residual
    joint MLP over the same input; same backbone, more head capacity.

Same call signature / ``PolicyOutput`` as every other policy here, so they are
selectable purely via ``cfg.model.arch``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..features import self_feature_dim, candidate_feature_dim, global_feature_dim
from .base import PolicyOutput, MASK_LOGIT
from .backbone import EntityEncoder, CrossSourceMixer, MultiQueryValueHead


class _EntityBackbonePolicy(nn.Module):
    """Shared encode + value plumbing; subclasses implement ``_score``.

    By default the policy and value branches use fully separate encoders (no
    shared weights), matching the upstream ActorCritic's split between action
    heads and the ValueHead. With ``shared_backbone=True`` the value head reuses
    the policy backbone (encoder + cross-source mixer) and only the heads stay
    separate — one encode pass instead of two. ``value_stop_grad`` then controls
    whether the value loss is allowed to reshape that shared backbone.
    """

    self_dim: int
    candidate_dim: int
    global_dim: int
    candidate_count: int
    hidden_size: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    value_queries: int = 4
    shared_backbone: bool = False
    value_stop_grad: bool = False

    def _score(
        self,
        self_emb: jax.Array,    # [N, H]
        global_emb: jax.Array,  # [N, H]
        cand_emb: jax.Array,    # [N, K, H]
    ) -> jax.Array:             # [N, K]
        raise NotImplementedError

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, candidate_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
        src_valid: jax.Array,           # [N] bool
    ) -> PolicyOutput:
        H = self.hidden_size

        # ── Policy branch ─────────────────────────────────────────────────
        p_self, p_global, p_cand = EntityEncoder(
            d_model=H, n_layers=self.n_layers, n_heads=self.n_heads, ff_dim=self.ff_dim,
            name="policy_encoder",
        )(self_features, global_features, candidate_features, candidate_mask)
        p_self = CrossSourceMixer(d_model=H, n_heads=self.n_heads, name="policy_xsrc")(
            p_self, src_valid
        )

        target_logits = self._score(p_self, p_global, p_cand)               # [N, K]
        # Finite floor (not -inf) keeps entropy / log_softmax NaN-free.
        target_logits = jnp.where(candidate_mask, target_logits, MASK_LOGIT)

        # ── Value branch ──────────────────────────────────────────────────
        if self.shared_backbone:
            # Reuse the policy backbone embeddings (no value_encoder/value_xsrc
            # params). Optionally detach so the value loss can't reshape the
            # shared backbone — the value head still trains, the backbone is
            # driven only by the policy objective.
            v_self, v_global, v_cand = p_self, p_global, p_cand
            if self.value_stop_grad:
                v_self   = jax.lax.stop_gradient(v_self)
                v_global = jax.lax.stop_gradient(v_global)
                v_cand   = jax.lax.stop_gradient(v_cand)
        else:
            # Fully separate value encoder (default).
            v_self, v_global, v_cand = EntityEncoder(
                d_model=H, n_layers=self.n_layers, n_heads=self.n_heads, ff_dim=self.ff_dim,
                name="value_encoder",
            )(self_features, global_features, candidate_features, candidate_mask)
            v_self = CrossSourceMixer(d_model=H, n_heads=self.n_heads, name="value_xsrc")(
                v_self, src_valid
            )
        value = MultiQueryValueHead(
            d_model=H, n_queries=self.value_queries, n_heads=self.n_heads, hidden=H,
            name="value_head",
        )(v_self, v_global, v_cand, candidate_mask)

        return PolicyOutput(target_logits=target_logits, value=value)

    # -- shared helper: broadcast (self, global) against the K candidate axis --
    def _joint(self, self_emb, global_emb, cand_emb):
        N, K, H = cand_emb.shape
        return jnp.concatenate([
            jnp.broadcast_to(self_emb[:, None, :],   (N, K, H)),
            jnp.broadcast_to(global_emb[:, None, :], (N, K, H)),
            cand_emb,
        ], axis=-1)                                                          # [N, K, 3H]


class EntityMLPPolicy(_EntityBackbonePolicy):
    """Single-hidden-layer joint MLP scoring head."""

    def _score(self, self_emb, global_emb, cand_emb):
        H = self.hidden_size
        joint = self._joint(self_emb, global_emb, cand_emb)
        return nn.Dense(1, name="score")(
            nn.gelu(nn.Dense(H, name="score_hidden")(joint))
        ).squeeze(-1)                                                        # [N, K]


class EntityDeepMLPPolicy(_EntityBackbonePolicy):
    """Deeper residual joint MLP scoring head (same backbone, more head capacity).

    Projects ``[self, global, cand]`` to ``H`` then applies ``score_blocks``
    pre-LN residual MLP blocks before the final scalar score.
    """

    score_blocks: int = 2

    def _score(self, self_emb, global_emb, cand_emb):
        H = self.hidden_size
        x = nn.gelu(nn.Dense(H, name="score_in")(self._joint(self_emb, global_emb, cand_emb)))
        for i in range(self.score_blocks):
            h = nn.LayerNorm(name=f"score_ln{i}")(x)
            h = nn.gelu(nn.Dense(H, name=f"score_fc{i}a")(h))
            h = nn.Dense(H, name=f"score_fc{i}b")(h)
            x = x + h                                                        # residual
        return nn.Dense(1, name="score")(x).squeeze(-1)                     # [N, K]


# Back-compat alias: the first entity policy was simply ``EntityPolicy``.
EntityPolicy = EntityMLPPolicy


def _common_kwargs(cfg) -> dict:
    hidden = cfg.model.hidden_size
    ff_dim = getattr(cfg.model, "ff_dim", 0) or 2 * hidden   # 0 → default 2*hidden
    return dict(
        self_dim        = self_feature_dim(),
        candidate_dim   = candidate_feature_dim(),
        global_dim      = global_feature_dim(),
        candidate_count = cfg.env.candidate_count,
        hidden_size     = hidden,
        n_heads         = getattr(cfg.model, "n_heads", 4),
        n_layers        = getattr(cfg.model, "n_layers", 2),
        ff_dim          = ff_dim,
        value_queries   = getattr(cfg.model, "value_queries", 4),
        shared_backbone = getattr(cfg.model, "shared_backbone", False),
        value_stop_grad = getattr(cfg.model, "value_stop_grad", False),
    )


def build_mlp(cfg) -> EntityMLPPolicy:
    return EntityMLPPolicy(**_common_kwargs(cfg))


def build_deep_mlp(cfg) -> EntityDeepMLPPolicy:
    return EntityDeepMLPPolicy(
        score_blocks=getattr(cfg.model, "score_blocks", 2),
        **_common_kwargs(cfg),
    )
