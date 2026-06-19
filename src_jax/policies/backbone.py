"""Shared entity-transformer backbone.

Reusable building blocks for the ``entity_*`` policy family, ported from
lvchao0428/OrbitWarRL ``net/`` (transformer.py + heads.py) and adapted to this
repo's per-source feature layout:

  * :class:`EntityEncoder`     — type-tagged pre-LN self-attention encoder.
  * :class:`CrossSourceMixer`  — cross-source coordination attention.
  * :class:`MultiQueryValueHead` — ``n_queries`` learned queries cross-attend to entities.

Policy variants in ``entity_transformer.py`` share this backbone and differ only
in their candidate-scoring head.

Leading axis ``S`` = sources (one row per owned source planet): ``MAX_SRC`` under
PPO's per-state ``vmap``, a flat source-row batch under BC. ``global_features`` is
the state's single global broadcast across ``S`` (one distinct global per state).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class _MLPBlock(nn.Module):
    d_model: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.ff_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, name="fc2")(x)
        return x


class _EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        # Pre-LN attention residual.
        h = nn.LayerNorm(name="ln1")(x)
        attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="attn",
        )(h, mask=mask)
        x = x + attn
        # Pre-LN FFN residual.
        h = nn.LayerNorm(name="ln2")(x)
        x = x + _MLPBlock(self.d_model, self.ff_dim, name="mlp")(h)
        return x


class EntityEncoder(nn.Module):
    """Encode one source row into per-token embeddings.

    Per source the token sequence is ``[self, global, cand_0 .. cand_{K-1}]``,
    each projected to ``d_model`` and tagged with a learned type embedding
    (0=self, 1=global, 2=candidate). A key padding mask hides invalid
    candidates; the self + global tokens are always valid keys.

    Returns ``(self_emb [S,H], global_emb [S,H], cand_emb [S,K,H])``.
    """

    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [S, self_dim]
        global_features: jax.Array,     # [S, global_dim]
        candidate_features: jax.Array,  # [S, K, cand_dim]
        candidate_mask: jax.Array,      # [S, K] bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        S, K = candidate_features.shape[0], candidate_features.shape[1]
        H = self.d_model

        type_embed = self.param(
            "type_embed", nn.initializers.normal(stddev=0.02), (3, H), jnp.float32
        )

        self_tok   = nn.Dense(H, name="self_proj")(self_features) + type_embed[0][None, :]
        global_tok = nn.Dense(H, name="global_proj")(global_features) + type_embed[1][None, :]
        cand_tok   = nn.Dense(H, name="cand_proj")(candidate_features) + type_embed[2][None, None, :]

        tokens = jnp.concatenate(
            [self_tok[:, None, :], global_tok[:, None, :], cand_tok], axis=1
        )  # [S, 2+K, H]

        always_valid = jnp.ones((S, 2), dtype=jnp.bool_)
        key_mask = jnp.concatenate([always_valid, candidate_mask], axis=1)   # [S, 2+K]
        attn_mask = jnp.broadcast_to(
            key_mask[:, None, None, :], (S, 1, 2 + K, 2 + K)
        )

        x = tokens
        for i in range(self.n_layers):
            x = _EncoderBlock(H, self.n_heads, self.ff_dim, name=f"block{i}")(x, attn_mask)
        x = nn.LayerNorm(name="ln_out")(x)

        self_emb   = x[:, 0, :]            # [S, H]
        global_emb = x[:, 1, :]            # [S, H]
        cand_emb   = x[:, 2:, :]           # [S, K, H]
        return self_emb, global_emb, cand_emb


class CrossSourceMixer(nn.Module):
    """Let each source attend to every valid sibling source (coalition signal).

    The mixer attends over the ``MAX_SRC`` (owned-planet) dimension, so a single
    block only gives pairwise source visibility. ``n_blocks`` stacks pre-LN
    attention+FFN blocks so sources can coordinate *transitively* (A↔B↔C) across
    the owned-planet set — coalition attacks / no double-sending over more than
    one hop.

    Back-compat: block 0 keeps the original *unindexed* parameter names
    (``cs_ln1``/``cs_attn``/``cs_ln2``/``cs_fc1``/``cs_fc2``), so an ``n_blocks=1``
    mixer is byte-identical to the pre-knob version and existing checkpoints load
    unchanged. Extra blocks get a ``_{i}`` suffix; warm-starting a 2-block mixer
    from a 1-block checkpoint restores block 0 and cold-inits the rest.
    """

    d_model: int
    n_heads: int
    n_blocks: int = 1

    @nn.compact
    def __call__(self, self_emb: jax.Array, src_valid: jax.Array) -> jax.Array:
        H = self.d_model
        # [1,1,1,S] broadcasts to [S, n_heads, S_q, S_k]; invalid sources masked
        # as keys so padding doesn't pollute the coalition signal.
        S = self_emb.shape[0]
        cs_mask = src_valid[None, None, None, :]
        for i in range(self.n_blocks):
            sfx = "" if i == 0 else f"_{i}"
            normed = nn.LayerNorm(name=f"cs_ln1{sfx}")(self_emb)
            # Attention can return broadcast leading dims from the 4-D mask;
            # collapse back to [S, H] (parity with CrossSourcePolicy's block).
            self_emb = (self_emb + nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=H, out_features=H,
                name=f"cs_attn{sfx}",
            )(normed, mask=cs_mask)).reshape(S, H)
            normed2 = nn.LayerNorm(name=f"cs_ln2{sfx}")(self_emb)
            self_emb = self_emb + nn.Dense(H, name=f"cs_fc2{sfx}")(
                nn.gelu(nn.Dense(4 * H, name=f"cs_fc1{sfx}")(normed2))
            )
        return self_emb


class MultiQueryValueHead(nn.Module):
    """Multi-query cross-attention value head (upstream heads.ValueHead)."""

    d_model: int
    n_queries: int
    n_heads: int
    hidden: int

    @nn.compact
    def __call__(
        self,
        self_emb: jax.Array,     # [S, H]
        global_emb: jax.Array,   # [S, H]
        cand_emb: jax.Array,     # [S, K, H]
        candidate_mask: jax.Array,  # [S, K] bool
    ) -> jax.Array:
        S, H = self_emb.shape
        queries = self.param(
            "queries", nn.initializers.normal(stddev=0.02),
            (self.n_queries, self.d_model), jnp.float32,
        )
        q = jnp.broadcast_to(queries[None, :, :], (S, self.n_queries, self.d_model))
        # Condition queries on global state so they aren't static slots.
        q = q + nn.Dense(self.d_model, name="q_cond")(global_emb)[:, None, :]

        kv = jnp.concatenate([self_emb[:, None, :], cand_emb], axis=1)        # [S, 1+K, H]
        kv_mask = jnp.concatenate(
            [jnp.ones((S, 1), dtype=jnp.bool_), candidate_mask], axis=1
        )
        attn_mask = kv_mask[:, None, None, :]
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.d_model,
            out_features=self.d_model, name="value_attn",
        )(q, kv, mask=attn_mask)

        pooled = attended.reshape(S, self.n_queries * self.d_model)
        joined = jnp.concatenate([pooled, global_emb], axis=-1)
        x = nn.gelu(nn.Dense(self.hidden, name="fc1")(joined))
        return nn.Dense(1, name="value")(x).squeeze(-1)                       # [S]
