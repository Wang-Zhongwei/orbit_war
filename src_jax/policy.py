from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .features import self_feature_dim, candidate_feature_dim, global_feature_dim, fleet_feature_dim


class PolicyOutput(NamedTuple):
    target_logits: jax.Array   # [N, K]
    ships_logits: jax.Array    # [N, K, B]
    value: jax.Array           # [N]


class PlanetPolicy(nn.Module):
    self_dim: int
    candidate_dim: int
    global_dim: int
    fleet_dim: int
    candidate_count: int
    ship_bucket_count: int = 8
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,       # [N, self_dim]
        candidate_features: jax.Array,  # [N, K, candidate_dim]
        global_features: jax.Array,     # [N, global_dim]
        candidate_mask: jax.Array,      # [N, K] bool
        fleet_features: jax.Array,      # [N, F, fleet_dim]
        fleet_mask: jax.Array,          # [N, F] bool
    ) -> PolicyOutput:
        H = self.hidden_size
        K = self.candidate_count
        N = self_features.shape[0]

        self_hidden   = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(self_features))))
        global_hidden = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(global_features))))

        cand_flat      = candidate_features.reshape(N * K, self.candidate_dim)
        cand_hidden    = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

        # Cross-attention: each candidate queries the fleet set.
        # Geometric "is this fleet aimed at me?" learned via q·k.
        fleet_hidden = nn.relu(nn.Dense(H)(fleet_features))                              # [N, F, H]
        q = nn.Dense(H)(cand_hidden)                                                     # [N, K, H]
        k = nn.Dense(H)(fleet_hidden)                                                    # [N, F, H]
        v = nn.Dense(H)(fleet_hidden)                                                    # [N, F, H]
        attn_logits = jnp.einsum('nkh,nfh->nkf', q, k) / jnp.sqrt(jnp.float32(H))
        shifted     = attn_logits - jnp.max(attn_logits, axis=-1, keepdims=True)
        exp_logits  = jnp.exp(shifted) * fleet_mask[:, None, :].astype(jnp.float32)
        attn        = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-9)       # safe when 0 alive
        fleet_ctx   = jnp.einsum('nkf,nfh->nkh', attn, v)                                # [N, K, H]
        cand_hidden = cand_hidden + fleet_ctx                                            # residual

        expanded_self   = jnp.broadcast_to(self_hidden[:, None, :],   (N, K, H))
        expanded_global = jnp.broadcast_to(global_hidden[:, None, :], (N, K, H))
        joint = jnp.concatenate([expanded_self, expanded_global, cand_hidden], axis=-1)

        target_logits = nn.Dense(1)(nn.relu(nn.Dense(H)(joint))).squeeze(-1)
        target_logits = jnp.where(candidate_mask, target_logits, jnp.finfo(jnp.float32).min)

        ships_logits = nn.Dense(self.ship_bucket_count)(nn.relu(nn.Dense(H)(joint)))

        pooled   = cand_hidden.mean(axis=1)
        value_in = jnp.concatenate([self_hidden, global_hidden, pooled], axis=-1)
        value    = nn.Dense(1)(nn.relu(nn.Dense(H)(value_in))).squeeze(-1)

        return PolicyOutput(target_logits=target_logits, ships_logits=ships_logits, value=value)


def build_policy(cfg) -> PlanetPolicy:
    return PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        fleet_dim=fleet_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
    )


def init_params(module: PlanetPolicy, seed: int = 0, fleet_count: int = 1) -> dict:
    K = module.candidate_count
    dummy = (
        jnp.zeros((1, module.self_dim)),
        jnp.zeros((1, K, module.candidate_dim)),
        jnp.zeros((1, module.global_dim)),
        jnp.ones((1, K), dtype=bool),
        jnp.zeros((1, fleet_count, module.fleet_dim)),
        jnp.ones((1, fleet_count), dtype=bool),
    )
    return module.init(jax.random.PRNGKey(seed), *dummy)
