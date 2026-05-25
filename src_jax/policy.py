from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .obs_schema import self_feature_dim, candidate_feature_dim, global_feature_dim


class PolicyOutput(NamedTuple):
    target_logits: jax.Array   # [N, K]
    ships_logits: jax.Array    # [N, K, B]
    value: jax.Array           # [N]


class PlanetPolicy(nn.Module):
    self_dim: int
    candidate_dim: int
    global_dim: int
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
    ) -> PolicyOutput:
        H = self.hidden_size
        K = self.candidate_count
        N = self_features.shape[0]

        self_hidden   = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(self_features))))
        global_hidden = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(global_features))))

        cand_flat      = candidate_features.reshape(N * K, self.candidate_dim)
        cand_hidden    = nn.relu(nn.Dense(H)(nn.relu(nn.Dense(H)(cand_flat)))).reshape(N, K, H)

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


def build_module(cfg) -> PlanetPolicy:
    return PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
    ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
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
