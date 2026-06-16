"""Policy networks.

Architectures are selected by ``cfg.model.arch``:

  * ``"cross_source"`` (default) — :class:`~.cross_source.CrossSourcePolicy`,
    the per-source candidate-scoring net (cross-source attention + candidate
    transformer block).
  * ``"entity_mlp"``      — :class:`~.entity_transformer.EntityMLPPolicy`,
    type-tagged entity-transformer backbone + single-hidden-layer joint MLP head.
  * ``"entity_deep_mlp"`` — :class:`~.entity_transformer.EntityDeepMLPPolicy`,
    same backbone + deeper residual joint MLP head.

Every variant exposes the same call signature and returns ``PolicyOutput``, so
the rest of the codebase imports ``build_policy`` / ``init_params`` /
``PolicyOutput`` from this package and never special-cases the architecture.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import PolicyOutput, MASK_LOGIT
from .cross_source import CrossSourcePolicy, PlanetPolicy, build as _build_cross_source
from .entity_transformer import (
    EntityMLPPolicy,
    EntityDeepMLPPolicy,
    EntityPolicy,
    build_mlp as _build_entity_mlp,
    build_deep_mlp as _build_entity_deep_mlp,
)

__all__ = [
    "PolicyOutput",
    "MASK_LOGIT",
    "CrossSourcePolicy",
    "EntityMLPPolicy",
    "EntityDeepMLPPolicy",
    "PlanetPolicy",   # legacy alias
    "EntityPolicy",   # legacy alias
    "build_policy",
    "init_params",
    "ARCH_BUILDERS",
]


# arch key → builder. Legacy keys map to their renamed builder.
ARCH_BUILDERS = {
    "cross_source":    _build_cross_source,
    "entity_mlp":      _build_entity_mlp,
    "entity_deep_mlp": _build_entity_deep_mlp,
    "":                _build_cross_source,
}


def build_policy(cfg):
    """Construct the policy module selected by ``cfg.model.arch``.

    The arch builders bake in the default ("v7") feature dims. For another
    ``cfg.model.feat_version`` we swap the (metadata-only) dim fields so
    ``init_params`` sizes its dummy input correctly — the net itself is
    dim-agnostic (flax ``Dense`` infers the input width at apply time).
    """
    arch = getattr(cfg.model, "arch", "cross_source")
    try:
        module = ARCH_BUILDERS[arch](cfg)
    except KeyError:
        valid = sorted(k for k in ARCH_BUILDERS if k)
        raise ValueError(f"Unknown model.arch={arch!r}; expected one of {valid}") from None

    version = getattr(cfg.model, "feat_version", "v7")
    if version and version != "v7":
        from ..feature_versions import feature_dims
        sd, cd, gd = feature_dims(version)
        module = module.clone(self_dim=sd, candidate_dim=cd, global_dim=gd)
    return module


def init_params(module, seed: int = 0) -> dict:
    """Initialise params for any policy in this package.

    Generic: reads only ``self_dim`` / ``candidate_dim`` / ``global_dim`` /
    ``candidate_count`` (which every variant exposes) and feeds the shared
    5-argument call signature.
    """
    K = module.candidate_count
    dummy = (
        jnp.zeros((1, module.self_dim)),
        jnp.zeros((1, K, module.candidate_dim)),
        jnp.zeros((1, module.global_dim)),
        jnp.ones((1, K), dtype=bool),
        jnp.ones((1,), dtype=bool),
    )
    return module.init(jax.random.PRNGKey(seed), *dummy)
