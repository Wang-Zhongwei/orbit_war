from __future__ import annotations

from dataclasses import dataclass

import jax

MIN_SOURCE_SHIPS: int = 13


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: jax.Array
    ship_counts: list[int]
    target_angles: list[float]
    source_ships: int


@dataclass(slots=True)
class TurnBatch:
    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    contexts: list[DecisionContext]
    state: object


def self_feature_dim() -> int:
    return 11


def candidate_feature_dim() -> int:
    return 10


def global_feature_dim() -> int:
    return 8
