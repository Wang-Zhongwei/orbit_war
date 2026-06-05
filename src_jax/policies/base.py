"""Shared policy primitives used by every architecture in this package."""
from __future__ import annotations

from typing import NamedTuple

import jax


# Finite floor used to mask invalid candidates in the target logits. Must be
# very negative (prob≈0) but finite so entropy/log_softmax stay NaN-free.
# (-inf would make softmax prob exactly 0, then exp(log_p)*log_p = 0*-inf = NaN,
# poisoning the PPO loss and every gradient after it.)
MASK_LOGIT = -1e9


class PolicyOutput(NamedTuple):
    target_logits: jax.Array   # [N, K]
    value: jax.Array           # [N]
