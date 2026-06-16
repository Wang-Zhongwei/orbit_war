"""Feature-version registry — selects which encoder / feature dims a policy uses.

Main carries two in-tree feature encoders:

  "v7" (default) — ``src_jax/features.py``     self=25, cand=30, global=12
                   the current main arch (richer horizons / capture-margin feats).
  "v9"           — ``src_jax/features_v9.py``  self=17, cand=26, global=12
                   the leaner arch vendored from the (now-removed) archv9 worktree.

A PPO run picks the learner's encoder via ``cfg.model.feat_version``. The opponent
and the KL reference may use a *different* version (whatever their checkpoint was
trained with): each side encodes the shared, byte-identical ``EnvState`` with its
own encoder and emits a distribution over the SAME action space — K candidates and
MAX_SRC sources are selected identically across versions (``_select_sources_and_
candidates`` and its constants are shared), so slot-wise comparison (KL) and
cross-version opponent play are well-defined.

The network itself is dim-agnostic (flax ``Dense`` infers input width), so the
only version-dependent wiring is (a) which encoder produces the features and
(b) the metadata dims ``init_params`` uses to size its dummy input.
"""
from __future__ import annotations

import importlib

# version key → submodule name within this package
_MODULES = {"v7": "features", "v9": "features_v9"}

DEFAULT_VERSION = "v7"


def _module(version: str | None):
    v = version or DEFAULT_VERSION
    if v not in _MODULES:
        raise ValueError(
            f"unknown feat_version {version!r}; expected one of {sorted(_MODULES)}"
        )
    return importlib.import_module(f".{_MODULES[v]}", package=__package__)


def feature_dims(version: str | None = DEFAULT_VERSION) -> tuple[int, int, int]:
    """(self_dim, candidate_dim, global_dim) for ``version``."""
    m = _module(version)
    return m.self_feature_dim(), m.candidate_feature_dim(), m.global_feature_dim()


def make_env_encoder(K: int, env_cfg, version: str | None = DEFAULT_VERSION):
    """Versioned encoder factory: ``(batched_state, players[N]) -> _EncodedTurn``."""
    return _module(version)._make_env_encoder(K, env_cfg)
