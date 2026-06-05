from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _setstate(instance: Any, state: Any) -> None:
    """Backward-compatible __setstate__ for these config dataclasses.

    Handles both checkpoint shapes: a plain dict (non-slots pickles) and the
    ``(dict_state, slot_state)`` tuple produced by older ``slots=True``
    checkpoints.  Fields missing from the stored state get their declared
    default values, so adding a new field never breaks loading an older
    checkpoint.

    These classes deliberately do NOT use ``slots=True``: a slots dataclass
    exposes its fields as member descriptors that can become read-only during
    unpickling (e.g. when a stale module copy is loaded), which raised
    ``AttributeError: '...' object attribute '...' is read-only``.  Plain
    ``__dict__``-backed instances are always writable, so unpickling can never
    hit that failure.
    """
    slot_state: dict = state[1] if isinstance(state, tuple) else (state or {})
    for f in fields(type(instance)):
        if f.name in slot_state:
            value = slot_state[f.name]
        elif f.default is not MISSING:
            value = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            value = f.default_factory()
        else:
            continue
        try:
            object.__setattr__(instance, f.name, value)
        except AttributeError:
            setattr(instance, f.name, value)


@dataclass
class EnvConfig:
    episode_steps: int = 500
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_planets: int = 48
    max_ships: float = 400.0
    # K-candidate selection strategy used by the feature encoder:
    #   "category" (default) — per-owner quotas (enemy/neutral/friendly) filled by
    #                          nearest-of-type, with nearest-of-any overflow backfill.
    #   "knn"                — simply the K-1 nearest planets of any owner.
    # Slot 0 is always the no-op in both; comets and the source are excluded.
    cand_select: str = "category"

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass
class ModelConfig:
    hidden_size: int = 128
    # Policy architecture (see src_jax/policies/__init__.py):
    #   "cross_source"    — per-source candidate-scoring net (default).
    #   "entity_mlp"      — entity-transformer backbone + joint MLP head.
    #   "entity_deep_mlp" — entity-transformer backbone + deep residual MLP head.
    # Legacy aliases: "planet" → cross_source, "entity" → entity_mlp.
    arch: str = "cross_source"
    # Entity-transformer hyperparameters (only used by the "entity_*" arches).
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 0          # 0 → default to 2 * hidden_size at build time
    value_queries: int = 4
    score_blocks: int = 2    # residual head depth (entity_deep_mlp only)

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    num_envs: int = -1
    total_updates: int = 1000
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95  # 1.0 = full Monte-Carlo return; <1 trades variance for bias
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    kl_coef: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.0  # AdamW decoupled weight decay; 0.0 = plain Adam
    max_grad_norm: float = 0.5
    augment: bool = False
    fleet_penalty:        float = 0.0    # per-fleet dispatch penalty (0 = disabled)
    capture_bonus:        float = 0.0    # per-planet capture bonus (0 = disabled)
    capture_time_weight:  float = 0.5    # extra bonus fraction awarded for early captures

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass
class BCConfig:
    epochs: int = 1
    batch_size: int = 2048
    lr: float = 3e-4
    winner_only: bool = False
    use_shard: bool = False
    stream_buffer: int = 200
    top_k_episodes: int | None = None
    out_dir: str = "artifacts/bc"
    run_name: str = "bc_pretrain"
    augment: bool = True
    encode_workers: int = 0  # CPU encoder processes; 0 = auto (cpu_count - 1)

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass
class TrainConfig:
    seed: int = 42
    run_name: str = "default"
    save_dir: str = "artifacts/ppo"
    checkpoint_every: int = 50
    log_every: int = 1
    resume: str | None = None       # checkpoint path to resume training from (CLI --resume overrides)
    ref_policy: str | None = None   # frozen KL reference checkpoint (CLI --ref_policy overrides)
    opponents: list[str] = field(default_factory=lambda: ["self"])
    eval_opponents: list[str] = field(default_factory=list)
    self_play_update_interval: int = 50
    eval_update_interval: int = 50
    eval_rounds: int = 10
    eval_workers: int = -1  # CPU processes for eval matches; -1 = all cores
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)

    def __setstate__(self, state) -> None:
        slot_state: dict = state[1] if isinstance(state, tuple) else (state or {})
        # Rename legacy field
        if "opponent" in slot_state and "opponents" not in slot_state:
            slot_state = dict(slot_state)
            slot_state["opponents"] = [slot_state.pop("opponent")]
        if "self_eval_update_interval" in slot_state and "eval_update_interval" not in slot_state:
            slot_state = dict(slot_state)
            slot_state["eval_update_interval"] = slot_state.pop("self_eval_update_interval")
        _setstate(self, slot_state)


def default_train_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "ppo_default.yaml"


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return _config_from_dict(data)


def _config_from_dict(data: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _apply(cfg, data, skip={"env", "model", "ppo", "bc", "opponents", "eval_opponents"})
    if "opponents" in data:
        raw = data["opponents"]
        cfg.opponents = [raw] if isinstance(raw, str) else list(raw)
    if "eval_opponents" in data:
        raw = data["eval_opponents"]
        cfg.eval_opponents = [raw] if isinstance(raw, str) else list(raw)
    _apply(cfg.env, data.get("env", {}))
    _apply(cfg.model, data.get("model", {}))
    _apply(cfg.ppo, data.get("ppo", {}))
    _apply(cfg.bc, data.get("bc", {}))
    return cfg


def _apply(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce(value, default))


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
