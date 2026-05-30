from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _setstate(instance: Any, state: Any) -> None:
    """Backward-compatible __setstate__ for slots dataclasses.

    Handles both old non-slots checkpoints (state is a plain dict) and new
    slots checkpoints (state is (dict_state, slot_state) tuple).  Fields
    missing from the stored state get their declared default values, so
    adding a new field never breaks loading an older checkpoint.
    """
    slot_state: dict = state[1] if isinstance(state, tuple) else (state or {})
    for f in fields(type(instance)):
        if f.name in slot_state:
            object.__setattr__(instance, f.name, slot_state[f.name])
        elif f.default is not MISSING:
            object.__setattr__(instance, f.name, f.default)
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            object.__setattr__(instance, f.name, f.default_factory())


@dataclass(slots=True)
class EnvConfig:
    episode_steps: int = 500
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_planets: int = 48
    max_ships: float = 400.0

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 128

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 64
    num_envs: int = -1
    total_updates: int = 1000
    epochs: int = 4
    minibatch_size: int = 2048
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    kl_coef: float = 0.0
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass(slots=True)
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

    def __setstate__(self, state) -> None: _setstate(self, state)


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "orbit_wars_jax_ppo"
    save_dir: str = "artifacts/jax"
    checkpoint_every: int = 50
    log_every: int = 1
    opponents: list[str] = field(default_factory=lambda: ["self"])
    eval_opponents: list[str] = field(default_factory=list)
    self_play_update_interval: int = 50
    self_eval_update_interval: int = 50
    eval_rounds: int = 10
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
        _setstate(self, slot_state)


def default_train_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default_cfg.yaml"


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
