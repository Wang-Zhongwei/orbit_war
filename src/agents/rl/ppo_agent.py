from __future__ import annotations

import importlib
import sys
import types
from typing import Callable

import torch

from src.features import DecisionContext
from src.policy import PlanetPolicy
from src.ppo import sample_actions


def _register_checkpoint_module_aliases() -> None:
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.rl_template", types.ModuleType("src.rl_template"))
    candidates_map = {
        "config":     ["src.rl_template.config",     "src.config",     "config"],
        "features":   ["src.rl_template.features",   "src.features",   "features"],
        "policy":     ["src.rl_template.policy",     "src.policy",     "policy"],
        "ppo":        ["src.rl_template.ppo",        "src.ppo",        "ppo"],
        "game_types": ["src.rl_template.game_types", "src.game_types", "game_types"],
        "opponents":  ["src.rl_template.opponents",  "src.opponents",  "opponents"],
        "env":        ["src.rl_template.env",        "src.env",        "env"],
        "train":      ["src.rl_template.train",      "src.train",      "train"],
    }
    for canonical, paths in candidates_map.items():
        module = None
        for path in paths:
            try:
                module = importlib.import_module(path)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            continue
        sys.modules[f"src.rl_template.{canonical}"] = module
        sys.modules[f"src.{canonical}"] = module


def build_move(context: DecisionContext, tgt_idx: int) -> list | None:
    if tgt_idx == 0 or tgt_idx >= len(context.candidate_ids):
        return None
    if not context.candidate_mask[tgt_idx]:
        return None
    ships = int(context.ship_counts[tgt_idx])
    if ships <= 0:
        return None
    return [context.source_id, float(context.target_angles[tgt_idx]), ships]


def build_policy(cfg, device: torch.device) -> PlanetPolicy:
    from src.features import self_feature_dim, candidate_feature_dim, global_feature_dim
    return PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size,
    ).to(device)


def load_checkpoint(policy: PlanetPolicy, checkpoint_path: str | None, device: torch.device) -> None:
    _register_checkpoint_module_aliases()
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict)


def build_moves(batch, policy: PlanetPolicy, device: torch.device, deterministic: bool) -> list:
    if batch.self_features.shape[0] == 0:
        return []
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(batch.self_features).to(device),
            torch.from_numpy(batch.candidate_features).to(device),
            torch.from_numpy(batch.global_features).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
        )
        sampled = sample_actions(outputs, deterministic=deterministic)
    target_indices = sampled.target_index.detach().cpu().numpy()

    moves: list = []
    for row_idx, context in enumerate(batch.contexts):
        move = build_move(context, int(target_indices[row_idx]))
        if move is not None:
            moves.append(move)
    return moves


def make_ppo_agent(cfg, checkpoint_path: str | None, device: torch.device, deterministic: bool = False) -> Callable:
    """Return an obs→moves callable backed by a loaded policy."""
    from src.features import encode_turn
    policy = build_policy(cfg, device)
    load_checkpoint(policy, checkpoint_path, device)
    policy.eval()

    def agent(obs) -> list:
        batch = encode_turn(obs, cfg.env, env_index=0)
        return build_moves(batch, policy, device, deterministic)

    return agent
