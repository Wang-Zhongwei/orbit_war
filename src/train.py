from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import multiprocessing
import numpy as np
import torch
import wandb

from .config import TrainConfig, default_train_config_path, load_train_config
from .agents.rl.ppo_agent import build_move, build_policy
from .features import (
    TurnBatch,
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from .game_types import PlanetState
from .policy import PlanetPolicy
from .ppo import TransitionBatch, ppo_update, sample_actions
from .vec_env import SubprocVecEnv


# Groups all planet decisions that happened within one game tick for one env.
# After env.step(), we know the reward for that tick and whether the episode ended.
# indices: row positions in the flat training buffer (self_rows, values, etc.)
@dataclass(slots=True)
class StepGroup:
    indices: list[int]
    reward: float
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT",
                        help="Path to a checkpoint to resume training from")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rollout(
    vec_env: SubprocVecEnv,
    # batches: current observation for each env (one TurnBatch per env, updated each tick).
    # Returned at the end so the next call can resume from where each env left off.
    batches: list[TurnBatch],
    policy: PlanetPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
) -> tuple[TransitionBatch, list[TurnBatch], int, dict[str, float]]:
    empty_candidate = (cfg.env.candidate_count, candidate_feature_dim())
    # Flat training buffer — one entry per planet decision across all envs and all ticks.
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []        # V(s_t) for each row — used to compute A_t = G_t - V(s_t)
    # Per-env list of StepGroups; used later to assign returns/advantages to each row.
    groups_per_env: list[list[StepGroup]] = [[] for _ in range(len(vec_env))]
    episode_rewards: list[float] = []
    running_episode_rewards = [0.0 for _ in range(len(vec_env))]

    for _ in range(cfg.ppo.rollout_steps):
        # Merge all per-env TurnBatches into one tensor for a single batched forward pass.
        # offsets[i] = start row in the merged tensor for env i.
        offsets = np.cumsum(
            [0] + [batch.self_features.shape[0] for batch in batches[:-1]]
        )
        merged = merge_batches(batches)
        if merged.self_features.shape[0] > 0:
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(merged.self_features).to(device),
                    torch.from_numpy(merged.candidate_features).to(device),
                    torch.from_numpy(merged.global_features).to(device),
                    torch.from_numpy(merged.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs, deterministic=False)
                row_values = outputs.value.detach().cpu().numpy()
                sampled_target_index = sampled.target_index.detach().cpu().numpy() # detach from computation graph to make it fixed in PPO update
                sampled_log_prob = sampled.log_prob.detach().cpu().numpy()
        else:
            row_values = np.zeros((0,), dtype=np.float32)
            sampled_target_index = np.zeros((0,), dtype=np.int64)
            sampled_log_prob = np.zeros((0,), dtype=np.float32)

        # Phase 1: fill training buffer and compute moves for all envs.
        moves_per_env: list[list] = []
        group_indices_per_env: list[list[int]] = []
        for env_idx in range(len(vec_env)):
            batch = batches[env_idx]
            start = int(offsets[env_idx])
            moves = []
            group_indices: list[int] = []  # row indices of every planet decision this tick
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                # Save observation and decision for this planet into the training buffer.
                self_rows.append(batch.self_features[local_idx])
                candidate_rows.append(batch.candidate_features[local_idx])
                global_rows.append(batch.global_features[local_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                tgt_idx = (
                    int(sampled_target_index[global_idx])
                    if batch.self_features.shape[0] > 0
                    else 0
                )
                target_indices.append(tgt_idx)
                log_probs.append(
                    float(sampled_log_prob[global_idx])
                    if batch.self_features.shape[0] > 0
                    else 0.0
                )
                group_indices.append(len(values) - 1)
                move = build_move(context, tgt_idx)
                if move is None:
                    continue
                # todo: merge into policy or replace with hard logic (defense, offense)
                src_planet = find_planet(batch.state.planets, context.source_id)
                if src_planet is None or src_planet.ships < move[2]:
                    continue
                moves.append(move)
            moves_per_env.append(moves)
            group_indices_per_env.append(group_indices)

        # Phase 2: dispatch all env steps in parallel, then collect.
        vec_env.step_async(moves_per_env)
        results = vec_env.step_wait()

        # Phase 3: process results; reset done envs (rare, so serial is fine).
        next_batches: list[TurnBatch] = []
        for env_idx, result in enumerate(results):
            running_episode_rewards[env_idx] += float(result.reward)
            # Link this tick's row indices to the reward received so we can compute returns later.
            groups_per_env[env_idx].append(
                StepGroup(
                    indices=group_indices_per_env[env_idx],
                    reward=float(result.reward),
                    done=result.done,
                )
            )
            if result.done:
                episode_rewards.append(running_episode_rewards[env_idx])
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                next_batch = vec_env.reset_one(env_idx, seed=next_seed)
            else:
                next_batch = result.batch
            next_batches.append(next_batch)
        batches = next_batches

    # --- Compute discounted returns and advantages ---
    # Bootstrap V(s_T) at the final state so truncated episodes don't treat the
    # last step as terminal (avoids underestimating the long-run value).
    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    next_state_values = bootstrap_values(policy, batches, device)
    for env_idx, groups in enumerate(groups_per_env):
        # Walk backwards, accumulating the discounted return:
        #   G_t = r_t + γ * G_{t+1}   (γ discounts future rewards)
        # done=1 zeros the future term so episode boundaries don't bleed across games.
        future_return = next_state_values[env_idx]
        for group in reversed(groups):
            future_return = group.reward + cfg.ppo.gamma * future_return * (
                1.0 - float(group.done)
            )
            for idx in group.indices:
                returns[idx] = future_return
                # A_t = G_t - V(s_t): how much better this outcome was than predicted.
                advantages[idx] = future_return - values[idx]
    batch = TransitionBatch(
        self_features=torch.from_numpy(
            np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())
        ),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(
                -1, empty_candidate[0], empty_candidate[1]
            )
        ),
        global_features=torch.from_numpy(
            np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())
        ),
        candidate_mask=torch.from_numpy(
            np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)
        ),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards))
        if episode_rewards
        else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(len(values)),
    }
    return batch, batches, next_seed, stats


def bootstrap_values(
    policy: PlanetPolicy, batches: list[TurnBatch], device: torch.device
) -> list[float]:
    # Estimate V(s_T) at the end of the rollout for each env.
    # Averaged across all planets in the final state — used as the starting point
    # for the backwards return computation in collect_rollout.
    merged = merge_batches(batches)
    if merged.self_features.shape[0] == 0:
        return [0.0 for _ in batches]
    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(merged.self_features).to(device),
            torch.from_numpy(merged.candidate_features).to(device),
            torch.from_numpy(merged.global_features).to(device),
            torch.from_numpy(merged.candidate_mask).to(device).bool(),
        )
    values = outputs.value.detach().cpu().numpy()
    per_env = []
    for env_idx, batch in enumerate(batches):
        start = int(offsets[env_idx])
        count = batch.self_features.shape[0]
        per_env.append(
            0.0 if count == 0 else float(values[start : start + count].mean())
        )
    return per_env


def merge_batches(batches: list[TurnBatch]) -> TurnBatch:
    if not batches:
        raise ValueError("batches must not be empty")
    has_rows = any(batch.self_features.shape[0] > 0 for batch in batches)
    self_rows = (
        np.concatenate([batch.self_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, self_feature_dim()), dtype=np.float32)
    )
    candidate_rows = (
        np.concatenate([batch.candidate_features for batch in batches], axis=0)
        if has_rows
        else np.zeros(
            (0, batches[0].candidate_features.shape[1], candidate_feature_dim()),
            dtype=np.float32,
        )
    )
    global_rows = (
        np.concatenate([batch.global_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, global_feature_dim()), dtype=np.float32)
    )
    candidate_masks = (
        np.concatenate([batch.candidate_mask for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_mask.shape[1]), dtype=bool)
    )
    return TurnBatch(
        self_features=self_rows,
        candidate_features=candidate_rows,
        global_features=global_rows,
        candidate_mask=candidate_masks,
        contexts=[context for batch in batches for context in batch.contexts],
        state=batches[0].state,
    )


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    policy: PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        run_dir / "ckpt_last.pt",
    )
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        run_dir / f"ckpt_{update:06d}.pt",
    )


def find_planet(planets: list[PlanetState], planet_id: int) -> PlanetState | None:
    for planet in planets:
        if planet.id == planet_id:
            return planet
    return None


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    num_envs = cfg.ppo.num_envs if cfg.ppo.num_envs > 0 else multiprocessing.cpu_count()
    print(f"Using {num_envs} parallel environments")
    import dataclasses
    wandb.init(
        project="orbit-war",
        name=cfg.run_name,
        config={
            "seed": cfg.seed,
            "num_envs": num_envs,
            "opponents": cfg.opponents,
            "self_play_update_interval": cfg.self_play_update_interval,
            **{f"env/{k}": v for k, v in dataclasses.asdict(cfg.env).items()},
            **{f"model/{k}": v for k, v in dataclasses.asdict(cfg.model).items()},
            **{f"ppo/{k}": v for k, v in dataclasses.asdict(cfg.ppo).items()},
        },
    )
    vec_env = SubprocVecEnv(num_envs, cfg, cfg.opponents, worker_device="cpu")
    next_seed = cfg.seed
    seeds = list(range(next_seed, next_seed + num_envs))
    next_seed += num_envs
    batches = vec_env.reset_all(seeds)

    policy = build_policy(cfg, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    start_update = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"] + 1
        print(f"Resumed from {args.resume} (update {ckpt['update']})")

    if "self" in cfg.opponents:
        vec_env.sync_opponent(policy.state_dict())

    save_dir = Path(cfg.save_dir)

    try:
        for update in range(start_update, cfg.ppo.total_updates + 1):
            batch, batches, next_seed, stats = collect_rollout(
                vec_env, batches, policy, cfg, device, next_seed
            )
            metrics = ppo_update(
                policy,
                optimizer,
                batch,
                clip_coef=cfg.ppo.clip_coef,
                ent_coef=cfg.ppo.ent_coef,
                vf_coef=cfg.ppo.vf_coef,
                max_grad_norm=cfg.ppo.max_grad_norm,
                epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                device=device,
            )
            if "self" in cfg.opponents and update % cfg.self_play_update_interval == 0:
                vec_env.sync_opponent(policy.state_dict())
            if update % cfg.log_every == 0:
                print(
                    f"update={update} episode_reward_mean={stats['episode_reward_mean']:.4f} "
                    f"episodes={int(stats['episodes_finished'])} samples={int(stats['samples'])} "
                    f"loss={metrics['loss']:.4f} ev={metrics['explained_variance']:.3f} "
                    f"entropy={metrics['entropy']:.3f} clip={metrics['clip_fraction']:.3f}"
                )
                wandb.log(
                    {
                        "reward/episode_mean": stats["episode_reward_mean"],
                        "reward/episodes_finished": stats["episodes_finished"],
                        "train/samples": stats["samples"],
                        "train/loss": metrics["loss"],
                        "train/policy_loss": metrics["policy_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/entropy": metrics["entropy"],
                        "train/clip_fraction": metrics["clip_fraction"],
                        "train/kl_divergence": metrics["kl_divergence"],
                        "train/explained_variance": metrics["explained_variance"],
                    },
                    step=update,
                )
            if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
                save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
