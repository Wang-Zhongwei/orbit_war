from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


# One row per planet decision collected during a rollout.
# Flattened across all envs and all rollout steps — fed directly into ppo_update.
@dataclass(slots=True)
class TransitionBatch:
    self_features: torch.Tensor       # [N, self_dim]    — features of the acting planet
    candidate_features: torch.Tensor  # [N, K, cand_dim] — features of each candidate target
    global_features: torch.Tensor     # [N, global_dim]  — game-level features
    candidate_mask: torch.Tensor      # [N, K]           — valid targets (True = selectable)
    target_index: torch.Tensor        # [N]              — action taken (index into candidates)
    log_prob: torch.Tensor            # [N]              — log π_old(a|s), frozen at collection time
    returns: torch.Tensor             # [N]              — discounted return G_t
    advantages: torch.Tensor          # [N]              — A_t = G_t - V(s_t)


def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_index = (
        target_logits.argmax(dim=-1) if deterministic else target_dist.sample()
    )

    log_prob, entropy = action_log_prob_and_entropy(
        outputs=outputs, target_index=target_index
    )
    return SampledAction(target_index=target_index, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()
    return target_log_prob, target_entropy


def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    # Guard against rows where every logit is -inf (no valid candidates).
    # Replace those rows with a uniform distribution over slot 0 so downstream
    # log_prob / entropy calls don't produce NaN.
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,     # ε in PPO clip: ratio stays in [1-ε, 1+ε]
    ent_coef: float,      # weight on entropy bonus (encourages exploration)
    vf_coef: float,       # weight on value function loss
    max_grad_norm: float,
    epochs: int,          # how many passes over the same batch (PPO data reuse)
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch.self_features.shape[0] == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "explained_variance": 0.0}
    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)  # log π_old(a|s) — fixed reference point
    target_index = batch.target_index.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    explained_variance = float(
        1.0 - advantages.var(unbiased=False) / (returns.var(unbiased=False) + 1e-8)
    )
    # Normalize advantages across the batch for stable gradient magnitudes.
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    size = self_features.shape[0]
    minibatch_size = min(size, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "kl_divergence": 0.0, "explained_variance": explained_variance}
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, minibatch_size):
            idx = order[start : start + minibatch_size]
            # Forward pass with current (updated) weights — gives log π_new(a|s)
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs,
                target_index[idx],
            )
            # Importance sampling ratio r_t = π_new(a|s) / π_old(a|s).
            # Starts at 1 on the very first minibatch; drifts as weights update across epochs.
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            # PPO clipped surrogate objective:
            #   L = -E[ min( r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t ) ]
            # The minimum prevents the policy from taking too large a step in either direction.
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            # MSE between predicted value V(s) and actual discounted return G_t
            value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()
            entropy_mean = entropy.mean()
            # Combined loss: policy + value critic - entropy bonus
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            clip_fraction = ((ratio.detach() - 1.0).abs() > clip_coef).float().mean()
            kl = (old_log_prob[idx] - new_log_prob).detach().mean()
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            metrics["clip_fraction"] += float(clip_fraction.cpu())
            metrics["kl_divergence"] += float(kl.cpu())
            updates += 1
    n = max(updates, 1)
    return {key: (value if key == "explained_variance" else value / n) for key, value in metrics.items()}
