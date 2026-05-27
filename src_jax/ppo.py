from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .policy import PolicyOutput, build_policy, init_params


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    """Thin subclass — makes isinstance checks easier."""


def build_train_state(cfg, seed: int = 0) -> TrainState:
    module = build_policy(cfg)
    params = init_params(module, seed)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(cfg.ppo.lr),
    )
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@dataclass(slots=True)
class TrainingBatch:
    self_features: jax.Array        # [N, self_dim]
    candidate_features: jax.Array   # [N, K, cand_dim]
    global_features: jax.Array      # [N, global_dim]
    candidate_mask: jax.Array       # [N, K]  bool
    target_index: jax.Array         # [N]     int32
    ships_bucket: jax.Array         # [N]     int32
    log_prob: jax.Array             # [N]     float32
    returns: jax.Array              # [N]     float32
    advantages: jax.Array           # [N]     float32


# ---------------------------------------------------------------------------
# Distribution helpers (pure JAX, JIT-compatible)
# ---------------------------------------------------------------------------

def _categorical_log_prob(logits: jax.Array, index: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, index[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


def _safe_target_logits(target_logits: jax.Array) -> jax.Array:
    invalid = ~jnp.isfinite(target_logits).any(axis=-1)
    safe = jnp.where(invalid[:, None], jnp.zeros_like(target_logits).at[:, 0].set(0.0), target_logits)
    return safe


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: jax.Array,
    ships_bucket: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    target_logits = _safe_target_logits(outputs.target_logits)
    N = target_index.shape[0]
    selected_ships = outputs.ships_logits[jnp.arange(N), target_index]
    log_prob = (
        _categorical_log_prob(target_logits, target_index)
        + _categorical_log_prob(selected_ships, ships_bucket)
    )
    entropy = (
        _categorical_entropy(target_logits)
        + _categorical_entropy(selected_ships)
    )
    return log_prob, entropy


def sample_actions(
    outputs: PolicyOutput,
    key: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample (target_index, ships_bucket) and return (ti, sb, log_prob, entropy, new_key)."""
    target_logits = _safe_target_logits(outputs.target_logits)
    N = outputs.value.shape[0]

    if deterministic:
        target_index = jnp.argmax(target_logits, axis=-1)
    else:
        key, k1 = jax.random.split(key)
        target_index = jax.random.categorical(k1, target_logits)

    selected_ships_logits = outputs.ships_logits[jnp.arange(N), target_index]

    if deterministic:
        ships_bucket = jnp.argmax(selected_ships_logits, axis=-1)
    else:
        key, k2 = jax.random.split(key)
        ships_bucket = jax.random.categorical(k2, selected_ships_logits)

    log_prob, entropy = action_log_prob_and_entropy(outputs, target_index, ships_bucket)
    return target_index, ships_bucket, log_prob, entropy, key


# ---------------------------------------------------------------------------
# JIT-compiled single-minibatch update step
# ---------------------------------------------------------------------------

@jax.jit
def _minibatch_update(
    state: TrainState,
    sf: jax.Array,
    cf: jax.Array,
    gf: jax.Array,
    cm: jax.Array,
    old_lp: jax.Array,
    ti: jax.Array,
    sb: jax.Array,
    ret: jax.Array,
    adv: jax.Array,
    clip_coef: jax.Array,
    ent_coef: jax.Array,
    vf_coef: jax.Array,
) -> tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def loss_fn(params):
        outputs = state.apply_fn(params, sf, cf, gf, cm)
        new_lp, ent = action_log_prob_and_entropy(outputs, ti, sb)
        ratio = jnp.exp(new_lp - old_lp)
        pg = jnp.maximum(
            -adv * ratio,
            -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        ).mean()
        vl  = 0.5 * jnp.mean((ret - outputs.value) ** 2)
        em  = ent.mean()
        return pg + vf_coef * vl - ent_coef * em, (pg, vl, em, ratio, new_lp)

    (loss, (pg, vl, em, ratio, new_lp)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, pg, vl, em, ratio, new_lp


# ---------------------------------------------------------------------------
# PPO update — Python loop over epochs × minibatches
# ---------------------------------------------------------------------------

def ppo_update(
    state: TrainState,
    batch: TrainingBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    epochs: int,
    minibatch_size: int,
    key: jax.Array,
) -> tuple[TrainState, dict[str, float], jax.Array]:
    if batch.self_features.shape[0] == 0:
        return state, {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "clip_fraction": 0.0, "kl_divergence": 0.0,
            "explained_variance": 0.0,
        }, key

    sf      = batch.self_features
    cf      = batch.candidate_features
    gf      = batch.global_features
    cm      = batch.candidate_mask
    old_lp  = batch.log_prob
    ti      = batch.target_index
    sb      = batch.ships_bucket
    ret     = batch.returns
    adv_raw = batch.advantages

    ev = float(1.0 - adv_raw.var() / (ret.var() + 1e-8))
    adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    size = sf.shape[0]
    mb   = min(size, max(1, minibatch_size))

    cc = jnp.float32(clip_coef)
    ec = jnp.float32(ent_coef)
    vc = jnp.float32(vf_coef)

    acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
           "entropy": 0.0, "clip_fraction": 0.0, "kl_divergence": 0.0}
    n_updates = 0

    for _ in range(epochs):
        key, subkey = jax.random.split(key)
        order = jax.random.permutation(subkey, size)
        for start in range(0, size, mb):
            idx = order[start : start + mb]
            state, loss, pg, vl, em, ratio, new_lp = _minibatch_update(
                state,
                sf[idx], cf[idx], gf[idx], cm[idx],
                old_lp[idx], ti[idx], sb[idx], ret[idx], adv[idx],
                cc, ec, vc,
            )
            acc["loss"]          += float(loss)
            acc["policy_loss"]   += float(pg)
            acc["value_loss"]    += float(vl)
            acc["entropy"]       += float(em)
            acc["clip_fraction"] += float((jnp.abs(ratio - 1.0) > clip_coef).mean())
            acc["kl_divergence"] += float((old_lp[idx] - new_lp).mean())
            n_updates += 1

    n = max(n_updates, 1)
    metrics = {k: v / n for k, v in acc.items()}
    metrics["explained_variance"] = ev
    return state, metrics, key
