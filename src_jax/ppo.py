from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .policies import PolicyOutput, build_policy, init_params


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    """Thin subclass — makes isinstance checks easier."""


def build_train_state(cfg, seed: int = 0) -> TrainState:
    module = build_policy(cfg)
    params = init_params(module, seed)
    wd = cfg.ppo.weight_decay
    opt = optax.adamw(cfg.ppo.lr, weight_decay=wd) if wd > 0.0 else optax.adam(cfg.ppo.lr)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        opt,
    )
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@dataclass(slots=True)
class TrainingBatch:
    self_features: jax.Array        # [B, MAX_SRC, self_dim]
    candidate_features: jax.Array   # [B, MAX_SRC, K, cand_dim]
    global_features: jax.Array      # [B, global_dim]
    candidate_mask: jax.Array       # [B, MAX_SRC, K]  bool
    target_index: jax.Array         # [B, MAX_SRC]     int32
    log_prob: jax.Array             # [B, MAX_SRC]     float32
    returns: jax.Array              # [B]              float32
    advantages: jax.Array           # [B, MAX_SRC]     float32
    src_valid: jax.Array            # [B, MAX_SRC]     bool


_TB_FIELDS = [
    'self_features', 'candidate_features', 'global_features', 'candidate_mask',
    'target_index', 'log_prob', 'returns', 'advantages', 'src_valid',
]
jax.tree_util.register_pytree_node(
    TrainingBatch,
    lambda b: ([getattr(b, f) for f in _TB_FIELDS], None),
    lambda _, xs: TrainingBatch(*xs),
)


# ---------------------------------------------------------------------------
# Distribution helpers (pure JAX, JIT-compatible)
# ---------------------------------------------------------------------------

def _categorical_log_prob(logits: jax.Array, index: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, index[..., None], axis=-1).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


def _categorical_kl(ref_logits: jax.Array, new_logits: jax.Array) -> jax.Array:
    """KL(p_ref || p_new) per row; last axis is the category dimension."""
    log_p_ref = jax.nn.log_softmax(ref_logits, axis=-1)
    log_p_new = jax.nn.log_softmax(new_logits, axis=-1)
    return jnp.sum(jnp.exp(log_p_ref) * (log_p_ref - log_p_new), axis=-1)


def _safe_target_logits(target_logits: jax.Array) -> jax.Array:
    invalid = ~jnp.isfinite(target_logits).any(axis=-1)
    safe = jnp.where(invalid[:, None], jnp.zeros_like(target_logits).at[:, 0].set(0.0), target_logits)
    return safe


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    target_logits = _safe_target_logits(outputs.target_logits)
    log_prob = _categorical_log_prob(target_logits, target_index)
    entropy  = _categorical_entropy(target_logits)
    return log_prob, entropy


def sample_actions(
    outputs: PolicyOutput,
    key: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample target_index and return (ti, log_prob, entropy, new_key)."""
    target_logits = _safe_target_logits(outputs.target_logits)

    if deterministic:
        target_index = jnp.argmax(target_logits, axis=-1)
    else:
        key, k1 = jax.random.split(key)
        target_index = jax.random.categorical(k1, target_logits)

    log_prob, entropy = action_log_prob_and_entropy(outputs, target_index)
    return target_index, log_prob, entropy, key


# ---------------------------------------------------------------------------
# PPO update — jitted, lax.scan over epochs × minibatches (no Python loop overhead)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('epochs', 'minibatch_size', 'use_kl'))
def _ppo_update_jit(
    state: TrainState,
    batch: TrainingBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    epochs: int,
    minibatch_size: int,
    key: jax.Array,
    ref_params: dict,
    kl_coef: float,
    use_kl: bool,
) -> tuple[TrainState, jax.Array, jax.Array]:
    sf      = batch.self_features        # [B, MAX_SRC, self_dim]
    cf      = batch.candidate_features   # [B, MAX_SRC, K, cand_dim]
    gf      = batch.global_features      # [B, glob_dim]
    cm      = batch.candidate_mask       # [B, MAX_SRC, K]
    sv      = batch.src_valid            # [B, MAX_SRC] bool
    old_lp  = batch.log_prob             # [B, MAX_SRC]
    ti      = batch.target_index         # [B, MAX_SRC]
    ret     = batch.returns              # [B]
    adv_raw = batch.advantages           # [B, MAX_SRC]

    valid   = sv.astype(jnp.float32)
    n_valid = valid.sum() + 1e-8

    adv_mean = (adv_raw * valid).sum() / n_valid
    adv_var  = ((adv_raw - adv_mean) ** 2 * valid).sum() / n_valid
    adv_std  = jnp.sqrt(adv_var + 1e-8)
    adv      = (adv_raw - adv_mean) / adv_std
    ev_jax   = 1.0 - adv_var / (ret.var() + 1e-8)

    B    = sf.shape[0]
    mb   = min(B, max(1, minibatch_size))
    n_mb = B // mb  # drop any remainder

    cc = jnp.float32(clip_coef)
    ec = jnp.float32(ent_coef)
    vc = jnp.float32(vf_coef)
    kc = jnp.float32(kl_coef)

    MAX_SRC  = sf.shape[1]
    GLOB_DIM = gf.shape[-1]
    K        = cm.shape[-1]

    # Precompute ref policy logits for the full batch once (outside any loop).
    # Avoids epochs×n_mb redundant forward passes when KL regularisation is active.
    if use_kl:
        gf_exp_full  = jnp.broadcast_to(gf[:, None, :], (B, MAX_SRC, GLOB_DIM))
        ref_out_full = jax.vmap(
            lambda sf_i, cf_i, gf_i, cm_i, sv_i: state.apply_fn(
                ref_params, sf_i, cf_i, gf_i, cm_i, sv_i
            )
        )(sf, cf, gf_exp_full, cm, sv)
        ref_tgt_all = _safe_target_logits(
            ref_out_full.target_logits.reshape(B * MAX_SRC, K)
        ).reshape(B, MAX_SRC, K)
    else:
        ref_tgt_all = None

    # Pre-split one key per epoch so the scan carries only the TrainState.
    key, *epoch_keys = jax.random.split(key, epochs + 1)
    epoch_keys = jnp.stack(epoch_keys)   # [epochs, 2]

    # ── Inner scan body: one minibatch gradient step ──────────────────────────
    def mb_step(state: TrainState, idx: jax.Array) -> tuple[TrainState, jax.Array]:
        """idx: [mb] int32 — row indices into the full batch."""
        sv_mb = sv[idx]

        def _fwd(params):
            gf_exp = jnp.broadcast_to(gf[idx][:, None, :], (mb, MAX_SRC, GLOB_DIM))
            return jax.vmap(
                lambda sf_i, cf_i, gf_i, cm_i, sv_i: state.apply_fn(
                    params, sf_i, cf_i, gf_i, cm_i, sv_i
                )
            )(sf[idx], cf[idx], gf_exp, cm[idx], sv_mb)

        def loss_fn(params):
            outputs  = _fwd(params)
            flat_out = PolicyOutput(
                target_logits = outputs.target_logits.reshape(mb * MAX_SRC, K),
                value         = outputs.value.reshape(-1),
            )
            new_lp_f, ent_f = action_log_prob_and_entropy(flat_out, ti[idx].reshape(-1))
            new_lp  = new_lp_f.reshape(mb, MAX_SRC)
            ent     = ent_f.reshape(mb, MAX_SRC)
            valid_m = sv_mb.astype(jnp.float32)
            n_v     = valid_m.sum() + 1e-8

            ratio = jnp.exp(new_lp - old_lp[idx])
            pg = (jnp.maximum(
                -adv[idx] * ratio,
                -adv[idx] * jnp.clip(ratio, 1.0 - cc, 1.0 + cc),
            ) * valid_m).sum() / n_v
            vl = 0.5 * ((ret[idx][:, None] - outputs.value) ** 2 * valid_m).sum() / n_v
            em = (ent * valid_m).sum() / n_v

            if use_kl:
                ref_tgt = ref_tgt_all[idx].reshape(mb * MAX_SRC, K)
                new_tgt = _safe_target_logits(flat_out.target_logits)
                kl      = (_categorical_kl(ref_tgt, new_tgt).reshape(mb, MAX_SRC) * valid_m).sum() / n_v
            else:
                kl = jnp.float32(0.0)

            loss = pg + vc * vl - ec * em + kc * kl
            return loss, (pg, vl, em, kl, ratio, new_lp)

        (loss, (pg, vl, em, kl, ratio, new_lp)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        state = state.apply_gradients(grads=grads)

        valid_m = sv_mb.astype(jnp.float32)
        n_vm    = valid_m.sum() + 1e-8
        cf_mb   = ((jnp.abs(ratio - 1.0) > cc) * valid_m).sum() / n_vm
        kd_mb   = ((old_lp[idx] - new_lp) * valid_m).sum() / n_vm
        return state, jnp.stack([loss, pg, vl, em, kl, cf_mb, kd_mb])

    # ── Outer scan body: one epoch ────────────────────────────────────────────
    def epoch_step(state: TrainState, epoch_key: jax.Array) -> tuple[TrainState, jax.Array]:
        perm = jax.random.permutation(epoch_key, B)[: n_mb * mb].reshape(n_mb, mb)
        state, mb_metrics = jax.lax.scan(mb_step, state, perm)
        return state, mb_metrics   # mb_metrics: [n_mb, 7]

    state, all_metrics = jax.lax.scan(epoch_step, state, epoch_keys)
    # all_metrics: [epochs, n_mb, 7]
    m = all_metrics.mean(axis=(0, 1))   # [7]

    # Pack ev_jax as 8th element so the caller can sync everything in one np.asarray call.
    return state, jnp.append(m, ev_jax), key


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
    ref_params: dict | None = None,
    kl_coef: float = 0.0,
) -> tuple[TrainState, dict[str, float], jax.Array]:
    if batch.self_features.shape[0] == 0:
        return state, {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "clip_fraction": 0.0, "kl_divergence": 0.0,
            "kl_ref": 0.0, "explained_variance": 0.0,
        }, key

    use_kl      = ref_params is not None and kl_coef > 0.0
    _ref_params = ref_params if use_kl else state.params

    state, m, key = _ppo_update_jit(
        state, batch,
        clip_coef=clip_coef, ent_coef=ent_coef, vf_coef=vf_coef,
        epochs=epochs, minibatch_size=minibatch_size,
        key=key, ref_params=_ref_params, kl_coef=kl_coef, use_kl=use_kl,
    )

    # Single device→host transfer for all 8 metric values.
    m_np = np.asarray(m)
    return state, {
        "loss":               float(m_np[0]),
        "policy_loss":        float(m_np[1]),
        "value_loss":         float(m_np[2]),
        "entropy":            float(m_np[3]),
        "kl_ref":             float(m_np[4]),
        "clip_fraction":      float(m_np[5]),
        "kl_divergence":      float(m_np[6]),
        "explained_variance": float(m_np[7]),
    }, key
