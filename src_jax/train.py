from __future__ import annotations

import argparse
import dataclasses
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import jax
import jax.numpy as jnp

from .features import _make_env_encoder
from .feature_versions import make_env_encoder
from .config import (
    TrainConfig, OpponentSpec, default_train_config_path, load_train_config,
    parse_opponents,
)
from .env import make_board_pool, make_initial_carry_state
from .policies import build_policy
from .ppo import TrainState, build_train_state, ppo_update
from .rollout import (
    RolloutCarry,
    build_training_batch,
    compute_bootstrap_val,
    compute_returns,
    episode_stats,
    make_rollout_fn,
)

_EVAL_BASE_SEED = 1000

def run_eval(params, cfg, base_seed: int = _EVAL_BASE_SEED) -> dict[str, float]:
    # All-JAX opponents (self / ckpt: / sub_vN networks) play on-device in one
    # batched, jitted scan (eval_jax) — orders of magnitude faster than looping
    # kaggle env.run per match. Rule-based opponents (sub_v52, sniper, …) have no
    # JAX policy, so they stay on the CPU worker-pool path (eval_mp), which keeps
    # the accelerator free for the trainer between rollouts.
    from .eval_jax import is_jax_arch_opponent

    results: dict[str, float] = {}

    if getattr(cfg, "eval_jax_fastpath", True):
        jax_opps = [o for o in cfg.eval_opponents if is_jax_arch_opponent(o)]
        cpu_opps = [o for o in cfg.eval_opponents if o not in jax_opps]
    else:
        jax_opps, cpu_opps = [], list(cfg.eval_opponents)

    if jax_opps:
        from .eval_jax import run_eval_jax

        try:
            results.update(run_eval_jax(params, cfg, jax_opps, base_seed))
        except Exception as exc:  # never let a fast-path failure abort the run
            print(
                f"[eval] JAX fast-path failed ({type(exc).__name__}: {exc}); "
                f"falling back to env.run for {jax_opps}",
                flush=True,
            )
            cpu_opps = list(cfg.eval_opponents)

        # When training 4-player (FFA), also report a 1-vs-3 win rate vs each JAX
        # opponent — the 2-player run above is only a 1v1 sanity signal. Logged
        # under a "4p_<opp>" key so it shows as eval/win_rate_4p_<opp> alongside
        # the 1v1 number. Same scoring scheme as arena.panel(num_players=4), so the
        # in-training and post-training 4p numbers are comparable.
        if getattr(cfg.env, "num_agents", 2) == 4:
            from .eval_jax import run_eval_jax_4p

            try:
                wr4 = run_eval_jax_4p(params, cfg, jax_opps, base_seed)
                results.update({f"4p_{opp}": wr for opp, wr in wr4.items()})
            except Exception as exc:  # 4p eval is additive — never abort the run
                print(
                    f"[eval] 4-player JAX eval failed "
                    f"({type(exc).__name__}: {exc}); skipping 4p metrics",
                    flush=True,
                )

    if cpu_opps:
        from .eval_mp import run_eval_mp

        results.update(
            run_eval_mp(params, cfg, base_seed, workers=cfg.eval_workers, opponents=cpu_opps)
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT")
    parser.add_argument("--ref_policy", type=str, default=None, metavar="CKPT",
                        help="Frozen reference policy for KL regularization. "
                             "Defaults to the --resume checkpoint when kl_coef > 0.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--total_updates", type=int, default=None)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def save_checkpoint(save_dir: Path, run_name: str, update: int, train_state: TrainState, cfg: TrainConfig) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "update":    update,
        "params":    train_state.params,
        "opt_state": train_state.opt_state,
        "config":    cfg,
    }
    for path in [run_dir / "ckpt_last.pkl", run_dir / f"ckpt_{update:06d}.pkl"]:
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)


def _restore_params(fresh_params, ckpt_params):
    """Load checkpoint params into the current model, tolerating a top-level
    module-key mismatch.

    When the trees match exactly (a normal same-arch resume) this is a plain
    strict load. When they differ — e.g. a separate-encoder BC checkpoint loaded
    into a ``shared_backbone`` model — take every top-level module the model
    still has and the checkpoint provides (matching structure), keep fresh init
    for any the model has but the checkpoint lacks, and drop any the model no
    longer has (the always-cold ``value_encoder``/``value_xsrc``). The exact
    disposition is printed so a genuine wrong-checkpoint mistake stays visible.
    """
    if jax.tree_util.tree_structure(fresh_params) == jax.tree_util.tree_structure(ckpt_params):
        return ckpt_params
    fresh, loaded = fresh_params["params"], ckpt_params["params"]
    merged, taken, kept = dict(fresh), [], []
    for name, sub in fresh.items():
        if name in loaded and jax.tree_util.tree_structure(loaded[name]) == jax.tree_util.tree_structure(sub):
            merged[name] = loaded[name]
            taken.append(name)
        else:
            kept.append(name)
    dropped = [name for name in loaded if name not in fresh]
    print("  partial restore (param tree changed since checkpoint):")
    print(f"    loaded from ckpt : {sorted(taken)}")
    print(f"    kept fresh init  : {sorted(kept)}")
    print(f"    dropped from ckpt: {sorted(dropped)}")
    return {"params": merged}


def _enable_compile_cache() -> None:
    """Point JAX at an on-disk compilation cache so the (slow, multi-minute) first
    compile of the rollout + ppo_update XLA programs is written to disk once and
    reused on every later launch — restarts, resumes, re-submits — instead of
    recompiling from scratch each time. Mirrors the BC encode path's cache. Shared
    dir with BC via JAX_CACHE_DIR (defaults to <repo>/.jax_cache)."""
    import os
    cache_dir = os.environ.get(
        "JAX_CACHE_DIR",
        os.environ.get("BC_JAX_CACHE_DIR",
                       str(Path(__file__).resolve().parents[1] / ".jax_cache")),
    )
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    print(f"JAX persistent compile cache: {cache_dir}", flush=True)


def main() -> None:
    from .accel import setup_accelerator
    setup_accelerator()  # CPU / GPU / TPU — pins matmul precision for backend parity
    _enable_compile_cache()

    args  = parse_args()
    cfg   = load_train_config(args.config)

    # Optional matmul-precision override (e.g. "default"/"bfloat16" to unlock
    # bf16 tensor cores on H100). Safe to set here: it's only a config flag, and
    # nothing has been compiled yet (build_train_state / rollout compile below).
    if getattr(cfg.ppo, "matmul_precision", ""):
        jax.config.update("jax_default_matmul_precision", cfg.ppo.matmul_precision)
        print(f"matmul precision override: {cfg.ppo.matmul_precision}", flush=True)

    N     = cfg.ppo.num_envs if cfg.ppo.num_envs > 0 else 10
    cfg.ppo.num_envs = N
    n_boards = cfg.ppo.n_boards
    K     = cfg.env.candidate_count
    use_wandb = not args.no_wandb
    run_name = args.run_name or cfg.run_name
    # CLI flags take precedence over yaml config for resume / KL reference.
    resume = args.resume or cfg.resume
    ref_policy = args.ref_policy or cfg.ref_policy
    if args.total_updates:
        cfg.ppo.total_updates = args.total_updates
    if args.augment:
        cfg.ppo.augment = True

    if use_wandb:
        import wandb

        wandb.init(
            project="orbit-war",
            name=run_name,
            config={
                "seed": cfg.seed, "num_envs": N,
                "n_boards": n_boards,
                "opponents": cfg.opponents,
                **{f"env/{k}": v for k, v in dataclasses.asdict(cfg.env).items()},
                **{f"model/{k}": v for k, v in dataclasses.asdict(cfg.model).items()},
                **{f"ppo/{k}": v for k, v in dataclasses.asdict(cfg.ppo).items()},
            },
        )

    print(f"Using {N} parallel environments, board pool size {n_boards}")

    # ── Board pool ────────────────────────────────────────────────────────────
    board_pool = make_board_pool(
        n_boards, cfg.seed, num_agents=cfg.env.num_agents,
        mix_4player_frac=cfg.env.mix_4player_frac,
    )

    # ── Encoder (shared between rollout and bootstrap value) ──────────────────
    encode_batch = make_env_encoder(K, cfg.env, getattr(cfg.model, "feat_version", "v7"))

    # ── Policy module + train state ───────────────────────────────────────────
    policy      = build_policy(cfg)
    train_state = build_train_state(cfg, seed=cfg.seed)
    key         = jax.random.PRNGKey(cfg.seed)

    learner_version = getattr(cfg.model, "feat_version", "v7")
    start_update = 1
    ref_params = None
    ref_version = None
    if resume:
        with open(resume, "rb") as f:
            ckpt = pickle.load(f)
        ckpt_params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        restored = {"params": _restore_params(train_state.params, ckpt_params)}
        if "opt_state" in ckpt:
            restored_opt = jax.tree_util.tree_map(jnp.asarray, ckpt["opt_state"])
            # Only reuse the saved optimizer state if it matches the current
            # optimizer; the structure changes when e.g. weight_decay turns
            # Adam into AdamW. Otherwise keep params but start the optimizer fresh.
            if jax.tree_util.tree_structure(restored_opt) == jax.tree_util.tree_structure(train_state.opt_state):
                restored["opt_state"] = restored_opt
            else:
                print("  optimizer structure changed since checkpoint "
                      "(e.g. weight_decay / Adam→AdamW); reinitializing optimizer state.")
        train_state = train_state.replace(**restored)
        start_update = ckpt["update"] + 1
        print(f"Resumed from {resume} (update {ckpt['update']})")
        if cfg.ppo.kl_coef > 0 and ref_policy is None:
            ref_params = restored["params"]
            ref_version = learner_version
            print(f"KL reference: resume checkpoint (kl_coef={cfg.ppo.kl_coef})")

    if ref_policy:
        with open(ref_policy, "rb") as f:
            ref_ckpt = pickle.load(f)
        ref_params = jax.tree_util.tree_map(jnp.asarray, ref_ckpt["params"])
        ref_version = cfg.ref_policy_feat_version or learner_version
        print(f"KL reference: {ref_policy} (feat_version={ref_version}, kl_coef={cfg.ppo.kl_coef})")

    # Build opponent pool from normalised specs (self / heuristic / ckpt[, feat_version]).
    # Heuristic slots store dummy params (learner params) that are never used; the
    # rollout bypasses neural inference for those envs entirely. A ckpt opponent may
    # use a different feat_version (encoder) than the learner — its params are run
    # through the same dim-agnostic module on its own-version features.
    opp_specs = parse_opponents(cfg.opponents)
    opp_pool: list[tuple[OpponentSpec, object]] = []
    opp_versions: set[str] = set()
    for spec in opp_specs:
        if spec.kind == "self":
            opp_pool.append((spec, train_state.params))
            opp_versions.add(learner_version)
        elif spec.kind == "heuristic":
            opp_pool.append((spec, train_state.params))  # dummy — never used
            print("Heuristic opponent registered (JAX stateless greedy).")
        elif spec.kind == "ckpt":
            with open(spec.path, "rb") as f:
                opp_ckpt = pickle.load(f)
            opp_pool.append((spec, jax.tree_util.tree_map(jnp.asarray, opp_ckpt["params"])))
            v = spec.feat_version or learner_version
            opp_versions.add(v)
            print(f"Checkpoint opponent loaded: {spec.path} (feat_version={v})")
        else:
            raise ValueError(f"Unknown opponent kind {spec.kind!r}.")

    # All neural opponents in one stacked pool must share an encoder (their params
    # are stacked along a leading axis, so the shapes must match).
    if len(opp_versions) > 1:
        raise ValueError(
            f"mixed opponent feat_versions {sorted(opp_versions)} in one pool is "
            f"unsupported; use a single opponent feat_version per run."
        )
    opp_version = next(iter(opp_versions), learner_version)
    opp_encode_batch = (
        encode_batch if opp_version == learner_version
        else make_env_encoder(K, cfg.env, opp_version)
    )

    heuristic_opp_mask = tuple(spec.kind == "heuristic" for spec, _ in opp_pool)

    # Stack pool into [n_opponents, ...] — env i uses slot i % n_opponents.
    stacked_opp_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[p for _, p in opp_pool]
    )

    # KL reference encoder: the frozen ref encodes the learner's visited states with
    # its own (possibly different) feat_version; its logits are precomputed in the
    # rollout and read by the update. None unless KL is active.
    compute_ref_logits = ref_params is not None and cfg.ppo.kl_coef > 0
    ref_encode_batch = None
    if compute_ref_logits:
        ref_encode_batch = (
            encode_batch if (ref_version or learner_version) == learner_version
            else make_env_encoder(K, cfg.env, ref_version)
        )
        print(f"KL reference encoder: feat_version={ref_version or learner_version}")

    # ── JIT-compiled rollout fn ───────────────────────────────────────────────
    collect = make_rollout_fn(
        board_pool          = board_pool,
        n_boards            = n_boards,
        cfg                 = cfg,
        module              = policy,
        rollout_steps       = cfg.ppo.rollout_steps,
        n_opponents         = len(opp_pool),
        heuristic_opp_mask  = heuristic_opp_mask,
        augment             = cfg.ppo.augment,
        opp_encode_batch    = opp_encode_batch,
        ref_params          = ref_params if compute_ref_logits else None,
        ref_encode_batch    = ref_encode_batch,
        compute_ref_logits  = compute_ref_logits,
    )

    # ── Initial carry ─────────────────────────────────────────────────────────
    batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
        board_pool, n_boards, N, cfg.seed, cfg.alternate_player_sides,
        num_agents=cfg.env.num_agents,
    )
    carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)

    save_dir = Path(cfg.save_dir)

    print("Compiling JAX rollout (first update may be slow) …", flush=True)

    t0 = time.time()
    try:
        for update in range(start_update, cfg.ppo.total_updates + 1):
            t_update = time.time()
            # ── Collect rollout ───────────────────────────────────────────────
            carry, steps = collect(train_state.params, stacked_opp_params, carry)
            jax.block_until_ready(steps)
            t_collect = time.time() - t_update

            # ── Bootstrap value from last carry state ─────────────────────────
            boot_val = compute_bootstrap_val(
                policy, train_state.params,
                carry.state, carry.learner_players,
                encode_batch, K, N,
            )

            # ── Returns and advantages ────────────────────────────────────────
            step_returns = compute_returns(
                steps.reward, steps.done, steps.values, steps.src_valid,
                boot_val, cfg.ppo.gamma, cfg.ppo.gae_lambda,
            )

            # ── Build TrainingBatch ───────────────────────────────────────────
            batch = build_training_batch(steps, step_returns)
            jax.block_until_ready(batch.self_features)
            t_batch = time.time() - t_update - t_collect

            # ── PPO update ────────────────────────────────────────────────────
            train_state, metrics, key = ppo_update(
                train_state, batch,
                clip_coef=cfg.ppo.clip_coef,
                ent_coef=cfg.ppo.ent_coef,
                vf_coef=cfg.ppo.vf_coef,
                epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                key=key,
                kl_coef=cfg.ppo.kl_coef,
                use_kl=compute_ref_logits,
                max_grad_norm=cfg.ppo.max_grad_norm,
            )
            jax.block_until_ready(train_state.params)
            t_ppo = time.time() - t_update - t_collect - t_batch

            # ── Self-play sync: refresh all "self" slots in the stacked pool ───
            updates_since_start = update - start_update
            if (
                any(spec.kind == "self" for spec, _ in opp_pool)
                and updates_since_start > 0
                and updates_since_start % cfg.self_play_update_interval == 0
            ):
                for i, (spec, _) in enumerate(opp_pool):
                    if spec.kind == "self":
                        stacked_opp_params = jax.tree_util.tree_map(
                            lambda s, p: s.at[i].set(p), stacked_opp_params, train_state.params
                        )

            # ── Eval vs rule-based opponents ──────────────────────────────────
            eval_results: dict[str, float] = {}
            if cfg.eval_opponents and update % cfg.eval_update_interval == 0:
                eval_results = run_eval(train_state.params, cfg)
                for opp, wr in eval_results.items():
                    print(f"  eval vs {opp}: win_rate={wr:.3f}", flush=True)

            # ── Logging ───────────────────────────────────────────────────────
            if update % cfg.log_every == 0:
                stats = episode_stats(steps)
                elapsed = time.time() - t0
                update_time = time.time() - t_update
                updates_per_s = (update - start_update + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"update={update}"
                    f" reward={stats['episode_reward_mean']:.4f}"
                    f" episodes={int(stats['episodes_finished'])}"
                    f" samples={int(stats['samples'])}"
                    f" loss={metrics['loss']:.4f}"
                    f" ev={metrics['explained_variance']:.3f}"
                    f" entropy={metrics['entropy']:.3f}"
                    f" clip={metrics['clip_fraction']:.3f}"
                    f" gnorm={metrics['grad_norm']:.3f}"
                    f" gclip={metrics['grad_clip_fraction']:.3f}"
                    f"  {elapsed:.0f}s ({updates_per_s:.2f} updates/s, {update_time:.1f}s/update)"
                    f"  [collect={t_collect:.2f}s batch={t_batch:.2f}s ppo={t_ppo:.2f}s]",
                    flush=True,  # else stdout block-buffers to the slurm file and
                                 # progress is invisible for ~1-2h (buffer ~4-8KB)
                )
                if use_wandb:
                    import wandb

                    wandb.log(
                        {
                            **{f"eval/win_rate_{opp}": wr for opp, wr in eval_results.items()},
                            "reward/episode_mean":      stats["episode_reward_mean"],
                            "reward/episodes_finished": stats["episodes_finished"],
                            "train/samples":            stats["samples"],
                            "train/loss":               metrics["loss"],
                            "train/policy_loss":        metrics["policy_loss"],
                            "train/value_loss":         metrics["value_loss"],
                            "train/entropy":            metrics["entropy"],
                            "train/clip_fraction":      metrics["clip_fraction"],
                            "train/kl_divergence":      metrics["kl_divergence"],
                            "train/kl_ref":             metrics["kl_ref"],
                            "train/grad_norm":          metrics["grad_norm"],
                            "train/grad_clip_fraction": metrics["grad_clip_fraction"],
                            "train/explained_variance": metrics["explained_variance"],
                        },
                        step=update,
                    )

            if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
                save_checkpoint(save_dir, run_name, update, train_state, cfg)

    finally:
        if use_wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
