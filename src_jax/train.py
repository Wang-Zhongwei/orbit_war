from __future__ import annotations

import argparse
import dataclasses
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from .features import _make_env_encoder
from .config import TrainConfig, default_train_config_path, load_train_config
from .env import make_board_pool, make_initial_carry_state
from .policy import build_policy
from .ppo import TrainState, build_train_state, ppo_update
from .rollout import (
    RolloutCarry,
    build_training_batch,
    compute_bootstrap_val,
    compute_returns,
    episode_stats,
    make_rollout_fn,
)

N_BOARDS = 512  # board pool size
_EVAL_BASE_SEED = 1000

def run_eval(params, cfg, base_seed: int = _EVAL_BASE_SEED) -> dict[str, float]:
    from .jax_agent import build_agent
    from src.evaluation import AGENTS, eval_agent

    agent = build_agent(params, cfg, deterministic=True)
    opponent_fns = {opp: AGENTS[opp] for opp in cfg.eval_opponents}
    return eval_agent(agent, opponent_fns, cfg.eval_rounds, base_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT")
    parser.add_argument("--ref_policy", type=str, default=None, metavar="CKPT",
                        help="Frozen reference policy for KL regularization. "
                             "Defaults to the --resume checkpoint when kl_coef > 0.")
    parser.add_argument("--run_name", type=str, default=None)
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


def main() -> None:
    args  = parse_args()
    cfg   = load_train_config(args.config)
    N     = cfg.ppo.num_envs if cfg.ppo.num_envs > 0 else 10
    cfg.ppo.num_envs = N
    K     = cfg.env.candidate_count
    use_wandb = not args.no_wandb
    run_name = args.run_name or cfg.run_name

    if use_wandb:
        import wandb

        wandb.init(
            project="orbit-war",
            name=run_name,
            config={
                "seed": cfg.seed, "num_envs": N,
                "n_boards": N_BOARDS,
                "opponents": cfg.opponents,
                **{f"env/{k}": v for k, v in dataclasses.asdict(cfg.env).items()},
                **{f"model/{k}": v for k, v in dataclasses.asdict(cfg.model).items()},
                **{f"ppo/{k}": v for k, v in dataclasses.asdict(cfg.ppo).items()},
            },
        )

    print(f"Using {N} parallel environments, board pool size {N_BOARDS}")

    # ── Board pool ────────────────────────────────────────────────────────────
    board_pool = make_board_pool(N_BOARDS, cfg.seed)

    # ── Encoder (shared between rollout and bootstrap value) ──────────────────
    encode_batch = _make_env_encoder(K, cfg.env)

    # ── Policy module + train state ───────────────────────────────────────────
    policy      = build_policy(cfg)
    train_state = build_train_state(cfg, seed=cfg.seed)
    key         = jax.random.PRNGKey(cfg.seed)

    start_update = 1
    ref_params = None
    if args.resume:
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        restored = {"params": jax.tree_util.tree_map(jnp.asarray, ckpt["params"])}
        if "opt_state" in ckpt:
            restored["opt_state"] = jax.tree_util.tree_map(jnp.asarray, ckpt["opt_state"])
        train_state = train_state.replace(**restored)
        start_update = ckpt["update"] + 1
        print(f"Resumed from {args.resume} (update {ckpt['update']})")
        if cfg.ppo.kl_coef > 0 and args.ref_policy is None:
            ref_params = restored["params"]
            print(f"KL reference: resume checkpoint (kl_coef={cfg.ppo.kl_coef})")

    if args.ref_policy:
        with open(args.ref_policy, "rb") as f:
            ref_ckpt = pickle.load(f)
        ref_params = jax.tree_util.tree_map(jnp.asarray, ref_ckpt["params"])
        print(f"KL reference: {args.ref_policy} (kl_coef={cfg.ppo.kl_coef})")

    # ── JIT-compiled rollout fn ───────────────────────────────────────────────
    collect = make_rollout_fn(
        board_pool    = board_pool,
        n_boards      = N_BOARDS,
        cfg           = cfg,
        module        = policy,
        rollout_steps = cfg.ppo.rollout_steps,
    )

    # ── Initial carry ─────────────────────────────────────────────────────────
    batched_state, board_idx, ep_reward, learner_players = make_initial_carry_state(
        board_pool, N_BOARDS, N, cfg.seed, cfg.alternate_player_sides
    )
    carry = RolloutCarry(batched_state, board_idx, ep_reward, learner_players, key)

    opp_params = train_state.params

    save_dir = Path(cfg.save_dir)

    print("Compiling JAX rollout (first update may be slow) …", flush=True)

    t0 = time.time()
    try:
        for update in range(start_update, cfg.ppo.total_updates + 1):
            t_update = time.time()
            # ── Collect rollout ───────────────────────────────────────────────
            carry, steps = collect(train_state.params, opp_params, carry)

            # ── Bootstrap value from last carry state ─────────────────────────
            boot_val = compute_bootstrap_val(
                policy, train_state.params,
                carry.state, carry.learner_players,
                encode_batch, K, N,
            )

            # ── Returns and advantages ────────────────────────────────────────
            step_returns = compute_returns(
                steps.reward, steps.done, boot_val, cfg.ppo.gamma
            )

            # ── Build TrainingBatch ───────────────────────────────────────────
            batch = build_training_batch(steps, step_returns)

            # ── PPO update ────────────────────────────────────────────────────
            train_state, metrics, key = ppo_update(
                train_state, batch,
                clip_coef=cfg.ppo.clip_coef,
                ent_coef=cfg.ppo.ent_coef,
                vf_coef=cfg.ppo.vf_coef,
                epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                key=key,
                ref_params=ref_params,
                kl_coef=cfg.ppo.kl_coef,
            )

            # ── Self-play opponent sync ───────────────────────────────────────
            if "self" in cfg.opponents and update % cfg.self_play_update_interval == 0:
                opp_params = train_state.params

            # ── Eval vs rule-based opponents ──────────────────────────────────
            eval_results: dict[str, float] = {}
            if cfg.eval_opponents and update % cfg.self_eval_update_interval == 0:
                eval_results = run_eval(train_state.params, cfg)
                for opp, wr in eval_results.items():
                    print(f"  eval vs {opp}: win_rate={wr:.3f}")

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
                    f"  {elapsed:.0f}s ({updates_per_s:.2f} updates/s, {update_time:.1f}s/update)"
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
