"""Evaluate PPO win rate across checkpoints and save results to JSON.

Usage
-----
    python evaluate_progress.py
    python evaluate_progress.py --opponents never_miss sub_v3 --rounds 10 --step 100
    python evaluate_progress.py --out results/progress.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import jax
import jax.numpy as jnp

from src_jax.jax_agent import build_agent
from src.evaluation import AGENTS, eval_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/orbit_wars_ppo"))
    p.add_argument("--opponents", nargs="+", default=["never_miss", "sub_v3"])
    p.add_argument("--rounds", type=int, default=10,
                   help="Matches per opponent.")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--step", type=int, default=100,
                   help="Evaluate every N checkpoints (default 100).")
    p.add_argument("--out", type=Path, default=Path("results/progress.json"))
    return p.parse_args()


def find_checkpoints(ckpt_dir: Path, step: int) -> list[tuple[int, Path]]:
    results = []
    for p in sorted(ckpt_dir.glob("ckpt_*.pkl")):
        m = re.fullmatch(r"ckpt_(\d+)\.pkl", p.name)
        if m and int(m.group(1)) % step == 0:
            results.append((int(m.group(1)), p))
    return results


def eval_checkpoint(
    ckpt_path: Path,
    opponent_fns: dict[str, object],
    rounds: int,
    base_seed: int,
) -> dict[str, float]:
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    return eval_agent(build_agent(params, ckpt["config"]), opponent_fns, rounds, base_seed)


def main() -> None:
    args = parse_args()

    for opp in args.opponents:
        if opp not in AGENTS:
            raise KeyError(f"Unknown opponent {opp!r}. Known: {sorted(AGENTS)}")
    opponent_fns = {opp: AGENTS[opp] for opp in args.opponents}

    checkpoints = find_checkpoints(args.ckpt_dir, args.step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints (step={args.step}) found in {args.ckpt_dir}")

    print(f"Evaluating {len(checkpoints)} checkpoints × {len(args.opponents)} opponents "
          f"× {args.rounds} matches each")

    records: list[dict] = []
    for i, (update, ckpt_path) in enumerate(checkpoints):
        print(f"[{i+1}/{len(checkpoints)}] {ckpt_path.name}", end="  ", flush=True)
        win_rates = eval_checkpoint(ckpt_path, opponent_fns, args.rounds, args.base_seed)
        records.append({"update": update, "checkpoint": ckpt_path.name, **win_rates})
        print("  ".join(f"{opp}={win_rates[opp]:.2f}" for opp in args.opponents))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
