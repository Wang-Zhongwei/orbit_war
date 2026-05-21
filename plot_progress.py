"""Plot PPO training progress from a JSON file produced by evaluate_progress.py.

Usage
-----
    python plot_progress.py
    python plot_progress.py --data results/progress.json --out artifacts/progress.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("results/progress.json"))
    p.add_argument("--out", type=Path, default=Path("artifacts/progress.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.data.read_text())

    updates = [r["update"] for r in records]
    opponents = [k for k in records[0] if k not in ("update", "checkpoint")]

    fig, ax = plt.subplots(figsize=(10, 5))
    for opp in opponents:
        ax.plot(updates, [r[opp] for r in records], marker="o", markersize=4, label=opp)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Win rate (draw=0.5)")
    ax.set_title("PPO agent win rate over training")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
