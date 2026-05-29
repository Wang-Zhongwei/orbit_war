"""
Pre-encode a day's episode zip into sharded .npz files for fast BC training.

Usage:
    # Formal mode — env config comes from YAML:
    python -m src_jax.bc_encode --episodes_dir episodes/2026-05-24 \\
                                 --config src_jax/configs/bc_cfg.yaml

    # Smoke-test mode — uses dataclass defaults, no config file needed:
    python -m src_jax.bc_encode --episodes_dir episodes/2026-05-24

Output: episodes/2026-05-24/shards/shard_NNNN.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bc_data import encode_to_shards
from .config import TrainConfig, load_train_config
from .features import _make_env_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes_dir", required=True,
                        help="Date directory containing the episode zip (e.g. episodes/2026-05-24)")
    parser.add_argument("--config", default=None,
                        help="TrainConfig YAML (formal mode). Omit to use dataclass defaults "
                             "(smoke-test mode).")
    parser.add_argument("--episodes_per_shard", type=int, default=100,
                        help="Episodes per shard file (default: 100)")
    parser.add_argument("--winner_only", action="store_true",
                        help="Encode only the winning agent's actions")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if shards already exist")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Worker processes for parallel encoding. -1=all CPUs, 1=single-threaded (default: -1)")
    args = parser.parse_args()

    cfg = load_train_config(args.config) if args.config is not None else TrainConfig()

    episodes_dir = Path(args.episodes_dir)
    print(f"Encoding {episodes_dir} …")
    shard_dir = encode_to_shards(
        episodes_dir,
        encode_fn=None,
        ship_bucket_count=cfg.env.ship_bucket_count,
        episodes_per_shard=args.episodes_per_shard,
        winner_only=args.winner_only,
        force=args.force,
        workers=args.workers,
        encode_fn_factory=_make_env_encoder,
        encode_fn_factory_args=(cfg.env.candidate_count, cfg.env),
    )
    print(f"Done → {shard_dir}")


if __name__ == "__main__":
    main()
