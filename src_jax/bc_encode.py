"""
Pre-encode a day's episode zip into sharded .npz files for fast BC training.

Run once per date directory before training:
    python -m src_jax.bc_encode --episodes_dir episodes/2026-05-24

Output: episodes/2026-05-24/shards/shard_NNNN.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bc_data import SHARD_DIR_NAME, encode_to_shards
from .config import default_train_config_path, load_train_config
from .features import _make_env_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes_dir", required=True,
                        help="Date directory containing the episode zip (e.g. episodes/2026-05-24)")
    parser.add_argument("--config", default=str(default_train_config_path()))
    parser.add_argument("--episodes_per_shard", type=int, default=100,
                        help="Episodes per shard file (default: 100)")
    parser.add_argument("--winner_only", action="store_true",
                        help="Encode only the winning agent's actions")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if shards already exist")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    encode_fn = _make_env_encoder(cfg.env.candidate_count, cfg.env)

    episodes_dir = Path(args.episodes_dir)
    print(f"Encoding {episodes_dir} …")
    shard_dir = encode_to_shards(
        episodes_dir,
        encode_fn,
        cfg.env.ship_bucket_count,
        episodes_per_shard=args.episodes_per_shard,
        winner_only=args.winner_only,
        force=args.force,
    )
    print(f"Done → {shard_dir}")


if __name__ == "__main__":
    main()
