"""
Pre-encode a day's episode zip into sharded .npz files for fast BC training.

Usage:
    # Formal mode — env config comes from YAML:
    python -m src_jax.bc_encode --episodes_dir episodes/2026-05-24 \\
                                 --config configs/bc_default.yaml

    # Smoke-test mode — uses dataclass defaults, no config file needed:
    python -m src_jax.bc_encode --episodes_dir episodes/2026-05-24

    # Recursive mode — encode every date subdirectory under a parent.
    # Dirs that already have shards are skipped (use --force to re-encode):
    python -m src_jax.bc_encode --episodes_dir episodes --recursive

Output: episodes/2026-05-24/shards/shard_NNNN.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bc_data import POOL_K, encode_to_shards, is_shard_dir_name
from .config import TrainConfig, load_train_config
from .features import _make_env_encoder


def _find_episode_dirs(root: Path) -> list[Path]:
    """Return every subdirectory under ``root`` that holds raw episode files.

    A directory qualifies if it directly contains a ``*.zip`` or ``*.json``
    episode file. Any generated ``shards``/``shards_<tag>`` subdirectory is ignored
    so re-runs don't treat already-encoded output as a new episode dir.
    """
    dirs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir() or is_shard_dir_name(path.name):
            continue
        if any(path.glob("*.zip")) or any(path.glob("*.json")):
            dirs.append(path)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes_dir", required=True,
                        help="Date directory containing the episode zip (e.g. episodes/2026-05-24). "
                             "With --recursive, a parent dir whose date subdirs are each encoded.")
    parser.add_argument("--recursive", action="store_true",
                        help="Treat --episodes_dir as a parent and encode every subdirectory that "
                             "contains episode files. Dirs that already have shards are skipped "
                             "unless --force is given.")
    parser.add_argument("--config", default=None,
                        help="TrainConfig YAML (formal mode). Omit to use dataclass defaults "
                             "(smoke-test mode).")
    parser.add_argument("--episodes_per_shard", type=int, default=100,
                        help="Episodes per shard file (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if shards already exist")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Worker processes for parallel encoding. -1=all CPUs, 1=single-threaded (default: -1)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Stop after writing this many shards (useful for quick tests).")
    parser.add_argument("--shard_tag", default=None,
                        help="Feature-experiment version: write to episodes/<date>/shards_<tag>/ "
                             "instead of .../shards/. Defaults to cfg.bc.shard_tag from --config. "
                             "Train with the matching cfg.bc.shard_tag.")
    args = parser.parse_args()

    cfg = load_train_config(args.config) if args.config is not None else TrainConfig()
    # CLI overrides the config; fall back to cfg.bc.shard_tag so a YAML can carry it.
    shard_tag = args.shard_tag if args.shard_tag is not None else cfg.bc.shard_tag

    # Shards are encoded once at the fixed K-pool size (POOL_K) and re-sliced to the
    # training K offline via assemble_for_k — so the encoded pool is decoupled from
    # cfg.env.candidate_count. POOL_K is the single source of truth; bump it (and
    # re-encode) if you ever need to train at K > POOL_K.
    print(f"Encoding candidate pool at K={POOL_K} (train at any K <= {POOL_K})")

    root = Path(args.episodes_dir)
    if args.recursive:
        targets = _find_episode_dirs(root)
        if not targets:
            print(f"No episode directories found under {root}")
            return
        print(f"Recursive: found {len(targets)} episode dir(s) under {root}")
    else:
        targets = [root]

    for episodes_dir in targets:
        print(f"Encoding {episodes_dir} …")
        shard_dir = encode_to_shards(
            episodes_dir,
            encode_fn=None,
            episodes_per_shard=args.episodes_per_shard,
            num_shards=args.num_shards,
            force=args.force,
            workers=args.workers,
            encode_fn_factory=_make_env_encoder,
            encode_fn_factory_args=(POOL_K, cfg.env),
            shard_tag=shard_tag,
        )
        print(f"Done → {shard_dir}")


if __name__ == "__main__":
    main()
