#!/usr/bin/env python3
"""Launch src_jax training pinned to a specific src_jax architecture root.

``src_jax/train.py`` (and ``bc_train.py``) bind their model/encoder/physics via
relative imports the moment the ``src_jax`` package is imported — so to train an
arch that differs from the checked-out branch (e.g. the ``v7arch`` worktree while
the repo sits on ``main``), the right ``src_jax`` must be on ``sys.path`` *before*
that first import happens. A plain config field read inside ``main()`` is too late.

This launcher does the pin first, then hands off to the trainer:

  1. Resolve the root from ``--jax-src-root`` (highest priority) or the
     ``jax_src_root`` key of the ``--config`` yaml.
  2. Prepend that root to ``sys.path`` and evict any already-imported ``src_jax``
     / ``src`` modules so the pinned tree wins.
  3. Import ``src_jax.<target>`` from the pinned tree and call its ``main()``.

With no root requested it is a thin passthrough that runs the live tree, so the
slurm templates can call it unconditionally.

Usage:
  python tools/train_pinned.py --target train    [--jax-src-root DIR] [trainer args...]
  python tools/train_pinned.py --target bc_train  [--jax-src-root DIR] [trainer args...]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml

PIN_ENV = "ORBIT_JAX_PINNED"


def _yaml_jax_src_root(config_path: str | None) -> str | None:
    if not config_path:
        return None
    try:
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    root = data.get("jax_src_root") if isinstance(data, dict) else None
    return str(root) if root else None


def _strip_flag(argv: list[str], *names: str) -> list[str]:
    """Drop ``--flag VALUE`` (and ``--flag=VALUE``) occurrences from argv so the
    downstream trainer's parser, which may not know these flags, won't choke."""
    out: list[str] = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in names:
            skip = True
            continue
        if any(tok.startswith(n + "=") for n in names):
            continue
        out.append(tok)
    return out


def main() -> None:
    # Parse only the launcher's own flags; leave the rest for the trainer.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--target", choices=["train", "bc_train"], default="train")
    pre.add_argument("--config", default=None)
    pre.add_argument("--jax-src-root", "--jax_src_root", dest="jax_src_root", default=None)
    known, _ = pre.parse_known_args(sys.argv[1:])

    repo_root = Path(__file__).resolve().parents[1]
    root = known.jax_src_root or _yaml_jax_src_root(known.config)
    resolved = Path(root).resolve() if root else repo_root

    if root and resolved != repo_root:
        if not (resolved / "src_jax").is_dir():
            sys.exit(f"--jax-src-root {resolved}: no src_jax/ found there")
        sys.path.insert(0, str(resolved))
        # Evict anything already imported from the live tree so the pinned tree wins.
        for mod in list(sys.modules):
            if mod == "src_jax" or mod.startswith("src_jax.") or mod == "src" or mod.startswith("src."):
                del sys.modules[mod]
        print(f"[train_pinned] pinned src_jax -> {resolved}", flush=True)
    else:
        print(f"[train_pinned] using live src_jax -> {repo_root}", flush=True)

    # Mark the process as pinned so the trainer's maybe_reexec_for_pin() is a no-op
    # (prevents an exec loop), and hand the trainer a clean argv.
    os.environ[PIN_ENV] = str(resolved)
    sys.argv = [sys.argv[0]] + _strip_flag(
        sys.argv[1:], "--target", "--jax-src-root", "--jax_src_root"
    )

    trainer = importlib.import_module(f"src_jax.{known.target}")
    trainer.main()


if __name__ == "__main__":
    main()
