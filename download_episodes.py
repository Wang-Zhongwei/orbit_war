"""
Download the full orbit-wars episode dataset from Kaggle.

Usage:
    python download_episodes.py [--out_dir DIR] [--dataset SLUG]

Requires the kaggle CLI installed and authenticated (~/.kaggle/kaggle.json).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET = "kaggle/orbit-wars-episodes-2026-05-24"
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _dataset_out_dir(dataset: str) -> Path:
    m = _DATE_RE.search(dataset)
    date = m.group(1) if m else dataset.split("/")[-1]
    return Path("episodes") / date


def _extract_manifest(out_dir: Path) -> None:
    zips = sorted(out_dir.glob("*.zip"))
    if not zips:
        return
    with zipfile.ZipFile(zips[0]) as zf:
        if "manifest.csv" in zf.namelist():
            zf.extract("manifest.csv", out_dir)
            print(f"Extracted manifest.csv → {out_dir / 'manifest.csv'}")


def download_episodes(dataset: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading full dataset {dataset} → {out_dir} …")
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    _extract_manifest(out_dir)
    print(f"Done. Episodes saved to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: episodes/<date> parsed from dataset slug)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _dataset_out_dir(args.dataset)
    download_episodes(args.dataset, out_dir)


if __name__ == "__main__":
    main()
