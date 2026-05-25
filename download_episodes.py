"""
Download top-rated episode replays from the Kaggle orbit-wars dataset.

Usage:
    python download_episodes.py [--top N] [--out_dir DIR] [--dataset SLUG]

Requires the kaggle CLI to be installed and authenticated (kaggle.json in ~/.kaggle/).
Downloads manifest.csv first, sorts by avg_score descending, then fetches each episode.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DATASET = "kaggle/orbit-wars-episodes-2026-05-24"
DEFAULT_TOP = 500
DEFAULT_OUT = "episodes"


def kaggle_download_file(dataset: str, filename: str, out_dir: Path) -> Path:
    out_path = out_dir / filename
    if out_path.exists():
        return out_path
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset,
        "--file", filename,
        "-p", str(out_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed for {filename}:\n{result.stderr.strip()}"
        )
    return out_path


def load_manifest(dataset: str, out_dir: Path) -> list[dict]:
    """Download manifest.csv and return rows sorted by avg_score descending."""
    manifest_path = kaggle_download_file(dataset, "manifest.csv", out_dir)
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r["avg_score"]), reverse=True)
    return rows


def download_episodes(
    dataset: str,
    out_dir: Path,
    top: int,
    retry: int = 3,
    delay: float = 1.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching manifest from {dataset} …")
    rows = load_manifest(dataset, out_dir)
    rows = rows[:top]
    print(f"Downloading top {len(rows)} episodes …")

    done = skipped = failed = 0
    for i, row in enumerate(rows):
        ep_id = row["episode_id"]
        fname = f"{ep_id}.json"
        dest = out_dir / fname
        if dest.exists():
            skipped += 1
            continue

        for attempt in range(retry):
            try:
                kaggle_download_file(dataset, fname, out_dir)
                done += 1
                break
            except RuntimeError as e:
                if attempt < retry - 1:
                    time.sleep(delay * (2 ** attempt))
                else:
                    print(f"  FAILED {fname}: {e}", file=sys.stderr)
                    failed += 1

        if (i + 1) % 50 == 0:
            print(
                f"  {i+1}/{len(rows)}: downloaded={done} skipped={skipped} failed={failed}"
            )

    print(f"\nDone. downloaded={done} skipped={skipped} failed={failed}")
    print(f"Episodes saved to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"Number of top episodes to download (default: {DEFAULT_TOP})")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help=f"Kaggle dataset slug (default: {DATASET})")
    args = parser.parse_args()
    download_episodes(args.dataset, Path(args.out_dir), args.top)


if __name__ == "__main__":
    main()
