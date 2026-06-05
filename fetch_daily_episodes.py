"""
Download Orbit Wars daily episode datasets from Kaggle.

Episodes are published once per day as the Kaggle dataset
``kaggle/orbit-wars-episodes-<YYYY-MM-DD>``. This script walks every day from a
start date through today, downloading each day's dataset into
``episodes/<date>/archive.zip``. Days that are already downloaded are skipped, so
it is safe to run repeatedly: do one big initial run to backfill, then run it
daily to pick up the newest day.

Usage:
    # Backfill everything since May 1 (the default start), then re-run daily:
    python fetch_daily_episodes.py

    # Custom range / output dir:
    python fetch_daily_episodes.py --start 2026-05-01 --end 2026-06-04
    python fetch_daily_episodes.py --out_dir episodes --force

Requires the kaggle CLI installed and authenticated
(~/.kaggle/kaggle.json or the KAGGLE_USERNAME / KAGGLE_KEY env vars).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

DATASET_PREFIX = "kaggle/orbit-wars-episodes-"
DEFAULT_START = "2026-05-01"
# A finished archive.zip is hundreds of MB; anything smaller is treated as a
# failed/partial download and re-fetched.
MIN_VALID_BYTES = 10 * 1024 * 1024  # 10 MB


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _already_downloaded(dest: Path) -> bool:
    return dest.exists() and dest.stat().st_size >= MIN_VALID_BYTES


def _extract_manifest(zip_path: Path) -> None:
    """If the archive bundles a manifest.csv, drop it next to the zip."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "manifest.csv" in zf.namelist():
                zf.extract("manifest.csv", zip_path.parent)
    except (zipfile.BadZipFile, OSError):
        pass


def download_day(day: date, out_root: Path, force: bool) -> str:
    """Download one day's dataset. Returns 'downloaded', 'skipped', 'missing', or 'failed'."""
    iso = day.isoformat()
    slug = f"{DATASET_PREFIX}{iso}"
    day_dir = out_root / iso
    dest = day_dir / "archive.zip"

    if not force and _already_downloaded(dest):
        print(f"[skip] {iso} (already have {dest})")
        return "skipped"

    day_dir.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {iso} -> {dest}")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(day_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    def _cleanup_empty() -> None:
        try:
            if not any(day_dir.iterdir()):
                day_dir.rmdir()
        except OSError:
            pass

    if result.returncode != 0:
        combined = ((result.stdout or "") + (result.stderr or "")).lower()
        # A day that isn't published yet (e.g. today's batch) returns 403/404
        # rather than a real error -> treat as "missing", not a failure.
        if any(tok in combined for tok in ("403", "404", "forbidden", "not found")):
            print(f"[miss] {iso} not published yet")
            _cleanup_empty()
            return "missing"
        print(f"[FAIL] {iso}: {combined.strip()[:500]}", file=sys.stderr)
        _cleanup_empty()
        return "failed"

    # kaggle writes <slug-tail>.zip; normalize to archive.zip.
    zips = [p for p in day_dir.glob("*.zip") if p.name != "archive.zip"]
    if zips:
        newest = max(zips, key=lambda p: p.stat().st_mtime)
        newest.replace(dest)

    if _already_downloaded(dest):
        _extract_manifest(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"[ ok ] {iso} ({size_mb:.0f} MB)")
        return "downloaded"

    print(f"[FAIL] {iso}: no usable archive.zip after download", file=sys.stderr)
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"First date to fetch, inclusive (default {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=None,
                        help="Last date to fetch, inclusive (default: today)")
    parser.add_argument("--out_dir", type=str, default="episodes",
                        help="Root output directory (default: episodes)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if archive.zip already exists")
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today()
    if start > end:
        parser.error(f"--start {start} is after --end {end}")
    out_root = Path(args.out_dir)

    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0}
    for day in _daterange(start, end):
        counts[download_day(day, out_root, args.force)] += 1

    print(
        f"\nDone {start}..{end}: "
        f"{counts['downloaded']} downloaded, {counts['skipped']} skipped, "
        f"{counts['missing']} not-yet-published, {counts['failed']} failed."
    )
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
