#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./delete_episode_shards.sh [--episodes-dir DIR] [--yes] [--dry-run]

Delete behavior-cloning shard files generated under:
  episodes/<date>/shards/shard_*.npz

Options:
  --episodes-dir DIR  Episodes root to clean. Defaults to ./episodes.
  --yes              Delete without an interactive confirmation prompt.
  --dry-run          Print matching shard files without deleting them.
  -h, --help         Show this help message.
EOF
}

episodes_dir="episodes"
assume_yes=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --episodes-dir requires a path" >&2
        exit 2
      fi
      episodes_dir="$2"
      shift 2
      ;;
    --yes|-y)
      assume_yes=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$episodes_dir" ]]; then
  echo "error: episodes directory does not exist: $episodes_dir" >&2
  exit 1
fi

shards=()
while IFS= read -r -d '' shard; do
  shards+=("$shard")
done < <(find "$episodes_dir" -type f -path '*/shards/shard_*.npz' -print0 | sort -z)

if [[ ${#shards[@]} -eq 0 ]]; then
  echo "No shard files found under $episodes_dir."
  exit 0
fi

echo "Found ${#shards[@]} shard file(s) under $episodes_dir:"
printf '  %s\n' "${shards[@]}"

if [[ "$dry_run" == true ]]; then
  echo "Dry run only; no files were deleted."
  exit 0
fi

if [[ "$assume_yes" != true ]]; then
  printf 'Delete these shard files? Type "yes" to continue: '
  read -r answer
  if [[ "$answer" != "yes" ]]; then
    echo "Aborted; no files were deleted."
    exit 1
  fi
fi

for shard in "${shards[@]}"; do
  rm -f -- "$shard"
done

while IFS= read -r -d '' shard_dir; do
  rmdir --ignore-fail-on-non-empty -- "$shard_dir"
done < <(find "$episodes_dir" -type d -name shards -print0)

echo "Deleted ${#shards[@]} shard file(s)."
