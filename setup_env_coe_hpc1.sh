#!/usr/bin/env bash
set -e

module load git-2.42.0-tgd62zd
module load gcc-12.2.0-gcc-11.2.0-7gle75f
module load rust-1.65.0-gcc-9.5.0-y2emkvg

uv pip install -r requirements.txt --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Done. Verify with: .venv/bin/python -c \"from kaggle_environments import make; make('orbit_wars'); print('OK')\""
