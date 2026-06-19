#!/usr/bin/env bash
# Create the venv used to AOT-compile the submission's encoder so the serialized
# executable LOADS on Kaggle instead of falling back to compiling (the 4-player
# timeout). It must match Kaggle's competition runner exactly:
#   gcr.io/kaggle-images/python:v163  ->  jax==0.5.2 / jaxlib==0.5.1
# (the orbit_wars env pins that base image; kaggle_environments leaves jax
# unconstrained, so the image's version wins).
#
# Usage:
#   bash tools/make_aot_venv.sh            # creates ./.venv_aot051
# Then regenerate normally — gen auto-detects .venv_aot051 for the AOT step:
#   python tools/gen_ppo_submission.py --ckpt <ckpt> --out submission/<name>.py
# (override with AOT_PYTHON=/path/to/python if your venv lives elsewhere).
set -euo pipefail
VENV="${1:-.venv_aot051}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --only-binary=:all: "jax==0.5.2" "jaxlib==0.5.1" "flax==0.10.6"
"$VENV/bin/python" - <<'PY'
import jax, jaxlib, flax
print(f"AOT venv ready: jax {jax.__version__} / jaxlib {jaxlib.__version__} / flax {flax.__version__}")
assert jaxlib.__version__ == "0.5.1", "jaxlib must be 0.5.1 to match Kaggle"
PY
