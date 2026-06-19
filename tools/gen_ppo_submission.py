"""Generate a single-file Kaggle submission for a PPO checkpoint.

Bundles the minimal src_jax inference modules + slim checkpoint (params+config,
no optimizer state) as base64 inside one self-contained .py file. All JAX
compilation happens at module import time (build_agent warms up the JITs), so
the per-step agent() call stays well under the 1s actTimeout.
"""
from __future__ import annotations

import base64
import io
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

REPO = Path("/home/018270239/orbit_war")
sys.path.insert(0, str(REPO))
# Defaults preserve the original ppo_from_bc_loser0 build; override via CLI.
CKPT = REPO / "artifacts/ppo/ppo_from_bc_loser0/ckpt_000080.pkl"
OUT = REPO / "submission/ppo_from_bc_loser0.py"
MAX_TICKS_IN: int | None = None  # None = keep the value from source; set via --max-ticks-in

# Minimal module closure for inference (no rollout/ppo/optax/wandb).
BUNDLE = [
    "src_jax/__init__.py",
    "src_jax/config.py",       # needed to unpickle TrainConfig (not imported by code)
    "src_jax/constants.py",
    "src_jax/generation.py",
    "src_jax/physics.py",
    "src_jax/features.py",
    "src_jax/features_v9.py",
    "src_jax/features_v91.py",
    "src_jax/features_v92.py",
    "src_jax/feature_versions.py",
    "src_jax/bc_data.py",
    "src_jax/jax_agent.py",
    "src_jax/policies/__init__.py",
    "src_jax/policies/base.py",
    "src_jax/policies/backbone.py",
    "src_jax/policies/cross_source.py",
    "src_jax/policies/entity_transformer.py",
]


# Repo to bundle src_jax FROM — must be the code the checkpoint was trained
# with (param shapes follow its feature dims), not necessarily this repo.
SRC_REPO = REPO


def patched_source(rel: str) -> bytes:
    src = (SRC_REPO / rel).read_text()
    if rel == "src_jax/jax_agent.py":
        # The only line that drags in the training stack (rollout->ppo->optax).
        # GLOB_DIM is just global_feature_dim(); compute it directly.
        needle = "from .rollout import GLOB_DIM"
        assert needle in src, "jax_agent.py layout changed; update patch"
        src = src.replace(
            needle,
            "from .features import global_feature_dim as _global_feature_dim\n"
            "GLOB_DIM = _global_feature_dim()",
        )
        # The in-game-call warmup import also lives inside build_agent; fine as-is.
    if MAX_TICKS_IN is not None and rel.startswith("src_jax/features"):
        # Override the module-level MAX_TICKS_IN constant so the bundled encoder
        # uses a shorter horizon at inference (fewer sim ticks → faster per-step).
        src = re.sub(
            r"^(MAX_TICKS_IN\s*:\s*int\s*=\s*)\d+",
            rf"\g<1>{MAX_TICKS_IN}",
            src,
            count=1,
            flags=re.MULTILINE,
        )
    return src.encode()


def build_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE:
            zf.writestr(rel, patched_source(rel))
    return base64.b64encode(buf.getvalue()).decode()


def build_ckpt_b64() -> str:
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    # Serialize params in a VERSION-INDEPENDENT form: each array leaf becomes a
    # ("__ARR__", shape, dtype_str, raw_bytes) record reconstructed with
    # np.frombuffer at load time. Why: training checkpoints store params as JAX
    # arrays, whose pickle pulls in jax-internal `jax._src.array._reconstruct_array`
    # (and numpy 2.x's `numpy._core`). Those are fragile across jax/numpy versions
    # — when Kaggle bumps jaxlib (e.g. the `GpuAllocatorConfig` removal) the embedded
    # pickle can fail to load or force a slow full-device init. frombuffer + a dtype
    # string is a stable public API on any jax/numpy. float32 leaves are downcast to
    # float16 here (halving size); the submission upcasts back to float32 on load.
    params = _arrayify(ck["params"])
    slim = {"update": ck.get("update"), "params": params, "config": ck["config"]}
    blob = zlib.compress(pickle.dumps(slim, protocol=4), level=9)
    return base64.b64encode(blob).decode()


def _arrayify(tree):
    """Recursively convert array leaves → ("__ARR__", shape, dtype, bytes) records.

    Materializes JAX *and* numpy arrays to raw bytes so the pickle has no jax/numpy
    array dependency. float32 → float16 to shrink; everything else kept as-is.
    """
    import numpy as np
    if isinstance(tree, dict):
        return {k: _arrayify(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_arrayify(v) for v in tree)
    if hasattr(tree, "shape") and hasattr(tree, "dtype"):
        a = np.asarray(tree)
        if a.dtype == np.float32:
            a = a.astype(np.float16)
        return ("__ARR__", list(a.shape), a.dtype.str, a.tobytes())
    return tree


# Persistent (git-trackable) cache of captured warm observations. Replaces the
# old per-job tmp path that got cleaned up between runs.
WARM_OBS = REPO / "submission/warm_obs.pkl"
EPISODES_DIR = REPO / "episodes"


def _capture_warm_obs(n_max: int = 8) -> dict[int, dict]:
    """Capture a small, dtype-diverse set of real observations from the most
    recent episode replay.

    Returns {step_idx: obs_dict} covering the empty initial board AND populated
    boards with/without fleets, so every distinct shape/dtype path through
    obs_to_env_state gets compiled at submission import time.
    """
    import json

    date_dirs = sorted(d for d in EPISODES_DIR.glob("*") if d.is_dir())
    if not date_dirs:
        raise FileNotFoundError(f"no episode date dirs under {EPISODES_DIR}")

    episode = None
    for date_dir in reversed(date_dirs):  # newest first
        zips = sorted(date_dir.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as zf:
                names = sorted(n for n in zf.namelist() if n.endswith(".json"))
                if names:
                    with zf.open(names[0]) as f:
                        episode = json.load(f)
        else:
            jsons = sorted(date_dir.glob("*.json"))
            if jsons:
                episode = json.loads(jsons[0].read_text())
        if episode is not None:
            break
    if episode is None:
        raise FileNotFoundError(f"no episode JSONs found under {EPISODES_DIR}")

    # Pick one obs per distinct (n_planets, has_fleets) signature so we cover the
    # empty-board and populated/with-fleet variants without bloating the bundle.
    steps = episode["steps"]
    out: dict[int, dict] = {}
    seen: set[tuple[int, bool]] = set()
    for i, step in enumerate(steps):
        obs = step[0]["observation"]  # primary agent: includes 'step'
        sig = (len(obs.get("planets", [])), bool(obs.get("fleets")))
        if sig in seen:
            continue
        seen.add(sig)
        out[i] = obs
        if len(out) >= n_max:
            break
    if not out:
        raise ValueError("captured no observations from episode")
    return out


def build_warm_b64() -> str:
    # dict {step_idx: obs_dict} captured from a real game: covers the empty
    # initial board AND populated boards (with/without fleets) so every distinct
    # dtype/shape variant the real obs path produces is compiled at import time.
    # Use the persistent cache if present; otherwise capture fresh and cache it.
    if WARM_OBS.exists():
        blob = WARM_OBS.read_bytes()
    else:
        warm = _capture_warm_obs()
        blob = pickle.dumps(warm, protocol=4)
        WARM_OBS.parent.mkdir(parents=True, exist_ok=True)
        WARM_OBS.write_bytes(blob)
        print(f"captured + cached {len(warm)} warm obs -> {WARM_OBS}")
    return base64.b64encode(blob).decode()


# Driver run in a clean CPU subprocess to populate a persistent XLA compile
# cache. It reproduces the runtime path EXACTLY (same bundled+patched src, same
# dearrayified params, same warm-obs replay, same CPU platform) so the cache
# keys match what the submission produces at import time. Anything else (a
# different jaxlib build, GPU platform, un-patched src) yields a different HLO
# and the cache silently misses → harmless recompile.
_CACHE_DRIVER = r'''
import os, sys, pickle, zlib, base64
src_dir, cache_dir, ckpt_b64_path, warm_b64_path = sys.argv[1:5]
# All cache config via env vars set BEFORE importing jax — config.update() after
# import lands too late for the metadata-exclusion flag (jax has already frozen
# the cache-key inputs), leaving the key path-sensitive. Env vars are read at
# import time and reliably exclude HLO op metadata (source paths) from the key,
# so the bundled cache hits regardless of where src is extracted at runtime
# (this host's /tmp vs Kaggle's). MUST match the runtime template exactly.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.environ["JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY"] = "0"
sys.path.insert(0, src_dir)
import jax
import numpy as np
from src_jax.jax_agent import build_agent
def _dearrayify(t):
    if isinstance(t, tuple) and len(t) == 4 and t[0] == "__ARR__":
        _, shape, dt, buf = t
        a = np.frombuffer(buf, dtype=np.dtype(dt)).reshape(shape)
        return a.astype(np.float32) if a.dtype == np.float16 else a
    if isinstance(t, dict):
        return {k: _dearrayify(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return type(t)(_dearrayify(v) for v in t)
    return t
ckpt = pickle.loads(zlib.decompress(base64.b64decode(open(ckpt_b64_path).read())))
agent = build_agent(_dearrayify(ckpt["params"]), ckpt["config"], deterministic=True)
warm = pickle.loads(base64.b64decode(open(warm_b64_path).read()))
for ob in warm.values():
    agent(ob); agent(ob)
print("CACHE_DRIVER_OK")
'''


def build_compile_cache_b64(zip_b64: str, ckpt_b64: str, warm_b64: str, tag: str) -> str:
    """Precompile the agent on CPU and return a base64 zip of the XLA cache.

    DISABLED — returns "" so no cache is bundled. Rigorously verified (Jun 2026,
    jax/jaxlib 0.5.3, CPU) that JAX's persistent-compilation-cache KEY is
    non-deterministic across processes for this agent: two byte-identical builds
    in identical clean environments (pinned PYTHONHASHSEED, JAX_PLATFORMS=cpu,
    metadata excluded from the key) produce entirely different keys for every
    entry, including the param-free `_encode_single`. So a cache built here can
    never reliably hit at runtime, and a bundled-but-missing cache only ADDS
    unpack+write time to Kaggle's first-act (billed) window. The real first-act
    compile win is removing the comet-evac JIT in jax_agent.py. Kept as a stub
    (not deleted) to document the finding; flip the early return to re-enable if
    a future jaxlib makes keys reproducible.

    The body below is retained for reference but unreachable.
    """
    return ""
    import shutil  # noqa: unreachable — see docstring
    with tempfile.TemporaryDirectory() as td:
        src_dir = os.path.join(tempfile.gettempdir(), "owars_ppo_src_" + tag)
        cache_dir = os.path.join(td, "cache")
        shutil.rmtree(src_dir, ignore_errors=True)
        os.makedirs(src_dir); os.makedirs(cache_dir)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64))) as zf:
            zf.extractall(src_dir)
        ckpt_path = os.path.join(td, "ckpt.b64"); Path(ckpt_path).write_text(ckpt_b64)
        warm_path = os.path.join(td, "warm.b64"); Path(warm_path).write_text(warm_b64)
        driver = os.path.join(td, "driver.py"); Path(driver).write_text(_CACHE_DRIVER)
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        try:
            r = subprocess.run(
                [sys.executable, driver, src_dir, cache_dir, ckpt_path, warm_path],
                env=env, capture_output=True, text=True, timeout=600,
            )
        except Exception as e:
            print(f"  compile-cache: subprocess failed ({e}); shipping without cache")
            return ""
        if r.returncode != 0 or "CACHE_DRIVER_OK" not in r.stdout:
            print("  compile-cache: driver did not finish cleanly; shipping without cache")
            print((r.stderr or "")[-800:])
            return ""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # entries already compressed
            for name in sorted(os.listdir(cache_dir)):
                zf.write(os.path.join(cache_dir, name), name)
        n = len(os.listdir(cache_dir))
        print(f"  compile-cache: bundled {n} XLA entries ({buf.tell()/1e6:.1f} MB)")
        return base64.b64encode(buf.getvalue()).decode()


# Driver (clean CPU subprocess) that AOT-compiles the ENCODER and serializes the
# executable. Unlike the persistent cache, serialize_executable stores the
# compiled program by value (no key lookup), so it is immune to the cross-process
# key non-determinism that killed the cache — it just needs the runtime jaxlib to
# match. Loading it at import skips the encoder's ~5s trace+compile, which is what
# blows Kaggle's first-act budget when 4 instances compile at once in FFA games.
_AOT_DRIVER = r'''
import os, sys, pickle, zlib, base64
os.environ["JAX_PLATFORMS"] = "cpu"
src_dir, out_path, ckpt_b64_path = sys.argv[1:4]
sys.path.insert(0, src_dir)
import jax, jax.numpy as jnp
from jax.experimental import serialize_executable as se
from src_jax.jax_agent import _make_dummy_state
from src_jax.feature_versions import make_env_encoder
ck = pickle.loads(zlib.decompress(base64.b64decode(open(ckpt_b64_path).read())))
cfg = ck["config"]
enc = make_env_encoder(cfg.env.candidate_count, cfg.env, getattr(cfg.model, "feat_version", "v7"))
ds = _make_dummy_state()
batched = jax.tree_util.tree_map(lambda x: x[None], ds)
compiled = jax.jit(enc).lower(batched, jnp.array([0], jnp.int32)).compile()
open(out_path, "wb").write(pickle.dumps(se.serialize(compiled)))
print("AOT_OK")
'''


def build_aot_encode_b64(zip_b64: str, ckpt_b64: str) -> str:
    """AOT-compile the encoder on CPU and return base64(pickle((serialized, in_tree,
    out_tree))). "" on any failure — the submission then just compiles the encoder
    normally at import (today's behaviour)."""
    with tempfile.TemporaryDirectory() as td:
        src_dir = os.path.join(td, "src"); os.makedirs(src_dir)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64))) as zf:
            zf.extractall(src_dir)
        ckpt_path = os.path.join(td, "ckpt.b64"); Path(ckpt_path).write_text(ckpt_b64)
        out_path = os.path.join(td, "aot.pkl")
        driver = os.path.join(td, "aot_driver.py"); Path(driver).write_text(_AOT_DRIVER)
        # The serialized executable is locked to the BUILD jaxlib. Kaggle's runner
        # (kaggle-images/python:v163) is jax 0.5.2 / jaxlib 0.5.1 — NOT this host's
        # version — so set AOT_PYTHON to a venv with jaxlib==0.5.1 or the blob is
        # rejected at runtime and the agent falls back to compiling (the timeout).
        aot_python = os.environ.get("AOT_PYTHON")
        if not aot_python:
            _cand = REPO / ".venv_aot051" / "bin" / "python"   # see tools/make_aot_venv.sh
            aot_python = str(_cand) if _cand.exists() else sys.executable
        print(f"  aot-encode: building with {aot_python}")
        if aot_python == sys.executable:
            print("  aot-encode: WARNING building with THIS host's jaxlib, not Kaggle's "
                  "0.5.1 — the blob will be rejected on Kaggle and fall back to compiling. "
                  "Run tools/make_aot_venv.sh (or set AOT_PYTHON).")
        try:
            r = subprocess.run(
                [aot_python, driver, src_dir, out_path, ckpt_path],
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
                capture_output=True, text=True, timeout=600,
            )
        except Exception as e:
            print(f"  aot-encode: subprocess failed ({e}); shipping without AOT")
            return ""
        if r.returncode != 0 or "AOT_OK" not in r.stdout or not os.path.exists(out_path):
            print("  aot-encode: driver did not finish cleanly; shipping without AOT")
            print((r.stderr or "")[-800:])
            return ""
        blob = Path(out_path).read_bytes()
        print(f"  aot-encode: bundled precompiled encoder ({len(blob)/1e6:.1f} MB)")
        return base64.b64encode(blob).decode()


TEMPLATE = '''"""Self-contained Orbit Wars PPO agent — Kaggle submission.

Source checkpoint: {ckpt_rel}
Generated by tools-side gen_submission.py.

How the timeout is handled
--------------------------
kaggle_environments enforces actTimeout=1s per step. Cold JAX/XLA compilation of
the feature encoder + policy forward pass costs several seconds, so it MUST NOT
happen inside a timed agent() call. This file forces ALL compilation to occur at
module-IMPORT time:
  * build_agent() JIT-compiles + warms up the encoder and policy on a dummy state;
  * then we replay several REAL captured observations (empty initial board +
    populated boards with/without fleets) so every distinct shape/dtype path
    through obs_to_env_state is compiled now, not in-game.
On Kaggle the agent module is imported once at process startup (before the timed
turns), so this import-time work is untimed and every in-game turn then runs in
~65 ms (verified: 499-step episode, mean 66 ms, max 107 ms, p95 77 ms, no
recompiles, no TIMEOUT).

NOTE: a LOCAL `env.run([this_file_path, ...])` is misleading — the local harness
lazy-exec's the file *inside* the first timed act(), so it will appear to time
out locally. To reproduce Kaggle's behaviour locally, import this module first
and pass the warmed `agent` callable: `env.run([mod.agent, opponent])`.
"""
from __future__ import annotations

import base64
import importlib
import io
import os
import pickle
import sys
import tempfile
import zipfile

# --- unpack bundled src_jax modules -----------------------------------------
_SRC_DIR = os.path.join(tempfile.gettempdir(), "owars_ppo_src_" + _SRC_TAG)
if not os.path.isdir(os.path.join(_SRC_DIR, "src_jax")):
    # Concurrent imports (e.g. 32 eval_mp workers) race inside extractall —
    # zipfile's makedirs has no exist_ok. Extract to a per-process dir and
    # atomically rename into place; losers discard their copy and use the
    # winner's (rename only fails once a complete tree is already there).
    _tmp_dir = _SRC_DIR + "." + str(os.getpid())
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_SRC_ZIP_B64))) as _zf:
        _zf.extractall(_tmp_dir)
    try:
        os.rename(_tmp_dir, _SRC_DIR)
    except OSError:
        import shutil
        shutil.rmtree(_tmp_dir, ignore_errors=True)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- quiet backend probes (+ optional XLA compile cache, currently disabled) -
# The rocm/tpu backend-probe failures are harmless (jax falls back to CPU); mute
# them so the episode log isn't noisy. The `if _CACHE_ZIP_B64:` block below is
# inert: the generator ships an empty _CACHE_ZIP_B64 because JAX's persistent
# compilation-cache key proved non-deterministic across processes on jax 0.5.3
# CPU (a bundled cache never reliably hits and only adds first-act overhead — see
# build_compile_cache_b64). The real first-act compile win is removing the
# comet-evac JIT in jax_agent.py. The block is kept (env vars set before any jax
# import) so re-enabling is a one-line generator change if a future jaxlib makes
# keys reproducible.
import logging as _logging
_logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)  # silence rocm/tpu probe spam
try:
    if _CACHE_ZIP_B64:
        _CACHE_DIR = os.path.join(tempfile.gettempdir(), "owars_xla_cache_" + _SRC_TAG)
        if not os.path.isdir(_CACHE_DIR):
            _ctmp = _CACHE_DIR + "." + str(os.getpid())
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(_CACHE_ZIP_B64))) as _czf:
                _czf.extractall(_ctmp)
            try:
                os.rename(_ctmp, _CACHE_DIR)
            except OSError:
                import shutil as _shutil
                _shutil.rmtree(_ctmp, ignore_errors=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
        # Path-independent cache key — lets the bundled cache hit even though
        # Kaggle extracts src to a different tmp dir than this build host.
        os.environ["JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY"] = "0"
except Exception:
    import traceback as _tb
    _tb.print_exc(file=sys.__stderr__)

# --- build the agent (compiles + warms up JITs at IMPORT time) --------------
_build_agent = importlib.import_module("src_jax.jax_agent").build_agent

import zlib as _zlib
import numpy as _np
_ckpt = pickle.loads(_zlib.decompress(base64.b64decode(_CKPT_B64)))
# Params were serialized as version-independent ("__ARR__", shape, dtype, bytes)
# records (no jax/numpy array pickle deps). Reconstruct via np.frombuffer and
# upcast float16 → float32 for inference.
def _dearrayify(tree):
    if isinstance(tree, tuple) and len(tree) == 4 and tree[0] == "__ARR__":
        _, shape, dtypestr, buf = tree
        a = _np.frombuffer(buf, dtype=_np.dtype(dtypestr)).reshape(shape)
        return a.astype(_np.float32) if a.dtype == _np.float16 else a
    if isinstance(tree, dict):
        return {k: _dearrayify(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_dearrayify(v) for v in tree)
    return tree
# Precompiled (AOT) encoder executable, if bundled — build_agent loads it instead
# of tracing+compiling the encoder (the ~5s cost that, x4 simultaneous instances
# in Kaggle FFA games, blows the first-act budget). Falls back to normal compile
# if absent or if the runtime jaxlib rejects the serialized executable.
_aot_encode = None
if _AOT_ENCODE_B64:
    try:
        _aot_encode = pickle.loads(base64.b64decode(_AOT_ENCODE_B64))
    except Exception:
        import traceback as _tb
        _tb.print_exc(file=sys.__stderr__)
_AGENT = _build_agent(_dearrayify(_ckpt["params"]), _ckpt["config"],
                      deterministic=True, aot_encode=_aot_encode)

# build_agent() compiles the jitted functions (encode_fn / _forward) on a
# synthetic dummy state whose shapes match the real obs path, so most real-obs
# work reuses those compilations. The small post-encoder ops (argmax/where/...)
# still compile on the FIRST real obs (~1s) since the dummy warmup doesn't cover
# them, so replay the captured obs ONCE here to force that compile at import
# rather than deferring it into a timed in-game act. The redundant second pass
# was dropped to shave first-act overhead.
try:
    _warm = pickle.loads(base64.b64decode(_WARM_B64))
    for _ob in _warm.values():
        _AGENT(_ob)
    del _warm
except Exception:
    # Warm-up is best-effort; never let it break agent construction, but do
    # surface the reason to real stderr (kaggle redirects sys.stdout/stderr
    # during exec, so go to the original fds) for debugging.
    import traceback as _tb
    _tb.print_exc(file=sys.__stderr__)


def agent(obs):
    """Kaggle entry point. Returns a list of [src_id, angle, ships] moves."""
    return _AGENT(obs)
'''


def main() -> None:
    import argparse
    import hashlib
    global CKPT, OUT, SRC_REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=CKPT,
                        help="Path to the source PPO .pkl checkpoint.")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Path to the generated single-file submission .py.")
    parser.add_argument("--src-repo", type=Path, default=SRC_REPO,
                        help="Repo/worktree to bundle src_jax from; must match "
                             "the code the checkpoint was trained with.")
    parser.add_argument("--max-ticks-in", type=int, default=None,
                        help="Override MAX_TICKS_IN in bundled feature files "
                             "(e.g. 30 for a faster inference encoder).")
    a = parser.parse_args()
    CKPT, OUT, SRC_REPO = a.ckpt, a.out, a.src_repo
    global MAX_TICKS_IN
    MAX_TICKS_IN = a.max_ticks_in

    zip_b64 = build_zip_b64()
    ckpt_b64 = build_ckpt_b64()
    warm_b64 = build_warm_b64()
    tag = hashlib.sha1(zip_b64.encode()).hexdigest()[:12]
    cache_b64 = build_compile_cache_b64(zip_b64, ckpt_b64, warm_b64, tag)
    aot_b64 = build_aot_encode_b64(zip_b64, ckpt_b64)
    header = (
        f'_SRC_TAG = "{tag}"\n\n'
        f'_SRC_ZIP_B64 = "{zip_b64}"\n\n'
        f'_CKPT_B64 = "{ckpt_b64}"\n\n'
        f'_WARM_B64 = "{warm_b64}"\n\n'
        f'_CACHE_ZIP_B64 = "{cache_b64}"\n\n'
        f'_AOT_ENCODE_B64 = "{aot_b64}"\n\n'
    )
    try:
        ckpt_rel = str(CKPT.resolve().relative_to(REPO))
    except ValueError:
        ckpt_rel = str(CKPT)
    OUT.write_text(TEMPLATE.replace("{ckpt_rel}", ckpt_rel, 1).replace(
        "import zipfile\n", "import zipfile\n\n" + header, 1
    ))
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size/1e6:.1f} MB)")
    print(f"  zip_b64 chars : {len(zip_b64):,}")
    print(f"  ckpt_b64 chars: {len(ckpt_b64):,}")


if __name__ == "__main__":
    main()
