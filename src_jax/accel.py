"""
Accelerator setup shared by all JAX entry points (CPU / GPU / TPU).

The compute path is already accelerator-agnostic — everything is float32 with
static (padding-based) shapes, so it compiles unchanged on TPU.  The one TPU
gotcha that bites silently is matmul precision:

    On TPU, ``jnp.dot`` / ``jnp.matmul`` default to a single-pass *bfloat16*
    multiply even when the inputs are float32.  That quietly halves the
    mantissa relative to GPU/CPU (which multiply in full float32), which can
    destabilize PPO value/advantage estimates and make a run that was healthy
    on GPU diverge on TPU.

``setup_accelerator`` sets the default matmul precision: TF32 on GPU (matches
PyTorch default; unlocks Tensor Cores on Ampere/Ada), full float32 on TPU/CPU
(guards against TPU's silent bfloat16 downgrade).

Call it once, before any rollout / training function is compiled.
"""
from __future__ import annotations


def setup_accelerator(precision: str | None = None, *, verbose: bool = True) -> str:
    """Configure JAX for the active backend and return the platform name.

    ``precision`` maps to ``jax_default_matmul_precision``:
      - ``"highest"`` / ``"float32"`` → full float32 matmuls (GPU/CPU parity).
      - ``"high"``    / ``"tensorfloat32"`` → TF32 Tensor Cores on Ampere/Ada GPUs
                                              (matches PyTorch default; safe for RL).
      - ``"default"`` / ``"bfloat16"`` → 1-pass bf16 (fastest, lowest accuracy).

    When ``precision`` is None (default), GPU uses ``"high"`` to enable TF32 Tensor
    Cores (standard practice; PyTorch does this automatically on A100/RTX).  TPU and
    CPU use ``"highest"`` because TPU silently degrades matmuls to bfloat16 at
    ``"default"``/``"high"``, which destabilises PPO value estimates.

    x64 is deliberately left disabled: the compute path is float32 throughout,
    and float64 is unsupported on TPU.
    """
    import jax

    platform = jax.default_backend()
    if precision is None:
        precision = "high" if platform == "gpu" else "highest"

    jax.config.update("jax_default_matmul_precision", precision)

    if verbose:
        devices = jax.devices()
        print(
            f"JAX backend: {platform} "
            f"({len(devices)} device(s); matmul precision={precision})",
            flush=True,
        )
    return platform
