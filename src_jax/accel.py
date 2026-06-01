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

``setup_accelerator`` pins the default matmul precision to full float32 so the
numerics match across backends.  The model here is tiny (hidden_size≈128), so
the extra passes cost essentially nothing while buying GPU/TPU parity.

Call it once, before any rollout / training function is compiled.
"""
from __future__ import annotations


def setup_accelerator(precision: str = "highest", *, verbose: bool = True) -> str:
    """Configure JAX for the active backend and return the platform name.

    ``precision`` maps to ``jax_default_matmul_precision``:
      - ``"highest"`` / ``"float32"`` → full float32 matmuls (GPU/CPU parity).
      - ``"high"``    / ``"tensorfloat32"`` → 3-pass bf16 (faster, ~tf32 accuracy).
      - ``"default"`` / ``"bfloat16"`` → 1-pass bf16 (fastest, lowest accuracy).

    x64 is deliberately left disabled: the compute path is float32 throughout,
    and float64 is unsupported on TPU.
    """
    import jax

    jax.config.update("jax_default_matmul_precision", precision)

    platform = jax.default_backend()
    if verbose:
        devices = jax.devices()
        print(
            f"JAX backend: {platform} "
            f"({len(devices)} device(s); matmul precision={precision})",
            flush=True,
        )
    return platform
