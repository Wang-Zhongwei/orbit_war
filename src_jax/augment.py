"""
D4 spatial data augmentation for BC pretraining.

Orbit Wars has 8-fold dihedral (D4) symmetry: any rotation by 0°/90°/180°/270° or
reflection about the 4 board axes leaves orbital mechanics, ownership relationships,
and candidate-slot assignments structurally identical.

aug_id → transform on centred coords (cx, cy) = (x − 0.5, y − 0.5):
    0 — identity:            ( cx,  cy)
    1 — rotate 90° CCW:      (-cy,  cx)
    2 — rotate 180°:         (-cx, -cy)
    3 — rotate 270° CCW:     ( cy, -cx)
    4 — flip-X (vert axis):  (-cx,  cy)  ← reflection: reverses angular_velocity
    5 — flip-Y (horiz axis): ( cx, -cy)  ← reflection: reverses angular_velocity
    6 — flip diagonal:       ( cy,  cx)  ← reflection: reverses angular_velocity
    7 — flip anti-diagonal:  (-cy, -cx)  ← reflection: reverses angular_velocity

Features modified per aug_id (all others are invariant):
  self_feats   dims [1, 2]   — absolute x / y position (normalised 0–1)
  cand_feats   dims [4, 5]   — Δx / Δy offset to candidate
  global_feat  dim  [11]     — angular_velocity (negated for aug_id ≥ 4)
  fleet_feats  dims [2, 3]   — fleet x / y position
                dims [4, 5]  — fleet velocity (vx, vy)

Augmentation does NOT change:
  • candidate ordering / target_idx labels
  • ship_bucket labels
  • candidate_mask
  • any scalar (bias, ship counts, production, ownership flags, distances,
    orbit radii, inbound-fleet sums, game-time, …)

Usage (BC training):
    from src_jax.augment import expand_with_augmentations

    sf, cf, gf, cm, ti, sb = expand_with_augmentations(sf, cf, gf, cm, ti, sb)
    # now N is 4× the original for 4-rotation augmentation

Reference: Team Durrett (4th, Lux AI 2021) used 4-rotation augmentation in BC,
which was one of the key factors behind their strong imitation-learning baseline.
"""
from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Primitive coordinate / vector transforms
# ---------------------------------------------------------------------------

def _rot_xy(
    x: jnp.ndarray,
    y: jnp.ndarray,
    aug_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate/reflect normalised [0, 1] absolute position around (0.5, 0.5)."""
    cx, cy = x - 0.5, y - 0.5
    if aug_id == 0:
        ncx, ncy =  cx,  cy
    elif aug_id == 1:   # rotate 90° CCW
        ncx, ncy = -cy,  cx
    elif aug_id == 2:   # rotate 180°
        ncx, ncy = -cx, -cy
    elif aug_id == 3:   # rotate 270° CCW (= 90° CW)
        ncx, ncy =  cy, -cx
    elif aug_id == 4:   # flip-X — mirror about vertical centre line
        ncx, ncy = -cx,  cy
    elif aug_id == 5:   # flip-Y — mirror about horizontal centre line
        ncx, ncy =  cx, -cy
    elif aug_id == 6:   # flip main diagonal (y = x)
        ncx, ncy =  cy,  cx
    elif aug_id == 7:   # flip anti-diagonal (y = −x)
        ncx, ncy = -cy, -cx
    else:
        raise ValueError(f"aug_id must be 0–7, got {aug_id!r}")
    return ncx + 0.5, ncy + 0.5


def _rot_vec(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    aug_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the same linear transform to a 2-D vector (delta or velocity)."""
    if aug_id == 0:
        return  vx,  vy
    elif aug_id == 1:
        return -vy,  vx
    elif aug_id == 2:
        return -vx, -vy
    elif aug_id == 3:
        return  vy, -vx
    elif aug_id == 4:
        return -vx,  vy
    elif aug_id == 5:
        return  vx, -vy
    elif aug_id == 6:
        return  vy,  vx
    elif aug_id == 7:
        return -vy, -vx
    else:
        raise ValueError(f"aug_id must be 0–7, got {aug_id!r}")


# ---------------------------------------------------------------------------
# Batch-level augmentation
# ---------------------------------------------------------------------------

def apply_augmentation(
    sf: jnp.ndarray,                    # [N, SELF_DIM=11]
    cf: jnp.ndarray,                    # [N, K, CAND_DIM=14]
    gf: jnp.ndarray,                    # [N, GLOB_DIM=12]
    aug_id: int,                        # 0–7
    # ff: jnp.ndarray | None = None,      # [N, MAX_FLEETS, 6]  optional
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Apply one D4 transform to a batch of encoded feature arrays.

    aug_id = 0 is a no-op (identity); all others modify at least two dims.
    JAX arrays are immutable so `.at[].set()` returns new arrays — the inputs
    are never modified in-place.

    Returns (sf', cf', gf', ff') — shapes unchanged.
    """
    if aug_id == 0:
        return sf, cf, gf

    # ── self_feats: absolute position at dims [1, 2] ──────────────────────
    sx, sy = _rot_xy(sf[..., 1], sf[..., 2], aug_id)
    sf = sf.at[..., 1].set(sx).at[..., 2].set(sy)

    # ── cand_feats: delta vector at dims [4, 5] ────────────────────────────
    dx, dy = _rot_vec(cf[..., 4], cf[..., 5], aug_id)
    cf = cf.at[..., 4].set(dx).at[..., 5].set(dy)

    # ── global_feat: angular_velocity at dim [11] ─────────────────────────
    # Reflections reverse the orbital handedness (CCW ↔ CW).
    if aug_id >= 4:
        gf = gf.at[..., 11].set(-gf[..., 11])

    # ── fleet_feats: position [2, 3] and velocity [4, 5] ─────────────────
    # if ff is not None:
    #     fx, fy = _rot_xy(ff[..., 2], ff[..., 3], aug_id)
    #     vx, vy = _rot_vec(ff[..., 4], ff[..., 5], aug_id)
    #     ff = (
    #         ff.at[..., 2].set(fx)
    #           .at[..., 3].set(fy)
    #           .at[..., 4].set(vx)
    #           .at[..., 5].set(vy)
    #     )

    return sf, cf, gf


# ---------------------------------------------------------------------------
# Dataset expansion
# ---------------------------------------------------------------------------
DEFAULT_AUG_IDS: list[int] = list(range(8))


def expand_with_augmentations(
    sf: jnp.ndarray,               # [N, SELF_DIM]
    cf: jnp.ndarray,               # [N, K, CAND_DIM]
    gf: jnp.ndarray,               # [N, GLOB_DIM]
    cm: jnp.ndarray,               # [N, K]   — unchanged
    ti: jnp.ndarray,               # [N]      — unchanged
    sb: jnp.ndarray,               # [N]      — unchanged
    aug_ids: list[int] | None = None,
    ff: jnp.ndarray | None = None, # [N, MAX_FLEETS, 6]  optional
    fm: jnp.ndarray | None = None, # [N, MAX_FLEETS]      optional
) -> tuple:
    """Expand a batch by stacking K augmented copies along the batch axis.

    With the default 4 rotations the batch grows from N to 4N samples.

    Returns:
        (sf', cf', gf', cm', ti', sb')                 — if ff is None
        (sf', cf', gf', cm', ti', sb', ff', fm')       — if ff and fm are given
    """
    if aug_ids is None:
        aug_ids = DEFAULT_AUG_IDS

    sf_list: list[jnp.ndarray] = []
    cf_list: list[jnp.ndarray] = []
    gf_list: list[jnp.ndarray] = []
    ff_list: list[jnp.ndarray] = [] if ff is not None else None  # type: ignore[assignment]

    for a in aug_ids:
        sf_a, cf_a, gf_a = apply_augmentation(sf, cf, gf, a)
        sf_list.append(sf_a)
        cf_list.append(cf_a)
        gf_list.append(gf_a)
        # if ff_list is not None:
        #     ff_list.append(ff_a)

    n = len(aug_ids)
    sf_out = jnp.concatenate(sf_list, axis=0)   # [N*n, SELF_DIM]
    cf_out = jnp.concatenate(cf_list, axis=0)   # [N*n, K, CAND_DIM]
    gf_out = jnp.concatenate(gf_list, axis=0)   # [N*n, GLOB_DIM]
    cm_out = jnp.concatenate([cm] * n, axis=0)  # [N*n, K]
    ti_out = jnp.concatenate([ti] * n, axis=0)  # [N*n]
    sb_out = jnp.concatenate([sb] * n, axis=0)  # [N*n]

    if ff is not None and fm is not None:
        ff_out = jnp.concatenate(ff_list, axis=0)   # type: ignore[arg-type]
        fm_out = jnp.concatenate([fm] * n, axis=0)
        return sf_out, cf_out, gf_out, cm_out, ti_out, sb_out, ff_out, fm_out

    return sf_out, cf_out, gf_out, cm_out, ti_out, sb_out
