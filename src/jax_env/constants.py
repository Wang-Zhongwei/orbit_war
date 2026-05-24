"""
Game constants and pure math utilities for the Orbit Wars environment.

These replace the corresponding values imported from kaggle_environments so the
rest of the codebase (helper.py, features.py, agents) can work without the
kaggle package at training time.
"""

from __future__ import annotations

import math
from collections import namedtuple

# ---------------------------------------------------------------------------
# Board / physics constants (match kaggle orbit_wars.py verbatim)
# ---------------------------------------------------------------------------

BOARD_SIZE: float = 100.0
CENTER: float = BOARD_SIZE / 2.0          # 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0       # planets with orbital_r + r < 50 rotate

COMET_RADIUS: float = 1.0
COMET_PRODUCTION: int = 1
COMET_SPAWN_STEPS: list[int] = [50, 150, 250, 350, 450]

PLANET_CLEARANCE: int = 7
MIN_PLANET_GROUPS: int = 5
MAX_PLANET_GROUPS: int = 10
MIN_STATIC_GROUPS: int = 3

MAX_SHIP_SPEED: float = 6.0
EPISODE_STEPS: int = 500
COMET_SPEED: float = 4.0

# ---------------------------------------------------------------------------
# Named tuples (same field layout as kaggle — rule-based agents use these)
# ---------------------------------------------------------------------------

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(
    p: tuple[float, float],
    v: tuple[float, float],
    w: tuple[float, float],
) -> float:
    """Minimum distance from point *p* to line segment *v*–*w*."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0,
        min(
            1,
            ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2,
        ),
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


def swept_pair_hit(
    A: tuple[float, float],
    B: tuple[float, float],
    P0: tuple[float, float],
    P1: tuple[float, float],
    r: float,
) -> bool:
    """True iff fleet segment A→B and planet segment P0→P1 come within *r*.

    Both objects are treated as moving linearly over one tick (planet rotation
    is linearised to its chord), matching the kaggle reference exactly.
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0
