"""
Game constants for the Orbit Wars environment.

These replace the corresponding values imported from kaggle_environments so the
rest of the codebase (helper.py, features.py, agents) can work without the
kaggle package at training time.
"""

from __future__ import annotations

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
MIN_ANGULAR_VELOCITY: float = 0.025
MAX_ANGULAR_VELOCITY: float = 0.05

