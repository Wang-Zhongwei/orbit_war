"""
jax_env — self-contained Orbit Wars environment (no kaggle_environments dependency).

Quick import:
    from src.jax_env import OrbitWarsEnv, StepResult
    from src.jax_env import init_game, step_game
    from src.jax_env.constants import CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT, Planet, Fleet
"""

from .constants import (  # noqa: F401
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    EPISODE_STEPS,
    Fleet,
    MAX_SHIP_SPEED,
    Planet,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    distance,
    point_to_segment_distance,
    swept_pair_hit,
)
from .physics import init_game, step_game  # noqa: F401
from .state import GameState, make_observation  # noqa: F401
# OrbitWarsEnv/StepResult are NOT re-exported here to avoid a circular import
# (env.py → features → helper → jax_env.__init__ → env).
# Import directly: from src.jax_env.env import OrbitWarsEnv, StepResult
