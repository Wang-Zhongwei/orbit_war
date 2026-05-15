
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# AgentConfig — all micro-decisions as tunable hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """All micro-decisions for ``run_agent``, exposed as tunable hyperparameters.

    Every field corresponds to one decision point in the shared pipeline and
    maps directly to a ``Param`` entry in ``tune.py``.

    Attributes:
        source_sort:
            Order of source planets entering the per-planet loop.

            ``"none"``          — observation order (fastest).
            ``"richest_first"`` — strongest garrison attacks first; richest planets
                                  claim the highest-value targets before weaker ones.

        target_mode:
            Strategy for picking an attack target for each source planet.

            ``"nearest"``       — closest enemy by Euclidean distance.
            ``"roi"``           — highest production/cost score via ``best_target()``.
            ``"random_biased"`` — distance-sorted pool, top-K kept, biased random draw.

        limit_top_k_fraction:
            ``"random_biased"`` only.  Fraction of the distance-sorted enemy pool to
            keep before drawing.  ``1.0`` = all enemies eligible; ``0.25`` = closest
            quarter only.

        bias_draws:
            ``"random_biased"`` only.  Number of uniform random draws whose minimum
            is taken as the selected index.  ``1`` = uniform; higher = stronger bias
            toward closer planets.

        accept_comet_src:
            When ``True``, a move from a comet-source planet is always issued even
            without ``can_capture``, because comet ships are temporary and should
            be spent aggressively.

        cost_mode:
            How to score candidate moves before capping at ``max_moves``.

            ``"none"``           — loop order, no sort.
            ``"turns_weighted"`` — cost = turns_to_arrive × (cost_weight − can_capture).
                                   Shorter, certain captures rank first.

        cost_weight:
            ``"turns_weighted"`` only.  Scaling factor (default ``2.0`` means a
            certain capture scores half as costly as an uncertain one at equal range).

        max_moves:
            Maximum number of moves issued per turn.  ``None`` = no limit.
    """

    source_sort:          str        = "none"
    target_mode:          str        = "random_biased"
    limit_top_k_fraction: float      = 0.5
    bias_draws:           int        = 2
    accept_comet_src:     bool       = True
    cost_mode:            str        = "turns_weighted"
    cost_weight:          float      = 2.0
    max_moves:            int | None = 6



@dataclass
class StatefulAgentConfig:
    """Tunable hyperparameters for :class:`StatefulAgent`."""

    score_enemy_bonus:     float = 14
    score_ships_penalty:   float = 0.48
    score_time_penalty:    float = 2
    min_ships_attack:      int   = 13
    top_k_targets:         int   = 3
    min_ships_coop_target: int   = 34
    max_tick:              int   = 25
    attack_buffer:         int   = 3