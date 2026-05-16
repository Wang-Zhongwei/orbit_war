"""base.py — BaseStatefulAgent: lifecycle wrapper that owns a GameContext."""
from __future__ import annotations

from abc import ABC, abstractmethod

from game_context import GameContext
from helper import MAX_TICKS


class BaseStatefulAgent(ABC):
    """Lifecycle wrapper: reset on game start, startup delay, then delegate to _run(ctx).

    All state and physics queries are available through the :class:`GameContext`
    passed to :meth:`_run`. Subclasses only need to implement strategy.
    """

    def __init__(self, max_tick: int = MAX_TICKS, path_blocking_detection: bool = True) -> None:
        self._ctx = GameContext(max_tick=max_tick, path_blocking_detection=path_blocking_detection)

    def __call__(self, obs) -> list:
        step = obs.get("step", 0)
        if step == 0:
            self._ctx.reset()
        self._ctx.update(obs)
        if step < 3:
            return []
        return self._run(self._ctx)

    @abstractmethod
    def _run(self, ctx: GameContext) -> list:
        """Execute one turn of agent strategy. Called after the startup delay."""
        ...

    @abstractmethod
    def target_score(self, src, tgt) -> float:
        """Score a target for an attack from a source planet."""
        ...
