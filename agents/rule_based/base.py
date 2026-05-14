"""base_agent.py — BaseStatefulAgent: shared lifecycle and state for stateful agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT, Planet

from helper import DirectedFleet, find_angle, static_angle

class BaseStatefulAgent(ABC):
    """Common state management and lifecycle shared by all stateful Orbit Wars agents.

    Subclasses must implement :meth:`_run` with their specific strategy.
    They may extend :meth:`_reset_state` to clear additional state.
    """

    def __init__(self) -> None:
        self._steps:      int  = 0
        self._moving_ids: set  = set()   # planet IDs confirmed to be in motion
        self._traj_cache: dict = {}      # planet_id -> [(x,y), …]
        self._comet_ids:  set  = set()   # planet IDs that are comets this turn
        self._comet_paths: dict = {}     # planet_id -> remaining waypoints
        self._my_fleets: list[DirectedFleet] = []      # my fleets en route
        self._planets:    list = []      # all planets this turn (for blocking checks)
        self._angular_velocity: float = 0.0
    # ── Public entry point ────────────────────────────────────────────────────

    def __call__(self, obs) -> list:
        if obs.get("step", 0) == 0:
            self._reset_state()
            self._detect_moving_planets(obs)

        self._planets = [Planet(*p) for p in obs.get("planets", [])]

        if self._steps < 3:
            self._steps += 1
            return []

        return self._run(obs)
    

    # ── State lifecycle ───────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        """Clear all cross-game state. Subclasses that hold extra state must extend this."""
        self._steps = 0
        self._moving_ids.clear()
        self._traj_cache.clear()
        self._comet_ids = set()
        self._comet_paths = {}
        self._my_fleets.clear()
        self._planets = []

    def _detect_moving_planets(self, obs) -> None:
        """Populate _moving_ids with planet IDs whose position differs from initial."""
        planets = [Planet(*p) for p in obs.get("planets", [])]
        for p in planets:
            if sqrt((p.x - CENTER) ** 2 + (p.y - CENTER) ** 2) + p.radius < ROTATION_RADIUS_LIMIT:
                self._moving_ids.add(p.id)

    # ── Trajectory management ─────────────────────────────────────────────────
    def _age_my_fleets(self) -> None:
        """Decrement arrive_tick; evict fleets that arrive this turn."""
        self._my_fleets = [
            DirectedFleet(f.src_id, f.tgt_id, f.angle, f.ships, f.arrive_tick - 1)
            for f in self._my_fleets
            if f.arrive_tick > 1
        ]


    def _add_my_fleet(self, src, tgt, angle: float, ships: int, arrive_tick: int) -> None:
        """Record an outgoing attack fleet for cross-turn accounting."""
        self._my_fleets.append(DirectedFleet(src.id, tgt.id, angle, ships, arrive_tick))

    # ── Shared combat helpers ────────────────────────────────────────────────
    def _avail(self, planet, under_attack: dict[int, list[DirectedFleet]]) -> int:
        """Effective ships on `planet` after subtracting confirmed incoming enemy fleets."""
        avail = planet.ships
        if planet.id in under_attack:
            avail -= sum(a.ships for a in under_attack[planet.id])
        return max(0, avail)

    def aim(self, src, tgt, ships: int) -> tuple[float | None, int | None]:
        """Return (angle, arrive_tick) toward tgt, routing through traj cache for moving targets."""
        # Comets take priority: _traj_cache excludes comet IDs, so check first.
        if tgt.id in self._comet_ids:
            traj = self._comet_paths.get(tgt.id, [])
        elif tgt.id in self._moving_ids:
            traj = self._traj_cache.get(tgt.id, [])
        else:
            traj = []

        if traj:
            return find_angle(src, traj, ships,
                              blockers=self._planets, traj_cache=self._traj_cache,
                              tgt_id=tgt.id, tgt_radius=tgt.radius)
        return static_angle(src, tgt, ships,
                            blockers=self._planets, traj_cache=self._traj_cache)

    def _refine_offense_ships_needed(self, src, tgt, needed: int) -> tuple[float | None, int | None, int | None]:
        """Iteratively refine ship count to account for enemy production during travel.

        Returns (angle, ships, arrive_tick) or (None, None, None) if unreachable.
        """
        angle, arrive_tick = None, None
        original_needed = needed
        last_needed = needed
        for _ in range(5):
            angle, arrive_tick = self.aim(src, tgt, needed)
            if angle is None:
                return None, None, None
            needed = original_needed + (arrive_tick * tgt.production if tgt.owner != -1 else 0)
            if last_needed == needed:
                break
            last_needed = needed
        return angle, needed, arrive_tick

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _run(self, obs) -> list:
        """Execute one turn of the agent strategy. Called after startup is complete."""
        ...
    
    @abstractmethod
    def target_score(self, src, tgt) -> float:
        """Score a target for an attack from a source planet."""
        ...
