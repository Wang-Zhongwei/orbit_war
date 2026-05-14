"""game_context.py — GameContext: per-turn game state + cross-turn memory + physics queries."""
from __future__ import annotations

from math import sqrt

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT

from helper import (
    DirectedFleet,
    MAX_TICKS,
    find_angle,
    my_planets_under_attack,
    planet_trajectory,
    reinforcement_plans,
    static_angle,
    unpack_obs,
)


class GameContext:
    """Per-turn game state with cross-turn memory and physics query methods.

    Per-turn state is refreshed by :meth:`update`. Cross-turn state
    (trajectory cache, fleet tracker, moving-planet set) persists until :meth:`reset`.

    Typical stateful usage::

        ctx = GameContext(max_tick=cfg.max_tick)   # once at construction
        # in __call__:
        if step == 0: ctx.reset()
        ctx.update(obs)
        # in _run(ctx):
        angle, tick = ctx.aim(src, tgt, ships)
        ctx.record_fleet(src, tgt, angle, ships, tick)

    Stateless (per-turn only)::

        ctx = GameContext.from_obs(obs)
    """

    # ── Construction / lifecycle ──────────────────────────────────────────────

    def __init__(self, max_tick: int = MAX_TICKS) -> None:
        self.max_tick = max_tick

        # Cross-turn state
        self._moving_ids: set[int]            = set()
        self._traj_cache: dict[int, list]     = {}
        self._my_fleets:  list[DirectedFleet] = []

        # Per-turn state (populated by update)
        self.player:           int   = 0
        self.planets:          list  = []
        self.my_planets:       list  = []
        self.enemies:          list  = []
        self.fleets:           list  = []
        self.comet_ids:        set   = set()
        self.comet_paths:      dict  = {}
        self.angular_velocity: float = 0.0
        self.step:             int   = 0

        self._under_attack_cache: dict | None = None

    def reset(self) -> None:
        """Clear cross-turn state at game start (call when obs.step == 0)."""
        self._moving_ids.clear()
        self._traj_cache.clear()
        self._my_fleets.clear()

    def update(self, obs) -> None:
        """Refresh per-turn state and advance cross-turn bookkeeping."""
        player, planets, comet_ids, angular_velocity, fleets, comet_paths = unpack_obs(obs)

        self.player           = player
        self.planets          = planets
        self.my_planets       = [p for p in planets if p.owner == player]
        self.enemies          = [p for p in planets if p.owner != player]
        self.fleets           = fleets
        self.comet_ids        = comet_ids
        self.comet_paths      = comet_paths
        self.angular_velocity = angular_velocity
        self.step             = obs.get("step", 0)

        # Detect moving planets once; _moving_ids is empty only after reset().
        if not self._moving_ids:
            for p in planets:
                if sqrt((p.x - CENTER) ** 2 + (p.y - CENTER) ** 2) + p.radius < ROTATION_RADIUS_LIMIT:
                    self._moving_ids.add(p.id)

        # Rebuild trajectory cache for non-comet moving planets.
        self._traj_cache = {
            p.id: planet_trajectory(p, self.max_tick, angular_velocity)
            for p in planets
            if p.id in self._moving_ids and p.id not in comet_ids
        }

        # Age tracked fleets; evict those that have arrived.
        self._my_fleets = [
            DirectedFleet(f.src_id, f.tgt_id, f.angle, f.ships, f.arrive_tick - 1)
            for f in self._my_fleets
            if f.arrive_tick > 1
        ]

        self._under_attack_cache = None

    @classmethod
    def from_obs(cls, obs) -> GameContext:
        """Construct a fresh per-turn context with no cross-turn memory."""
        ctx = cls()
        ctx.update(obs)
        return ctx

    # ── Cross-turn fleet tracking ─────────────────────────────────────────────

    def record_fleet(self, src, tgt, angle: float, ships: int, arrive_tick: int) -> None:
        """Record an outgoing fleet for cross-turn accounting."""
        self._my_fleets.append(DirectedFleet(src.id, tgt.id, angle, ships, arrive_tick))

    @property
    def my_fleets(self) -> list[DirectedFleet]:
        return self._my_fleets

    # ── Threat detection ──────────────────────────────────────────────────────

    @property
    def under_attack(self) -> dict[int, list[DirectedFleet]]:
        """Enemy fleets targeting my planets this turn (cached for the duration of the turn)."""
        if self._under_attack_cache is None:
            self._under_attack_cache = my_planets_under_attack(
                self.my_planets, self.fleets, self.player, self._traj_cache, self.max_tick
            )
        return self._under_attack_cache

    def avail(self, planet) -> int:
        """Effective ships on `planet` after subtracting confirmed incoming enemy fleets."""
        ships = planet.ships
        ua = self.under_attack
        if planet.id in ua:
            ships -= sum(a.ships for a in ua[planet.id])
        return max(0, ships)

    def reinforcements_needed(self, min_ships: int) -> dict:
        """Planets needing defense and required ship counts, accounting for already-planned fleets."""
        return reinforcement_plans(self.my_planets, self.under_attack, self._my_fleets, min_ships)

    # ── Aiming ───────────────────────────────────────────────────────────────

    def aim(self, src, tgt, ships: int) -> tuple[float | None, int | None]:
        """Return (angle, arrive_tick) toward tgt, using trajectory cache for moving targets."""
        if tgt.id in self.comet_ids:
            traj = self.comet_paths.get(tgt.id, [])
        elif tgt.id in self._moving_ids:
            traj = self._traj_cache.get(tgt.id, [])
        else:
            traj = []

        if traj:
            return find_angle(
                src, traj, ships,
                blockers=self.planets, traj_cache=self._traj_cache,
                tgt_id=tgt.id, tgt_radius=tgt.radius,
            )
        return static_angle(src, tgt, ships, blockers=self.planets, traj_cache=self._traj_cache)

    def trajectory(self, planet_id: int) -> list | None:
        """Precomputed trajectory for a moving planet, or None for static planets."""
        return self._traj_cache.get(planet_id)

    def ships_to_capture(self, src, tgt, needed: int) -> tuple[float | None, int | None, int | None]:
        """Refine ship count to account for enemy production during travel.

        Returns (angle, ships, arrive_tick), or (None, None, None) if unreachable.
        """
        original_needed = needed
        last_needed = needed
        angle: float | None = None
        arrive_tick: int | None = None
        for _ in range(5):
            angle, arrive_tick = self.aim(src, tgt, needed)
            if angle is None:
                return None, None, None
            needed = original_needed + (arrive_tick * tgt.production if tgt.owner != -1 else 0)
            if last_needed == needed:
                break
            last_needed = needed
        return angle, needed, arrive_tick
