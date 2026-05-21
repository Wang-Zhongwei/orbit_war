"""game_context.py — GameContext: per-turn game state + cross-turn memory + physics queries."""

from __future__ import annotations

from itertools import groupby
import math

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    point_to_segment_distance,
)

from src.helper import (
    DirectedFleet,
    MAX_TICKS,
    path_blocked,
    find_angle,
    fleet_speed,
    hits_sun,
    planet_trajectory,
    reinforcement_plans,
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

    def __init__(
        self, max_tick: int = MAX_TICKS, path_blocking_detection: bool = False
    ) -> None:
        self.max_tick = max_tick
        self.path_blocking_detection = path_blocking_detection

        # Cross-turn state
        self._moving_ids: set[int] = set()
        self._traj_cache: dict[int, list] = {}
        self._my_fleets: list[DirectedFleet] = []

        # Per-turn state (populated by update)
        self.player: int = 0
        self.planets: list = []
        self.my_planets: list = []
        self.targets: list = []
        self.fleets: list = []
        self.comet_ids: set = set()
        self.comet_paths: dict = {}
        self.angular_velocity: float = 0.0
        self.step: int = 0

        self._under_attack_cache: dict | None = None
        self.planet_rank_fraction: dict[int, float] = {}
        self._player_centroids: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        """Clear cross-turn state at game start (call when obs.step == 0)."""
        self._moving_ids.clear()
        self._traj_cache.clear()
        self._my_fleets.clear()

    def update(self, obs) -> None:
        """Refresh per-turn state and advance cross-turn bookkeeping."""
        player, planets, comet_ids, angular_velocity, fleets, comet_paths = unpack_obs(
            obs
        )

        self.player = player
        self.planets = planets
        self.my_planets = [p for p in planets if p.owner == player]
        self.targets = [p for p in planets if p.owner != player]
        self.fleets = fleets
        self.comet_ids = comet_ids
        self.comet_paths = comet_paths
        self.angular_velocity = angular_velocity
        self.step = obs.get("step", 0)

        # Detect moving planets once; _moving_ids is empty only after reset().
        if not self._moving_ids:
            for p in planets:
                if (
                    math.sqrt((p.x - CENTER) ** 2 + (p.y - CENTER) ** 2) + p.radius
                    < ROTATION_RADIUS_LIMIT
                ):
                    self._moving_ids.add(p.id)

        # Rebuild trajectory cache for non-comet moving planets.
        self._traj_cache = {
            p.id: planet_trajectory(p, self.max_tick, angular_velocity)
            for p in planets
            if p.id in self._moving_ids and p.id not in comet_ids
        }
        # Add comets as moving blockers using their precomputed future paths.
        for pid, path in comet_paths.items():
            if path:
                self._traj_cache[pid] = path

        # Sync tracked fleets against the observation; evict those that have arrived.
        self._refresh_my_fleets()

        self._under_attack_cache = None

        # Tally fleet ships, planet ships, and planet count per player.
        self.player_strength: dict[int, dict] = {}
        for pid in range(4):
            p_ships = sum(p.ships for p in planets if p.owner == pid)
            f_ships = sum(f.ships for f in fleets if f.owner == pid)
            self.player_strength[pid] = {
                "planets": sum(1 for p in planets if p.owner == pid),
                "production": sum(p.production for p in planets if p.owner == pid),
                "planet_ships": p_ships,
                "fleet_ships": f_ships,
                "total": p_ships + f_ships,
            }

        # calculate the centroid of each player
        self._player_centroids.clear() # player -> (centroid_x, centroid_y)
        self.planet_rank_fraction.clear() # planet_id -> rank / tot_planets of that player
        for player in range(4):
            player_planets = [p for p in self.planets if p.owner == player]
            # compute weighted average of the centroids
            centroid_x, centroid_y = 0, 0
            tot_ships = 0

            for planet in player_planets:
                centroid_x += planet.x * planet.ships
                centroid_y += planet.y * planet.ships
                tot_ships += planet.ships

            if tot_ships > 0:
                centroid_x /= tot_ships
                centroid_y /= tot_ships
                self._player_centroids[player] = (centroid_x, centroid_y)
            

            if player not in self._player_centroids:
                continue

            # compute the distance from each player's planet to the centroid and rank (0=closest)
            cx, cy = self._player_centroids[player]
            distances_to_centroid = sorted(
                [(math.hypot(planet.x - cx, planet.y - cy), planet.id) for planet in player_planets],
                key=lambda x: x[0],
            )
            n = len(distances_to_centroid)
            for i, (_, pid) in enumerate(distances_to_centroid):
                self.planet_rank_fraction[pid] = i / n


    # todo: what if miss the target?
    def _refresh_my_fleets(self) -> None:
        """Sync _my_fleets against the current observation.

        Matches each tracked fleet to an actual obs fleet by source planet, angle,
        and ship count, then recomputes arrive_tick from the fleet's real position.
        Fleets absent from the observation have arrived (or were destroyed) and are
        evicted. This avoids the off-by-one of blind tick decrement and correctly
        handles mid-flight attrition.
        """
        my_obs_fleets = [f for f in self.fleets if f.owner == self.player]
        planet_by_id = {p.id: p for p in self.planets}

        updated = []
        for df in self._my_fleets:
            match = next(
                (
                    f
                    for f in my_obs_fleets
                    if f.from_planet_id == df.src_id
                    and abs(f.angle - df.angle) < 1e-4
                    and f.ships == df.ships
                ),
                None,
            )
            if match is None:
                continue  # arrived or destroyed

            tgt = planet_by_id.get(df.tgt_id)
            if tgt is None:
                continue

            spd = fleet_speed(match.ships)
            dist = (
                math.sqrt((match.x - tgt.x) ** 2 + (match.y - tgt.y) ** 2) - tgt.radius
            )
            arrive_tick = max(1, math.ceil(max(0.0, dist) / spd))
            updated.append(
                DirectedFleet(df.owner, df.src_id, df.tgt_id, df.angle, df.ships, arrive_tick)
            )

        self._my_fleets = updated

    @classmethod
    def from_obs(cls, obs) -> GameContext:
        """Construct a fresh per-turn context with no cross-turn memory."""
        ctx = cls()
        ctx.update(obs)
        return ctx

    # ── Cross-turn fleet tracking ─────────────────────────────────────────────

    def record_fleet(
        self, src, tgt, angle: float, ships: int, arrive_tick: int
    ) -> None:
        """Record an outgoing fleet for cross-turn accounting."""
        self._my_fleets.append(DirectedFleet(self.player, src.id, tgt.id, angle, ships, arrive_tick))

    @property
    def my_fleets(self) -> list[DirectedFleet]:
        return self._my_fleets

    # ── Threat detection ──────────────────────────────────────────────────────
    @property
    def under_attack(self) -> dict[int, list[DirectedFleet]]:
        """Enemy fleets targeting my planets this turn (cached for the duration of the turn)."""
        if self._under_attack_cache is None:
            self._under_attack_cache = {
                p.id: fleets
                for p in self.my_planets
                if (fleets := self.get_enemy_en_route(p))
            }
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
        return reinforcement_plans(
            self.my_planets, self.under_attack, self._my_fleets, min_ships
        )

    # ── Aiming ───────────────────────────────────────────────────────────────

    def aim(self, src, tgt, ships: int) -> tuple[float | None, int | None, bool]:
        """Return (angle, arrive_tick, is_blocked) toward tgt, using trajectory cache for moving targets."""

        return find_angle(
            src,
            tgt,
            ships,
            traj_cache=self._traj_cache,
            blockers=[p for p in self.planets if p.id not in (src.id, tgt.id)]
            if self.path_blocking_detection
            else [],
        )

    def trajectory(self, planet_id: int) -> list | None:
        """Precomputed trajectory for a moving planet, or None for static planets."""
        return self._traj_cache.get(planet_id)

    
    def is_capture_possible(self, tgt, en_route: list[DirectedFleet]):
        en_route_sorted = sorted(en_route, key=lambda df: df.arrive_tick)
        batches: list[list[DirectedFleet]] = [
            list(g) for _, g in groupby(en_route_sorted, key=lambda df: df.arrive_tick)
        ]

        last_arrive_tick = 0
        last_owner = tgt.owner
        garrison = tgt.ships

        for batch in batches:
            batch_tick = batch[0].arrive_tick
            duration = batch_tick - last_arrive_tick

            # calculate net ships and owner of the batch
            net_ships = batch[0].ships
            batch_owner = batch[0].owner
            for df in batch[1:]:
                if df.owner == batch_owner:
                    net_ships += df.ships
                else:
                    net_ships -= df.ships
                
                # switchover
                if net_ships < 0:
                    net_ships = -net_ships
                    batch_owner = df.owner

            if net_ships == 0:
                continue
            
            # simulate the arrival process
            if last_owner != -1:
                garrison += duration * tgt.production

            if batch_owner == last_owner:
                garrison += net_ships
            else:
                garrison -= net_ships

            if garrison < 0 or garrison == 0 and batch_owner == -1:
                last_owner = batch_owner
                garrison = -garrison

            last_arrive_tick = batch_tick
        
        return last_owner == self.player

    def ships_to_defend(
        self, p, mine_en_route: list[DirectedFleet], enemy_en_route: list[DirectedFleet],
        min_ships: int, helper
    ) -> tuple[float | None, int | None, int | None]:
        """Find minimum ships to send from helper so that p stays owned by the player.

        Returns (angle, ships, arrive_tick), or (None, None, None) if helper cannot save p.
        """
        lo, hi = min_ships, 200
        while lo < hi:
            mid = (lo + hi) // 2
            angle, arrive_tick, is_blocked = self.aim(helper, p, mid)
            if arrive_tick is None:
                lo = mid + 1
                continue
            new_fleet = DirectedFleet(self.player, helper.id, p.id, angle, mid, arrive_tick)
            if self.is_capture_possible(p, mine_en_route + enemy_en_route + [new_fleet]):
                hi = mid
            else:
                lo = mid + 1

        angle, arrive_tick, is_blocked = self.aim(helper, p, lo)
        if arrive_tick is None or is_blocked:
            return None, None, None

        new_fleet = DirectedFleet(self.player, helper.id, p.id, angle, lo, arrive_tick)
        if not self.is_capture_possible(p, mine_en_route + enemy_en_route + [new_fleet]):
            return None, None, None

        return angle, lo, arrive_tick

    def ships_to_capture(
        self, src, tgt, mine_en_route: list[DirectedFleet], enemy_en_route: list[DirectedFleet],
    min_ships: int) -> tuple[float | None, int | None, int | None]:
        """Refine ship count to account for enemy production during travel.

        Returns (angle, ships, arrive_tick), or (None, None, None) if unreachable.
        """
        # Binary search for minimum ships regardless of being blocked
        lo, hi = min_ships, 200
        while lo < hi:
            mid = (lo + hi) // 2
            angle, arrive_tick, is_blocked = self.aim(src, tgt, mid)
            if arrive_tick is None:  # truly unreachable (beyond max_tick)
                lo = mid + 1
                continue
            
            if self.is_capture_possible(tgt, mine_en_route + enemy_en_route + [DirectedFleet(self.player, src.id, tgt.id, angle, mid, arrive_tick)]):
                
                hi = mid
            else:
                lo = mid + 1

        angle, arrive_tick, is_blocked = self.aim(src, tgt, lo)
        if arrive_tick is None or is_blocked:
            return None, None, None
            
        return angle, lo, arrive_tick

    def get_enemy_en_route(self, tgt) -> list[DirectedFleet]:
        """Return enemy fleets heading toward `tgt` within max_tick turns, sorted by arrive_tick.

        Filters paths blocked by the sun or intermediate planets, matching aiming logic.
        """
        tgt_traj = self._traj_cache.get(tgt.id)
        blockers = [p for p in self.planets if p.id != tgt.id]
        result: list[DirectedFleet] = []
        for f in self.fleets:
            if f.owner == self.player or f.ships <= 0:
                continue
            spd = fleet_speed(f.ships)
            cos_a = math.cos(f.angle) * spd
            sin_a = math.sin(f.angle) * spd
            px, py = f.x, f.y
            for tick in range(1, self.max_tick + 1):
                nx = f.x + cos_a * tick
                ny = f.y + sin_a * tick
                tx, ty = (
                    (tgt_traj[tick - 1] if tick <= len(tgt_traj) else (tgt.x, tgt.y))
                    if tgt_traj
                    else (tgt.x, tgt.y)
                )
                if point_to_segment_distance((tx, ty), (px, py), (nx, ny)) < tgt.radius:
                    if not hits_sun(f.x, f.y, f.angle, spd, tick) and self.path_blocking_detection and not path_blocked(
                        f.x, f.y, nx, ny, tick, spd, blockers, self._traj_cache
                    ):
                        result.append(
                            DirectedFleet(
                                f.owner, f.from_planet_id, tgt.id, f.angle, f.ships, tick
                            )
                        )
                    break
                px, py = nx, ny
        return sorted(result, key=lambda df: df.arrive_tick)
