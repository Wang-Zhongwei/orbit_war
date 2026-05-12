"""
rule_based.py — rule-based agents sharing a common configurable pipeline.

All three canonical agents (sniper, roi, never_miss) are thin wrappers over
``run_agent``, which executes this shared decision pipeline each turn:

    1. Parse obs → ``GameContext``
    2. Order source planets  (``AgentConfig.source_sort``)
    3. For each source planet:
        a. Select one target, skipping already-assigned and comet planets.
        b. Mark target as assigned (prevents two sources hitting the same planet).
        c. Call ``aim_for_planet`` to compute angle, ship count, and ETA.
        d. Accept when capturable, or when source is a comet and
           ``AgentConfig.accept_comet_src`` is True.
        e. Record the move with its cost scalar.
    4. Sort candidates by cost (ascending) and cap at ``AgentConfig.max_moves``.

Adding a new agent
------------------
Define an ``AgentConfig`` preset and a 1-arg wrapper::

    MY_CONFIG = AgentConfig(target_mode="nearest", max_moves=4)

    def my_agent(obs) -> list:
        return run_agent(obs, MY_CONFIG)

To make it tunable, register it in ``tune.py``::

    REGISTRY.register("my_agent", TunableStrategy(
        factory=lambda cfg: (lambda obs: run_agent(obs, AgentConfig(**cfg))),
        params=[CategoricalParam("target_mode", ["nearest", "roi"]), ...],
        agent_fn_name="run_agent",
        description="my_agent: ...",
    ))
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from helper import (
    CENTER,
    SUN_RADIUS,
    aim_for_planet,
    best_target,
    fleet_speed,
    point_to_segment_distance,
    predict_planet_position,
    unpack_obs,
)


# ---------------------------------------------------------------------------
# GameContext — parsed observation for one turn
# ---------------------------------------------------------------------------

@dataclass
class GameContext:
    """Parsed game state for one turn, derived from a raw observation dict.

    Attributes:
        player: This agent's player ID.
        planets: All planets this turn.
        comet_ids: Planet IDs that are currently comets.
        angular_velocity: Inner-planet rotation speed (rad/turn).
        fleets: All active fleets.
        comet_paths: Remaining waypoints per comet planet ID.
        my_planets: Planets owned by this agent (derived in ``__post_init__``).
        enemies: Planets not owned by this agent (derived in ``__post_init__``).
    """

    player: int
    planets: list[Planet]
    comet_ids: set[int]
    angular_velocity: float
    fleets: list
    comet_paths: dict
    my_planets: list[Planet] = field(init=False, repr=False)
    enemies: list[Planet]    = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.enemies    = [p for p in self.planets if p.owner != self.player]

    @classmethod
    def from_obs(cls, obs) -> GameContext:
        """Construct from a raw Kaggle Environments observation dict.

        Args:
            obs: Raw observation dict from the environment.

        Returns:
            Populated ``GameContext`` instance.
        """
        player, planets, comet_ids, angular_velocity, fleets, comet_paths = unpack_obs(obs)
        return cls(
            player=player,
            planets=planets,
            comet_ids=comet_ids,
            angular_velocity=angular_velocity,
            fleets=fleets,
            comet_paths=comet_paths,
        )


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


# ---------------------------------------------------------------------------
# Pre-built configs for the three canonical agents
# ---------------------------------------------------------------------------

SNIPER_CONFIG = AgentConfig(
    source_sort="none",
    target_mode="nearest",
    accept_comet_src=False,
    cost_mode="none",
    max_moves=None,
)

ROI_CONFIG = AgentConfig(
    source_sort="richest_first",
    target_mode="roi",
    accept_comet_src=False,
    cost_mode="none",
    max_moves=None,
)

NEVER_MISS_DEFAULT_CONFIG = AgentConfig(
    source_sort="none",
    target_mode="random_biased",
    limit_top_k_fraction=0.12,
    bias_draws=1,
    accept_comet_src=True,
    cost_mode="turns_weighted",
    cost_weight=2.0,
    max_moves=10,
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _sort_sources(planets: list[Planet], mode: str) -> list[Planet]:
    """Return source planets ordered by ``mode``.

    Args:
        planets: All planets owned by this agent.
        mode: ``"none"`` or ``"richest_first"``.

    Returns:
        Reordered list (does not mutate the input).

    Raises:
        ValueError: If ``mode`` is not a recognised value.
    """
    if mode == "none":
        return planets
    if mode == "richest_first":
        return sorted(planets, key=lambda p: p.ships, reverse=True)
    raise ValueError(f"Unknown source_sort: {mode!r}")


def _select_target(
    src: Planet,
    ctx: GameContext,
    config: AgentConfig,
    assigned: set[int],
) -> Planet | None:
    """Pick one attack target for ``src``, respecting the assigned set.

    Comet planets are excluded from all pools — ``aim_for_planet`` cannot
    reliably track them along their irregular paths.

    Args:
        src: Source planet that will fire.
        ctx: Current game context.
        config: Agent configuration controlling target selection.
        assigned: Target IDs already claimed by earlier source planets this turn.

    Returns:
        Selected enemy ``Planet``, or ``None`` if no valid target exists.

    Raises:
        ValueError: If ``config.target_mode`` is not a recognised value.
    """
    if config.target_mode == "roi":
        # best_target handles assigned filtering and ROI scoring internally.
        # It may return a comet; the caller checks ctx.comet_ids afterward.
        return best_target(ctx.planets, src, ctx.player, assigned, ctx.comet_ids)

    # Explicit pool: enemies not yet assigned, comets excluded up front.
    pool = [
        p for p in ctx.enemies
        if p.id not in assigned and p.id not in ctx.comet_ids
    ]
    if not pool:
        return None

    # Sort ascending by squared distance (avoids sqrt, monotone with distance).
    pool.sort(key=lambda t: (src.x - t.x) ** 2 + (src.y - t.y) ** 2)

    if config.target_mode == "nearest":
        return pool[0]

    if config.target_mode == "random_biased":
        k = max(1, int(len(pool) * config.limit_top_k_fraction))
        pool = pool[:k]
        n = len(pool)
        # min of ``bias_draws`` uniform samples → closer indices more likely.
        idx = min(random.randint(0, n - 1) for _ in range(max(1, config.bias_draws)))
        return pool[idx]

    raise ValueError(f"Unknown target_mode: {config.target_mode!r}")


def _compute_cost(turns_to_arrive: int, can_capture: bool, config: AgentConfig) -> float:
    """Compute the sort key for one candidate move (lower = higher priority).

    Args:
        turns_to_arrive: Estimated flight time in turns.
        can_capture: Whether the fleet is large enough to capture the planet.
        config: Agent configuration controlling cost computation.

    Returns:
        Scalar cost used for sorting candidates before the ``max_moves`` cap.

    Raises:
        ValueError: If ``config.cost_mode`` is not a recognised value.
    """
    if config.cost_mode == "none":
        return 0.0
    if config.cost_mode == "turns_weighted":
        return turns_to_arrive * (config.cost_weight - float(can_capture))
    raise ValueError(f"Unknown cost_mode: {config.cost_mode!r}")


# ---------------------------------------------------------------------------
# Shared agent pipeline
# ---------------------------------------------------------------------------

def run_agent(obs, config: AgentConfig) -> list[list]:
    """Execute the shared decision pipeline and return a list of moves.

    Pipeline steps:
        1. Parse ``obs`` into a ``GameContext``.
        2. Order source planets by ``config.source_sort``.
        3. For each source:
           - Select one target (assigned + comet exclusion).
           - Mark target assigned (prevents duplicate attacks).
           - Run ``aim_for_planet`` for angle, ship count, and ETA.
           - Accept when capturable, or source is comet + ``accept_comet_src``.
           - Record ``(move, cost)`` pair.
        4. Sort by cost, cap at ``config.max_moves``, return move triples.

    Args:
        obs: Raw observation dict from the Kaggle Environments environment.
        config: Micro-decision configuration controlling every branch.

    Returns:
        List of ``[from_planet_id, angle_rad, num_ships]`` triples.
    """
    ctx = GameContext.from_obs(obs)
    if not ctx.my_planets or not ctx.enemies:
        return []

    sources   = _sort_sources(ctx.my_planets, config.source_sort)
    assigned: set[int] = set()
    candidates: list[tuple[list, float]] = []

    for src in sources:
        target = _select_target(src, ctx, config, assigned)
        if target is None:
            continue

        # Consume the target slot regardless of aim success, so later source
        # planets don't pile onto the same unreachable or comet target.
        assigned.add(target.id)

        if target.id in ctx.comet_ids:
            continue  # roi mode may return comets; skip them here

        result = aim_for_planet(target, src, ctx.angular_velocity)
        if result is None:
            continue

        angle, ships, can_capture, turns_to_arrive = result
        my_is_comet = src.id in ctx.comet_ids

        if not can_capture and not (config.accept_comet_src and my_is_comet):
            continue

        cost = _compute_cost(turns_to_arrive, can_capture, config)
        candidates.append(([src.id, angle, ships], cost))

    candidates.sort(key=lambda x: x[1])
    cap = config.max_moves if config.max_moves is not None else len(candidates)
    return [move for move, _ in candidates[:cap]]


# ---------------------------------------------------------------------------
# Named agent wrappers — backward-compatible 1-arg entry points
# ---------------------------------------------------------------------------

def nearest_planet_sniper(obs) -> list:
    """Sniper agent: attacks the closest enemy from each owned planet.

    Args:
        obs: Raw observation dict from the environment.

    Returns:
        List of ``[planet_id, angle, ships]`` moves.
    """
    return run_agent(obs, SNIPER_CONFIG)


def roi_agent(obs) -> list:
    """ROI agent: richest-first source ordering, production/cost target scoring.

    Args:
        obs: Raw observation dict from the environment.

    Returns:
        List of ``[planet_id, angle, ships]`` moves.
    """
    return run_agent(obs, ROI_CONFIG)


def never_miss_agent(obs) -> list:
    """Distance-biased random attacker, fully configurable via ``AgentConfig``.

    This is the primary tunable agent: pass a custom ``AgentConfig`` (built by
    ``tune.py``'s factory) to evaluate any combination of micro-decisions.

    Args:
        obs: Raw observation dict from the environment.

    Returns:
        List of ``[planet_id, angle, ships]`` moves.
    """
    return run_agent(obs, NEVER_MISS_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Stateful agent — cross-turn memory, active defence, cooperative attacks
# ---------------------------------------------------------------------------

_MIN_SHIPS_COOP_TARGET = 20   # not tuned
_COOP_PLANET_CAP       = 8    # not tuned


@dataclass
class StatefulAgentConfig:
    """Tunable hyperparameters for :class:`StatefulAgent`."""

    score_dist_base:     float = 100.0
    score_prod_mult:     float = 15.0
    score_enemy_bonus:   float = 10.0
    score_ships_penalty: float = 0.7
    score_eta_penalty:   float = 2.0
    min_ships_attack:    int   = 10
    top_k_targets:       int   = 3


class StatefulAgent:
    """Stateful Orbit Wars agent with cross-turn memory, active defence, and coop attacks."""

    def __init__(self, config: StatefulAgentConfig | None = None) -> None:
        self.cfg           = config or StatefulAgentConfig()
        self._steps:       int  = 0
        self._moving_ids:  list = []
        self._fleet_traj:  list = []
        self._reinf_traj:  list = []

    def __call__(self, obs) -> list:
        return self._run(obs)

    # ── Pure helpers ──────────────────────────────────────────────────────────

    def _hits_sun(self, src_x: float, src_y: float, angle: float, spd: float) -> bool:
        end_x = src_x + math.cos(angle) * spd * 60
        end_y = src_y + math.sin(angle) * spd * 60
        return point_to_segment_distance((CENTER, CENTER), (src_x, src_y), (end_x, end_y)) <= SUN_RADIUS

    def _find_angle(self, src, tgt, ships: int, angular_velocity: float):
        spd = fleet_speed(ships)
        for tick in range(1, 61):
            tx, ty = predict_planet_position(tgt, tick, angular_velocity)
            dist = math.sqrt((src.x - tx) ** 2 + (src.y - ty) ** 2)
            if abs(math.floor(dist / spd) - tick) <= 1:
                angle = math.atan2(ty - src.y, tx - src.x)
                if self._hits_sun(src.x, src.y, angle, spd):
                    return None, None
                return angle, tick
        return None, None

    def _refine_ships_static_target(self, src, tgt, base: int) -> int:
        total = base
        for _ in range(3):
            dist = math.sqrt((tgt.x - src.x) ** 2 + (tgt.y - src.y) ** 2)
            spd  = fleet_speed(total)
            tick = math.floor(dist / spd)
            new  = base + (tick * tgt.production if tgt.owner != -1 else 0)
            if new == total:
                break
            total = new
        return total

    def _refine_ships_moving_target(self, src, tgt, needed: int, angular_velocity: float, src_avail: int):
        total_ships = needed
        arrive_tick = None
        for _ in range(5):
            angle, arrive_tick = self._find_angle(src, tgt, total_ships, angular_velocity)
            if angle is None:
                return None, needed, None
            new_total = needed + (arrive_tick * tgt.production if tgt.owner != -1 else 0)
            if new_total > src_avail:
                return None, needed, None
            if new_total == total_ships:
                break
            total_ships = new_total
        return angle, total_ships, arrive_tick

    def _score(self, src, tgt) -> float:
        dist = math.sqrt((src.x - tgt.x) ** 2 + (src.y - tgt.y) ** 2)
        if dist == 0:
            return -1e9
        cfg   = self.cfg
        base  = tgt.ships + 1
        spd   = fleet_speed(base)
        eta   = dist / spd
        prod  = tgt.production
        enemy = tgt.owner != -1
        return (
            (cfg.score_dist_base - dist)
            + cfg.score_prod_mult * prod
            + (cfg.score_enemy_bonus * prod if enemy else 0)
            - cfg.score_ships_penalty * (base + (eta * prod if enemy else 0))
            - cfg.score_eta_penalty * eta
        )

    # ── State-mutating helpers ────────────────────────────────────────────────

    def _detect_moving_planets(self, obs) -> None:
        planets  = [Planet(*p) for p in obs.get("planets", [])]
        init_map = {i[0]: Planet(*i) for i in obs.get("initial_planets", [])}
        for p in planets:
            i = init_map.get(p.id)
            if i and (round(p.x, 3), round(p.y, 3)) != (round(i.x, 3), round(i.y, 3)):
                if p.id not in self._moving_ids:
                    self._moving_ids.append(p.id)

    def _age_fleet_traj(self, live_fleets) -> None:
        for ft in self._fleet_traj[:]:
            alive = any(
                f.from_planet_id == ft["mine"].id and abs(f.angle - ft["angle"]) < 1e-3
                for f in live_fleets
            )
            if alive:
                ft["arrive_tick"] = max(0, ft["arrive_tick"] - 1)
            else:
                self._fleet_traj.remove(ft)

    def _age_reinf_traj(self) -> None:
        for rt in self._reinf_traj[:]:
            rt["arrive_tick"] -= 1
            if rt["arrive_tick"] <= 0:
                self._reinf_traj.remove(rt)

    def _planets_under_attack(self, mine, fleets, player: int, angular_velocity: float) -> dict:
        enemy  = [f for f in fleets if f.owner != player and f.ships > 0]
        result: dict = {}
        seen:   set  = set()

        mov_traj = {
            m.id: [predict_planet_position(m, t, angular_velocity) for t in range(1, 61)]
            for m in mine if m.id in self._moving_ids
        }

        for f in enemy:
            spd = fleet_speed(f.ships)
            px, py = f.x, f.y
            for tick in range(1, 61):
                nx = f.x + math.cos(f.angle) * spd * tick
                ny = f.y + math.sin(f.angle) * spd * tick
                for m in mine:
                    mx, my = mov_traj[m.id][tick - 1] if m.id in mov_traj else (m.x, m.y)
                    if point_to_segment_distance((mx, my), (px, py), (nx, ny)) <= m.radius:
                        key = (m.id, f.id)
                        if key not in seen:
                            seen.add(key)
                            result.setdefault(m.id, {"planet": m, "fleets": []})
                            result[m.id]["fleets"].append({"fleet": f, "arrive_tick": tick})
                px, py = nx, ny
        return result

    def _sum_enemy_en_route(self, tgt, fleets, player: int, angular_velocity: float) -> int:
        tgt_traj = (
            [predict_planet_position(tgt, t, angular_velocity) for t in range(1, 61)]
            if tgt.id in self._moving_ids else None
        )
        total = 0
        for f in fleets:
            if f.owner == player or f.ships <= 0:
                continue
            spd = fleet_speed(f.ships)
            px, py = f.x, f.y
            for tick in range(1, 61):
                nx = f.x + math.cos(f.angle) * spd * tick
                ny = f.y + math.sin(f.angle) * spd * tick
                tx, ty = tgt_traj[tick - 1] if tgt_traj else (tgt.x, tgt.y)
                if point_to_segment_distance((tx, ty), (px, py), (nx, ny)) <= tgt.radius:
                    total += f.ships
                    break
                px, py = nx, ny
        return total

    def _refine_coalition(
        self, coalition: list, needed: int, tgt, angular_velocity: float | None = None
    ) -> tuple[int, list | None]:
        moving  = angular_velocity is not None and tgt.id in self._moving_ids
        refined = needed
        plans   = None
        min_ships = self.cfg.min_ships_attack
        for _ in range(5):
            max_arr  = 0
            rem      = refined
            cur_plans = []
            ok        = True
            for a in coalition:
                if rem <= 0:
                    break
                p_s = max(min(a["ships"], rem), min_ships)
                if moving:
                    ang, arr = self._find_angle(a["planet"], tgt, p_s, angular_velocity)
                    if arr is None:
                        return refined, None
                else:
                    dx, dy = tgt.x - a["planet"].x, tgt.y - a["planet"].y
                    ang    = math.atan2(dy, dx)
                    spd    = fleet_speed(p_s)
                    if self._hits_sun(a["planet"].x, a["planet"].y, ang, spd):
                        ok = False
                        break
                    dist = math.sqrt(dx * dx + dy * dy)
                    arr  = math.floor(dist / spd)
                cur_plans.append([a["planet"], ang, p_s, arr])
                max_arr = max(max_arr, arr)
                rem     = max(0, rem - p_s)
            if not ok:
                return refined, None
            new_needed = needed + (max_arr * tgt.production if tgt.owner != -1 else 0)
            plans = cur_plans
            if new_needed == refined:
                break
            refined = new_needed
        return refined, plans

    def _reinforcement_plans(self, mine, under_attack: dict) -> dict:
        plans:     dict = {}
        min_ships        = self.cfg.min_ships_attack
        for m in mine:
            if m.id not in under_attack:
                continue
            atk_sorted = sorted(under_attack[m.id]["fleets"], key=lambda a: a["arrive_tick"])
            incoming   = sorted(
                [r for r in self._reinf_traj if r["target"].id == m.id],
                key=lambda r: r["arrive_tick"],
            )
            avail     = float(m.ships)
            prev_tick = 0
            r_idx     = 0
            for att in atk_sorted:
                t = att["arrive_tick"]
                avail += (t - prev_tick) * m.production
                while r_idx < len(incoming) and incoming[r_idx]["arrive_tick"] <= t:
                    avail += incoming[r_idx]["total_ships"]
                    r_idx += 1
                avail    -= att["fleet"].ships
                prev_tick = t
                if avail < 0:
                    plans[m] = {
                        "ships_needed":   max(min_ships, int(abs(avail))),
                        "needed_by_tick": t,
                    }
                    break
        return plans

    # ── Main agent logic ──────────────────────────────────────────────────────

    def _run(self, obs) -> list:
        if int(obs.get("step", -1)) == 0 and self._steps > 0:
            self._steps = 0
            self._moving_ids.clear()
            self._fleet_traj.clear()
            self._reinf_traj.clear()

        if self._steps < 2:
            self._steps += 1
            return []
        if self._steps == 2:
            self._detect_moving_planets(obs)
            self._steps = 3

        ctx              = GameContext.from_obs(obs)
        player           = ctx.player
        fleets           = ctx.fleets
        angular_velocity = ctx.angular_velocity
        comet_ids        = ctx.comet_ids
        my_planets       = ctx.my_planets
        targets          = ctx.enemies

        if not my_planets or not targets:
            return []

        self._age_fleet_traj(fleets)
        self._age_reinf_traj()

        min_ships     = self.cfg.min_ships_attack
        under_attack  = self._planets_under_attack(my_planets, fleets, player, angular_velocity)
        exhausted_ids: set  = set()
        moves: list         = []

        # ── Phase 1: Defence ─────────────────────────────────────────────────
        for p, plan in self._reinforcement_plans(my_planets, under_attack).items():
            if any(r["target"].id == p.id for r in self._reinf_traj):
                continue

            ships_needed   = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]

            helpers = sorted(
                [m for m in my_planets if m.id != p.id and m.id not in exhausted_ids],
                key=lambda m: (m.x - p.x) ** 2 + (m.y - p.y) ** 2,
            )

            for helper in helpers:
                promised = sum(r["total_ships"] for r in self._reinf_traj if r["mine"].id == helper.id)
                if helper.id in under_attack:
                    promised += sum(a["fleet"].ships for a in under_attack[helper.id]["fleets"])
                avail = max(0, helper.ships - promised)

                to_send = max(min_ships, ships_needed)
                if avail < to_send:
                    continue

                if p.id in self._moving_ids:
                    angle, arrive_tick = self._find_angle(helper, p, to_send, angular_velocity)
                else:
                    dx, dy      = p.x - helper.x, p.y - helper.y
                    angle       = math.atan2(dy, dx)
                    spd         = fleet_speed(to_send)
                    dist        = math.sqrt(dx * dx + dy * dy)
                    arrive_tick = math.floor(dist / spd)

                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue

                moves.append([helper.id, angle, to_send])
                exhausted_ids.add(helper.id)
                self._reinf_traj.append({
                    "mine": helper, "target": p,
                    "angle": angle, "total_ships": to_send, "arrive_tick": arrive_tick,
                })
                break

        # ── Phase 2: Offence ─────────────────────────────────────────────────
        valid_targets = [t for t in targets if t.id not in comet_ids]
        owned_frac    = len(my_planets) / max(1, len(ctx.planets))

        for src in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            if src.id in exhausted_ids or src.ships < min_ships:
                continue

            ranked = sorted(valid_targets, key=lambda t: self._score(src, t), reverse=True)

            for tgt in ranked[:self.cfg.top_k_targets]:
                en_route = sum(ft["total_ships"] for ft in self._fleet_traj if ft["target"].id == tgt.id)

                enemy_racing = self._sum_enemy_en_route(tgt, fleets, player, angular_velocity)
                base = enemy_racing + tgt.ships + 1
                if tgt.owner != -1:
                    base += 2 * tgt.production

                if owned_frac < 0.75 and en_route >= base:
                    continue

                needed = max(min_ships, base - en_route)

                src_avail = src.ships
                if src.id in under_attack:
                    src_avail -= sum(a["fleet"].ships for a in under_attack[src.id]["fleets"])
                src_avail = max(0, src_avail)

                angle, arrive_tick, total_ships = None, None, 0

                if src_avail >= needed:
                    if tgt.id in self._moving_ids:
                        angle, total_ships, arrive_tick = self._refine_ships_moving_target(
                            src, tgt, needed, angular_velocity, src_avail
                        )
                    else:
                        dx, dy      = tgt.x - src.x, tgt.y - src.y
                        angle_raw   = math.atan2(dy, dx)
                        total_ships = self._refine_ships_static_target(src, tgt, needed) if tgt.owner != -1 else needed
                        spd         = fleet_speed(total_ships)
                        if total_ships <= src_avail and not self._hits_sun(src.x, src.y, angle_raw, spd):
                            angle       = angle_raw
                            dist        = math.sqrt(dx * dx + dy * dy)
                            arrive_tick = math.floor(dist / spd)

                    if angle is not None:
                        moves.append([src.id, angle, total_ships])
                        exhausted_ids.add(src.id)
                        self._fleet_traj.append({
                            "mine": src, "target": tgt,
                            "angle": angle, "total_ships": total_ships, "arrive_tick": arrive_tick,
                        })
                        break

                elif len(my_planets) > 1 and tgt.ships >= _MIN_SHIPS_COOP_TARGET:
                    helpers = []
                    for h in sorted(my_planets, key=lambda p: (p.x - tgt.x) ** 2 + (p.y - tgt.y) ** 2):
                        if h.id == src.id or h.id in exhausted_ids:
                            continue
                        h_avail = h.ships
                        if h.id in under_attack:
                            h_avail -= sum(a["fleet"].ships for a in under_attack[h.id]["fleets"])
                        h_avail = max(0, h_avail)
                        if h_avail < min_ships:
                            continue
                        helpers.append((h, h_avail))

                    coalition = [{"planet": src, "ships": src_avail}]
                    for h, h_avail in helpers:
                        coalition.append({"planet": h, "ships": h_avail})
                        if sum(a["ships"] for a in coalition) >= needed:
                            break
                        if len(coalition) > _COOP_PLANET_CAP:
                            break

                    if sum(a["ships"] for a in coalition) < needed:
                        continue

                    needed, planned = self._refine_coalition(coalition, needed, tgt, angular_velocity)
                    if planned is None:
                        continue

                    if sum(a["ships"] for a in coalition) < needed:
                        continue

                    for move in planned:
                        self._fleet_traj.append({
                            "mine": move[0], "target": tgt,
                            "angle": move[1], "total_ships": move[2], "arrive_tick": move[3],
                        })
                        exhausted_ids.add(move[0].id)
                        moves.append([move[0].id, move[1], move[2]])
                    break

        return moves


def make_stateful_agent(config: StatefulAgentConfig) -> StatefulAgent:
    """Factory for tuning: returns a fresh StatefulAgent with isolated state."""
    return StatefulAgent(config)


_sa_default = StatefulAgent()


def stateful_agent(obs) -> list:
    return _sa_default(obs)
