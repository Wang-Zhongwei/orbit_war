"""stateful_agent.py — cross-turn memory, active defence, cooperative attacks, and comet logic."""
from __future__ import annotations

from contextlib import contextmanager

from src.config import StatefulAgentConfig
from src.helper import DirectedFleet, distance, fleet_speed
from src.game_context import GameContext
from .base import BaseStatefulAgent

@contextmanager
def record_target_scores(agent):
    """Temporarily intercept target_score on *agent* and collect all return values."""
    log: list[float] = []
    original = type(agent).target_score

    def patched(src, tgt):
        score = original(agent, src, tgt)
        log.append(score)
        return score

    agent.target_score = patched
    try:
        yield log
    finally:
        del agent.target_score


_COOP_PLANET_CAP = 8
_PROD_MULT = 10


class StatefulAgent(BaseStatefulAgent):
    """Stateful Orbit Wars agent with cross-turn memory, active defence, coop attacks, and comet awareness."""

    def __init__(self, config: StatefulAgentConfig | None = None) -> None:
        self.cfg = config or StatefulAgentConfig()
        super().__init__(max_tick=self.cfg.max_tick)

    def target_score(self, src, tgt, use_actual_time: bool = False) -> float:
        cfg   = self.cfg
        base  = tgt.ships + 1

        # add extra time if planet is rotating away from the source planet
        if use_actual_time: 
            _, time, _ = self._ctx.aim(src, tgt, base)
            if time is None:
                time = distance(src, tgt) / fleet_speed(base)
        else:
            time = distance(src, tgt) / fleet_speed(base)

        prod  = tgt.production
        enemy = tgt.owner != -1
        remaining = len(self._ctx.comet_paths.get(tgt.id, []))
        is_comet  = tgt.id in self._ctx.comet_ids
        # rank_fraction = self._ctx.planet_rank_fraction.get(tgt.id, 0.5) # 0: for core planets, 1: for boundary planets

        return (
            _PROD_MULT * prod
            + (cfg.enemy_bonus * prod if enemy else 0)
            # + cfg.boundary_score * rank_fraction
            - cfg.ships_penalty * (base + (time * prod if enemy else 0))
            - cfg.time_penalty * time
            - (cfg.comet_penalty / (remaining + 1) if is_comet else 0)
        )

    def _run(self, ctx: GameContext) -> list:
        my_planets = ctx.my_planets
        targets    = ctx.targets
        comet_ids  = ctx.comet_ids

        if not my_planets or not targets:
            return []

        min_ships    = self.cfg.min_ships_attack
        used_ids: set = set()
        moves: list   = []

        # ── Phase 0: Comet evacuation ─────────────────────────────────────────
        for comet_src in [p for p in my_planets if p.id in comet_ids]:
            if len(ctx.comet_paths.get(comet_src.id, [])) > 0:
                continue

            best_tgt = max(targets, key=lambda t: self.target_score(comet_src, t))
            angle, arrive_tick, is_blocked = ctx.aim(comet_src, best_tgt, comet_src.ships)

            if angle is not None and not is_blocked:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                ctx.record_fleet(comet_src, best_tgt, angle, comet_src.ships, arrive_tick)
                continue

            # fallback: send to closest own planet (excluding self)
            candidates = [p for p in my_planets if p.id != comet_src.id]
            if not candidates:
                continue
            closest = min(candidates, key=lambda p: distance(p, comet_src))
            angle, arrive_tick, is_blocked = ctx.aim(comet_src, closest, comet_src.ships)
            if angle is not None and not is_blocked:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                ctx.record_fleet(comet_src, closest, angle, comet_src.ships, arrive_tick)

        # ── Phase 1: Defence ─────────────────────────────────────────────────
        for p, plan in sorted(ctx.reinforcements_needed(min_ships).items(), key=lambda item: item[0].production, reverse=True):
            ships_needed   = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]

            helpers = sorted(
                [m for m in my_planets if m.id != p.id and m.id not in used_ids],
                key=lambda m: distance(m, p),
            )

            for helper in helpers:
                avail   = ctx.avail(helper)
                to_send = max(min_ships, ships_needed)
                if avail < to_send:
                    continue

                angle, arrive_tick, is_blocked = ctx.aim(helper, p, to_send)
                if angle is None or is_blocked or arrive_tick > needed_by_tick:
                    continue

                moves.append([helper.id, angle, to_send])
                used_ids.add(helper.id)
                ctx.record_fleet(helper, p, angle, to_send, arrive_tick)
                break

        # ── Phase 2: Offence ─────────────────────────────────────────────────
        owned_frac   = len(my_planets) / max(1, len(ctx.planets))
        attacked_ids = set()

        for src in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            if src.id in used_ids or src.ships < min_ships:
                continue

            src_avail = ctx.avail(src)
            if src_avail < min_ships:
                continue

            ranked = sorted(targets, key=lambda t: self.target_score(src, t), reverse=True)

            for tgt in ranked[:self.cfg.top_k_targets]:
                if tgt.id in attacked_ids:
                    continue
                
                enemy_en_route = ctx.get_enemy_en_route(tgt)
                mine_en_route = [f for f in ctx.my_fleets if f.tgt_id == tgt.id]

                if ctx.is_capture_possible(tgt, mine_en_route + enemy_en_route) and owned_frac < 0.75:
                    continue
               
                src_angle, src_needed, src_arrive = ctx.ships_to_capture(src, tgt, mine_en_route, enemy_en_route, self.cfg.min_ships_attack)

                if src_angle is None:
                    continue

                # 2.1 Single source attack
                if src_avail >= src_needed:
                    moves.append([src.id, src_angle, src_needed])
                    used_ids.add(src.id)
                    ctx.record_fleet(src, tgt, src_angle, src_needed, src_arrive)
                    attacked_ids.add(tgt.id)
                    break

                # 2.2 Coalition attack
                elif (
                    len(my_planets) > 1
                    and tgt.ships >= self.cfg.min_ships_coop_target
                    and tgt.id not in comet_ids
                ):
                    # src_avail < needed: recompute angle and arrive_tick for actual ship count
                    src_angle, src_arrive, is_blocked = ctx.aim(src, tgt, src_avail)
                    if src_angle is None or is_blocked:
                        continue
                    
                    coalition  = [{"planet": src, "angle": src_angle, "ships": src_avail, "arrive_tick": src_arrive}]
                    mine_en_route.append(DirectedFleet(ctx.player, src.id, tgt.id, src_angle, src_avail, src_arrive))
                    found = False

                    for h in sorted(my_planets, key=lambda p: distance(p, tgt)):
                        if h.id == src.id or h.id in used_ids:
                            continue

                        h_avail = ctx.avail(h)
                        if h_avail < min_ships:
                            continue

                        h_angle, h_needed, h_arrive = ctx.ships_to_capture(h, tgt, mine_en_route, enemy_en_route, self.cfg.min_ships_attack)
                        if h_angle is None:
                            continue

                        if h_needed > h_avail:
                            # update angle and arrive_tick since num_ships is h_avail rather than h_needed
                            h_angle, h_arrive, is_blocked = ctx.aim(h, tgt, h_avail)
                            if h_angle is None or is_blocked:
                                continue

                            coalition.append({"planet": h, "angle": h_angle, "ships": h_avail, "arrive_tick": h_arrive})
                            mine_en_route.append(DirectedFleet(ctx.player, h.id, tgt.id, h_angle, h_avail, h_arrive))
                        else:
                            coalition.append({"planet": h, "angle": h_angle, "ships": h_needed, "arrive_tick": h_arrive})
                            mine_en_route.append(DirectedFleet(ctx.player, h.id, tgt.id, h_angle, h_needed, h_arrive))
                            found = True
                            break

                        if len(coalition) >= _COOP_PLANET_CAP:
                            break
                       

                    if not found:
                        continue

                    for entry in coalition:
                        ctx.record_fleet(entry["planet"], tgt, entry["angle"], entry["ships"], entry["arrive_tick"])
                        used_ids.add(entry["planet"].id)
                        moves.append([entry["planet"].id, entry["angle"], entry["ships"]])

                    attacked_ids.add(tgt.id)
                    break

        return moves


def make_stateful_agent(config: StatefulAgentConfig) -> StatefulAgent:
    """Factory for tuning: returns a fresh StatefulAgent with isolated state."""
    return StatefulAgent(config)


_sa_instances: dict[int, StatefulAgent] = {}

def stateful_agent(obs) -> list:
    player = obs.get("player", 0)
    if player not in _sa_instances:
        _sa_instances[player] = StatefulAgent()
    return _sa_instances[player](obs)
