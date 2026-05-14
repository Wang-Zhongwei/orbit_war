"""stateful_agent.py — cross-turn memory, active defence, cooperative attacks, and comet logic."""
from __future__ import annotations

from ..configs import StatefulAgentConfig
from helper import distance, fleet_speed, sum_enemy_en_route
from game_context import GameContext
from .base import BaseStatefulAgent


_COOP_PLANET_CAP = 8
_SCORE_PROD_MULT = 10


class StatefulAgent(BaseStatefulAgent):
    """Stateful Orbit Wars agent with cross-turn memory, active defence, coop attacks, and comet awareness."""

    def __init__(self, config: StatefulAgentConfig | None = None) -> None:
        self.cfg = config or StatefulAgentConfig()
        super().__init__(max_tick=self.cfg.max_tick)

    def target_score(self, src, tgt) -> float:
        dist  = distance(src, tgt)
        cfg   = self.cfg
        base  = tgt.ships + 1
        time  = dist / fleet_speed(base)
        prod  = tgt.production
        enemy = tgt.owner != -1
        return (
            _SCORE_PROD_MULT * prod
            + (cfg.score_enemy_bonus * prod if enemy else 0)
            - cfg.score_ships_penalty * (base + (time * prod if enemy else 0))
            - cfg.score_time_penalty * time
        )

    def _run(self, ctx: GameContext) -> list:
        my_planets = ctx.my_planets
        targets    = ctx.enemies
        comet_ids  = ctx.comet_ids

        if not my_planets or not targets:
            return []

        min_ships    = self.cfg.min_ships_attack
        used_ids: set = set()
        moves: list   = []

        # ── Phase 0: Comet evacuation ─────────────────────────────────────────
        _COMET_LEAVE_THRESHOLD = 1
        for comet_src in [p for p in my_planets if p.id in comet_ids]:
            if len(ctx.comet_paths.get(comet_src.id, [])) > _COMET_LEAVE_THRESHOLD:
                continue

            best_tgt = max(targets, key=lambda t: self.target_score(comet_src, t))
            angle, arrive_tick = ctx.aim(comet_src, best_tgt, comet_src.ships)

            if angle is not None:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                ctx.record_fleet(comet_src, best_tgt, angle, comet_src.ships, arrive_tick)
                break

            # fallback: send to closest own planet
            closest = min(my_planets, key=lambda p: (p.x - comet_src.x) ** 2 + (p.y - comet_src.y) ** 2)
            angle, arrive_tick = ctx.aim(comet_src, closest, comet_src.ships)
            if angle is not None:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                ctx.record_fleet(comet_src, closest, angle, comet_src.ships, arrive_tick)
                break

        # ── Phase 1: Defence ─────────────────────────────────────────────────
        for p, plan in ctx.reinforcements_needed(min_ships).items():
            ships_needed   = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]

            helpers = sorted(
                [m for m in my_planets if m.id != p.id and m.id not in used_ids],
                key=lambda m: (m.x - p.x) ** 2 + (m.y - p.y) ** 2,
            )

            for helper in helpers:
                avail   = ctx.avail(helper)
                to_send = max(min_ships, ships_needed)
                if avail < to_send:
                    continue

                angle, arrive_tick = ctx.aim(helper, p, to_send)
                if angle is None or arrive_tick > needed_by_tick:
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
            ranked    = sorted(targets, key=lambda t: self.target_score(src, t), reverse=True)

            for tgt in ranked[:self.cfg.top_k_targets]:
                if tgt.id in attacked_ids:
                    continue

                enemy_racing = sum_enemy_en_route(
                    tgt, ctx.fleets, ctx.player, ctx.trajectory(tgt.id), self.cfg.max_tick
                )
                en_route = sum(f.ships for f in ctx.my_fleets if f.tgt_id == tgt.id)

                base = enemy_racing + tgt.ships + 1
                if tgt.owner != -1:
                    base += self.cfg.attack_buffer * tgt.production

                if owned_frac < 0.75 and en_route >= base:
                    continue

                needed = max(min_ships, base - en_route)
                src_angle, needed, src_arrive = ctx.ships_to_capture(src, tgt, needed)

                if src_angle is None:
                    continue

                # 2.1 Single source attack
                if src_avail >= needed:
                    moves.append([src.id, src_angle, needed])
                    used_ids.add(src.id)
                    ctx.record_fleet(src, tgt, src_angle, needed, src_arrive)
                    attacked_ids.add(tgt.id)
                    break

                # 2.2 Coalition attack
                elif (
                    len(my_planets) > 1
                    and tgt.ships >= self.cfg.min_ships_coop_target
                    and tgt.id not in comet_ids
                ):
                    coalition  = [{"planet": src, "angle": src_angle, "ships": src_avail, "arrive_tick": src_arrive}]
                    max_tick_v = src_arrive
                    tot_ships  = src_avail

                    for h in sorted(my_planets, key=lambda p: (p.x - tgt.x) ** 2 + (p.y - tgt.y) ** 2):
                        if h.id == src.id or h.id in used_ids:
                            continue

                        h_avail = ctx.avail(h)
                        if h_avail < min_ships:
                            continue

                        h_angle, h_arrive = ctx.aim(h, tgt, h_avail)
                        if h_angle is None:
                            continue

                        max_tick_v = max(max_tick_v, h_arrive)
                        tot_ships += h_avail
                        coalition.append({"planet": h, "angle": h_angle, "ships": h_avail, "arrive_tick": h_arrive})

                        if len(coalition) >= _COOP_PLANET_CAP:
                            break
                        if tot_ships >= needed + max_tick_v * tgt.production:
                            break

                    if tot_ships < needed + max_tick_v * tgt.production:
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
