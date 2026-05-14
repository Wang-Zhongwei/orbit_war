"""stateful_agent.py — cross-turn memory, active defence, cooperative attacks, and comet logic."""
from __future__ import annotations

from ..configs import StatefulAgentConfig
from helper import (
    distance,
    fleet_speed,
    planet_trajectory,
    my_planets_under_attack,
    reinforcement_plans,
    sum_enemy_en_route,
)

from .base import BaseStatefulAgent
from .stateless_agents import GameContext


_COOP_PLANET_CAP       = 8
_SCORE_PROD_MULT       = 10

# Assumptions: 1. each turn one source can send up to one fleet 
class StatefulAgent(BaseStatefulAgent):
    """Stateful Orbit Wars agent with cross-turn memory, active defence, coop attacks, and comet awareness."""

    def __init__(self, config: StatefulAgentConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or StatefulAgentConfig()

    def target_score(self, src, tgt) -> float:
        dist = distance(src, tgt)
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

    # ── Main agent logic ──────────────────────────────────────────────────────
    def _run(self, obs) -> list:
        ctx              = GameContext.from_obs(obs)
        player           = ctx.player
        fleets           = ctx.fleets
        angular_velocity = ctx.angular_velocity
        comet_ids        = ctx.comet_ids
        my_planets       = ctx.my_planets
        targets          = ctx.enemies

        self._comet_ids   = comet_ids
        self._comet_paths = ctx.comet_paths
        self._traj_cache  = {
            p.id: planet_trajectory(p, self.cfg.max_tick, angular_velocity)
            for p in ctx.planets if p.id in self._moving_ids and p.id not in comet_ids
        }

        if not my_planets or not targets:
            return []

        self._age_my_fleets()

        min_ships     = self.cfg.min_ships_attack
        under_attack  = my_planets_under_attack(my_planets, fleets, player, self._traj_cache, self.cfg.max_tick)
        used_ids: set  = set()
        moves: list         = []

        # ── Phase 0: Comet evacuation ─────────────────────────────────────────
        _COMET_LEAVE_THRESHOLD = 1
        for comet_src in [p for p in my_planets if p.id in comet_ids]:
            if len(ctx.comet_paths.get(comet_src.id, [])) > _COMET_LEAVE_THRESHOLD:
                continue

            best_tgt = max(targets, key=lambda t: self.target_score(comet_src, t))
            angle, arrive_tick = self.aim(comet_src, best_tgt, comet_src.ships)

            if angle is not None:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                self._add_my_fleet(comet_src, best_tgt, angle, comet_src.ships, arrive_tick)
                break

            # fallback: send to closest own planet
            cloest_my_planet = min(my_planets, key=lambda p: (p.x - comet_src.x) ** 2 + (p.y - comet_src.y) ** 2)
            angle, arrive_tick = self.aim(comet_src, cloest_my_planet, comet_src.ships)

            if angle is not None:
                moves.append([comet_src.id, angle, comet_src.ships])
                used_ids.add(comet_src.id)
                self._add_my_fleet(comet_src, cloest_my_planet, angle, comet_src.ships, arrive_tick)
                break
         
        # ── Phase 1: Defence ─────────────────────────────────────────────────
        for p, plan in reinforcement_plans(my_planets, under_attack, self._my_fleets, min_ships).items():

            ships_needed   = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]

            helpers = sorted(
                [m for m in my_planets if m.id != p.id and m.id not in used_ids],
                key=lambda m: (m.x - p.x) ** 2 + (m.y - p.y) ** 2,
            )

            for helper in helpers:
                avail = self._avail(helper, under_attack)

                to_send = max(min_ships, ships_needed)
                if avail < to_send:
                    continue

                angle, arrive_tick = self.aim(helper, p, to_send)

                if angle is None or arrive_tick > needed_by_tick:
                    continue

                moves.append([helper.id, angle, to_send])
                used_ids.add(helper.id)
                self._add_my_fleet(helper, p, angle, to_send, arrive_tick)
                break


        # ── Phase 2: Offence ─────────────────────────────────────────────────
        owned_frac    = len(my_planets) / max(1, len(ctx.planets))
        attacked_ids = set()
        # Find source sorted by ships
        for src in sorted(my_planets, key=lambda p: p.ships, reverse=True):
            if src.id in used_ids or src.ships < min_ships:
                continue

            src_avail = self._avail(src, under_attack)
            ranked    = sorted(targets, key=lambda t: self.target_score(src, t), reverse=True)

            for tgt in ranked[:self.cfg.top_k_targets]:
                if tgt.id in attacked_ids:
                    continue

                enemy_racing = sum_enemy_en_route(tgt, fleets, player, self._traj_cache.get(tgt.id), self.cfg.max_tick)

                en_route = sum(f.ships for f in self._my_fleets if f.tgt_id == tgt.id)

                base = enemy_racing + tgt.ships + 1
                if tgt.owner != -1:
                    base += self.cfg.attack_buffer * tgt.production

                if owned_frac < 0.75 and en_route >= base:
                    continue

                needed = max(min_ships, base - en_route)

                # 2.1 Single source attack
                angle, needed, arrive_tick = self._refine_offense_ships_needed(src, tgt, needed)

                if angle is None:
                    continue

                if src_avail >= needed:
                    moves.append([src.id, angle, needed])
                    used_ids.add(src.id)
                    self._add_my_fleet(src, tgt, angle, needed, arrive_tick)
                    attacked_ids.add(tgt.id)
                    break

                # 2.2 Coalition attack
                elif len(my_planets) > 1 and tgt.ships >= self.cfg.min_ships_coop_target and tgt.id not in comet_ids:
                    
                    coalition = [{"planet": src, "angle": angle, "ships": src_avail, "arrive_tick": arrive_tick}]
                    max_tick = arrive_tick
                    tot_ships = src_avail
                    for h in sorted(my_planets, key=lambda p: (p.x - tgt.x) ** 2 + (p.y - tgt.y) ** 2):
                        if h.id == src.id or h.id in used_ids:
                            continue

                        h_avail = self._avail(h, under_attack)
                        if h_avail < min_ships: 
                            continue

                        angle, arrive_tick = self.aim(h, tgt, h_avail)
                        if angle is None: 
                            continue

                        max_tick = max(max_tick, arrive_tick)
                        tot_ships += h_avail

                        coalition.append({"planet": h, "angle": angle, "ships": h_avail, "arrive_tick": arrive_tick})

                        if len(coalition) >= _COOP_PLANET_CAP:
                            break

                        if tot_ships >= needed + max_tick * tgt.production:
                            break
                    
                    if tot_ships < needed + max_tick * tgt.production:
                        continue


                    for entry in coalition:
                        self._add_my_fleet(entry["planet"], tgt, entry["angle"], entry["ships"], entry["arrive_tick"])
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


