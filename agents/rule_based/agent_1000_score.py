"""agent_1000_score.py — class-based wrapper for submission/1000_score.py with isolated per-game state."""
from __future__ import annotations

import importlib.util
import os


def _load_module():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "submission", "1000_score.py")
    spec = importlib.util.spec_from_file_location("_submission_1000_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Agent1000Score:
    """Per-player instance of the 1000_score submission agent with isolated state.

    State is reset automatically on obs.step == 0 (new game).
    """

    def __init__(self):
        self._steps = 0
        self._fleet_traj: list = []
        self._reinf_traj: list = []
        self._moving_planets: list = []
        self._mod = _load_module()

    def __call__(self, obs) -> list:
        mod = self._mod
        moves = []

        if self._steps < 2:
            self._steps += 1
            return []
        if self._steps == 2:
            self._fill_moving_planets(obs)
            self._steps = 3

        lobs = mod.refresh_local_obs(obs)
        self._update_fleet_trajectories(lobs.get("fleets", []))
        self._update_reinforcement_trajectories()
        comet_ids = obs.get("comet_planet_ids", [])
        under_attack = mod.get_planets_under_attack(
            lobs.get("mine", []), lobs.get("fleets", []),
            lobs.get("player", -2), obs.angular_velocity,
        )
        exhausted = set()

        if not lobs.get("targets", []):
            return []

        reinf_plans = mod.get_reinforcement_plans(lobs.get("mine", []), under_attack)
        for p, plan in reinf_plans.items():
            if any(r["target"].id == p.id and r["arrive_tick"] >= 0 for r in self._reinf_traj):
                continue
            ships_needed = plan["ships_needed"]
            needed_by = plan["needed_by_tick"]
            for p_np, _ in mod.get_closest_planets_to_target(lobs.get("mine", []), p):
                if p_np.id == p.id or p_np.id in exhausted:
                    continue
                avail = p_np.ships - sum(r["ships"] for r in self._reinf_traj if r["mine"].id == p_np.id)
                if p_np.id in under_attack:
                    avail = max(0, avail - sum(a["fleet"].ships for a in under_attack[p_np.id]["fleets"]))
                sent = max(mod.MIN_SHIPS_MINE_ATTACK, ships_needed)
                if avail < sent:
                    continue
                p_moving = p.id in self._moving_planets
                angle, arrive = mod.find_angle_to_planet(p_np, p, sent, obs.angular_velocity, moving=p_moving)
                if angle is None or arrive is None or arrive > needed_by:
                    continue
                moves.append([p_np.id, angle, sent])
                exhausted.add(p_np.id)
                self._reinf_traj.append({"mine": p_np, "target": p, "angle": angle, "ships": sent, "arrive_tick": arrive})
                break

        for m in sorted(lobs.get("mine", []), key=lambda p: p.ships, reverse=True):
            if m.id in exhausted or m.ships < mod.MIN_SHIPS_MINE_ATTACK:
                continue
            for m, t, _s in mod.get_candidate_targets(m, lobs.get("targets", []), comet_ids)[:3]:
                m_avail = m.ships
                if m.id in under_attack:
                    m_avail = max(0, m.ships - sum(a["fleet"].ships for a in under_attack[m.id]["fleets"]))
                if m_avail < mod.MIN_SHIPS_MINE_ATTACK:
                    continue
                nearest = mod.get_closest_planets_to_target(lobs.get("mine", []), t)
                safe_near = []
                for p, dist in nearest:
                    if p.id == m.id or p.id in exhausted:
                        continue
                    avail = p.ships
                    if p.id in under_attack:
                        avail = max(0, avail - sum(a["fleet"].ships for a in under_attack[p.id]["fleets"]))
                    if avail < mod.MIN_SHIPS_MINE_ATTACK:
                        continue
                    safe_near.append((p, dist, avail))
                owned = len(lobs.get("mine", []))
                total = len(lobs.get("planets", []))
                en_route = sum(f["ships"] for f in self._fleet_traj if f["target"].id == t.id)
                needed_now = t.ships + 1
                if t.owner != -1:
                    needed_now += 3 * t.production
                if owned < total * 0.75 and en_route >= needed_now:
                    continue
                base = max(mod.MIN_SHIPS_MINE_ATTACK, needed_now - en_route)
                if m_avail >= base:
                    t_moving = t.id in self._moving_planets
                    total_ships, angle, arrive = mod.predict_total_ships(m, t, obs.angular_velocity, base, m_avail, moving=t_moving)
                    if angle is not None and arrive is not None:
                        spd = mod.get_fleet_speed(max(1, total_ships))
                        if mod.sun_collision(m, spd, angle):
                            continue
                        moves.append([m.id, angle, total_ships])
                        exhausted.add(m.id)
                        self._fleet_traj.append({"mine": m, "target": t, "angle": angle, "ships": total_ships, "arrive_tick": arrive})
                elif m_avail < base and len(lobs.get("mine", [])) > 1 and t.ships >= mod.MIN_SHIPS_TARGET_COOP_ATTACK:
                    accum = m_avail
                    attacking = [{"planet": m, "ships": m_avail}]
                    coop_sent = False
                    for p, dist, p_avail in safe_near:
                        if coop_sent:
                            break
                        attacking.append({"planet": p, "ships": p_avail})
                        accum += p_avail
                        if len(attacking) > mod.COOP_PLANET_CAP:
                            break
                        if accum < base:
                            continue
                        remainder, planned = mod.plan_coop_attack(attacking, t, base, obs.angular_velocity, moving=False)
                        if remainder > 0:
                            continue
                        for move in planned:
                            self._fleet_traj.append({"mine": move[0], "target": t, "angle": move[1], "ships": move[2], "arrive_tick": move[3]})
                            exhausted.add(move[0].id)
                            move[0] = move[0].id
                            moves.append(move)
                        coop_sent = True
                        break
        return moves

    def _fill_moving_planets(self, obs):
        import kaggle_environments.envs.orbit_wars.orbit_wars as ow
        planets = [ow.Planet(*p) for p in obs.get("planets", [])]
        initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
        for p in planets:
            i = initial_by_id[p.id]
            if (p.x, p.y) != (i.x, i.y) and p.id not in self._moving_planets:
                self._moving_planets.append(p.id)

    def _update_fleet_trajectories(self, fleets):
        for ft in self._fleet_traj[:]:
            found = any(f.from_planet_id == ft["mine"].id and abs(f.angle - ft["angle"]) < 1e-6 for f in fleets)
            if found:
                ft["arrive_tick"] = max(0, ft["arrive_tick"] - 1)
            else:
                self._fleet_traj.remove(ft)

    def _update_reinforcement_trajectories(self):
        for rt in self._reinf_traj[:]:
            rt["arrive_tick"] -= 1
            if rt["arrive_tick"] <= 0:
                self._reinf_traj.remove(rt)


_instances: dict[int, Agent1000Score] = {}


def agent_1000_score(obs) -> list:
    player = obs.get("player", 0)
    if obs.get("step", -1) == 0 or player not in _instances:
        _instances[player] = Agent1000Score()
    return _instances[player](obs)
