"""Flight-time (hit_tick) distributions split by the target planet's OWNER.

For every launch we forward-simulate the fleet (bc_data's batched swept-collision
sim) to find the planet it reaches and how many ticks the flight took, then bucket
that flight time by what the launcher was sending the fleet AT:

    1. NEUTRAL  target  (owner == -1 at launch)
    2. FRIENDLY target  (owner == launcher)
    3. ENEMY    target  (owner == some other player)
    4. ENEMY -> CAPTURE : enemy-targeted launches where the target planet's owner
       actually flips to the launcher shortly after the fleet arrives (ground truth
       from the replay, not a combat model).

Owner is absolute (-1 neutral, 0/1 player) and identical across both agents'
observations, so capture is resolved by looking up the target planet id's owner in
the replay state at/just after the simulated arrival tick.

Usage:
    python -m test.flight_time_by_owner --episodes_dir episodes/2026-06-09 --n 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import src_jax.bc_data as bd

_GEOM = ("p_x", "p_y", "p_init_x", "p_init_y", "p_radius", "p_alive", "p_is_comet")


def _print_dist(name: str, ticks: np.ndarray, bw: int, max_ticks: int) -> None:
    """Print summary + full fixed-width-bin PMF/CDF for one flight-time array."""
    print(f"\n{'='*72}\n{name}: {ticks.size} fleets")
    if ticks.size == 0:
        print("  (none)")
        return
    pcts = [50, 75, 90, 95, 99, 100]
    pv = np.percentile(ticks, pcts)
    print(f"  mean={ticks.mean():.1f}  std={ticks.std():.1f}  "
          f"min={int(ticks.min())}  max={int(ticks.max())}")
    print("  " + "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, pv)))
    top = min(int(ticks.max()), max_ticks)
    fedges = list(range(0, top + bw, bw))
    fhist, _ = np.histogram(ticks, bins=fedges)
    fmax = max(fhist.max(), 1)
    cum = 0
    print(f"  {'bin':>11} {'count':>8} {'PMF':>7} {'CDF':>7}")
    for lo, hi, c in zip(fedges[:-1], fedges[1:], fhist):
        cum += int(c)
        pmf = 100 * c / ticks.size
        cdf = 100 * cum / ticks.size
        bar = "#" * int(46 * c / fmax)
        print(f"  [{lo:>4},{hi:>4}) {int(c):>8} {pmf:6.2f}% {cdf:6.2f}% {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-09")
    ap.add_argument("--n", type=int, default=300, help="max episodes to scan")
    ap.add_argument("--max_ticks", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=20000, help="fleets per sim call")
    ap.add_argument("--bin_width", type=int, default=5,
                    help="bin width (ticks) for the full distribution tables")
    ap.add_argument("--capture_window", type=int, default=3,
                    help="ticks after arrival to allow the ownership flip to land")
    args = ap.parse_args()

    print(f"episodes_dir={args.episodes_dir}  n={args.n}  max_ticks={args.max_ticks}  "
          f"bin_width={args.bin_width}  capture_window={args.capture_window}")

    geom = {f: [] for f in _GEOM}
    pid_list, own_list, ang_list, step_list = [], [], [], []
    ep_of_state: list[int] = []                       # episode index per global state
    flat: list[tuple[int, int, float, float]] = []    # (gts, src_slot, angle, ships)
    launch_step: list[int] = []
    launch_player: list[int] = []
    ep_step_to_gts: list[dict[int, int]] = []         # per-episode {step -> a gts}

    t0 = time.perf_counter()
    n_ep = n_states = n_unresolved = 0
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        if not isinstance(ep, dict) or "steps" not in ep:
            continue
        states, players, actions = bd._collect_states_actions(ep)
        if not states:
            continue
        base = n_states
        for s in states:
            for f in _GEOM:
                geom[f].append(np.asarray(getattr(s, f)))
            pid_list.append(np.asarray(s.p_id))
            own_list.append(np.asarray(s.p_owner))
            ang_list.append(float(s.angular_velocity))
            step_list.append(int(s.step))
            ep_of_state.append(n_ep)
        # first global state index seen for each game tick (owners are shared across agents)
        step_map: dict[int, int] = {}
        for off, s in enumerate(states):
            step_map.setdefault(int(s.step), base + off)
        ep_step_to_gts.append(step_map)
        for i, acts in enumerate(actions):
            gts = base + i
            pid_to_slot = {int(p): k for k, p in enumerate(pid_list[gts]) if p >= 0}
            for act in acts:
                slot = pid_to_slot.get(int(act[0]), -1)
                if slot < 0:
                    n_unresolved += 1
                    continue
                flat.append((gts, slot, float(act[1]), float(act[2])))
                launch_step.append(int(step_list[gts]))
                launch_player.append(int(players[i]))
        n_states += len(states)
        n_ep += 1
        if n_ep >= args.n:
            break

    n_launch = len(flat)
    print(f"scanned {n_ep} episodes, {n_states} states, {n_launch} launches "
          f"({n_unresolved} unresolved src) in {time.perf_counter()-t0:.1f}s")
    if not flat:
        print("no launches")
        return 1

    G = {f: np.stack(geom[f]) for f in _GEOM}
    OWN = np.stack(own_list)
    PID = np.stack(pid_list)
    ang = np.asarray(ang_list, np.float32)
    step = np.asarray(step_list, np.int32)

    # --- simulate every fleet ---
    t0 = time.perf_counter()
    hk = np.empty(n_launch, np.int32)
    hslot = np.empty(n_launch, np.int32)
    ht = np.empty(n_launch, np.int32)
    C = args.chunk
    for s0 in range(0, n_launch, C):
        sub = flat[s0:s0 + C]
        pad = C - len(sub)
        if pad:
            sub = sub + [sub[-1]] * pad
        k, sl, t = bd.simulate_episode_targets_jax(
            sub, p_x=G["p_x"], p_y=G["p_y"], p_init_x=G["p_init_x"],
            p_init_y=G["p_init_y"], p_radius=G["p_radius"], p_alive=G["p_alive"],
            p_is_comet=G["p_is_comet"], angular_velocity=ang, step=step,
            max_ticks=args.max_ticks)
        m = len(sub) - pad
        hk[s0:s0 + m] = np.asarray(k)[:m]
        hslot[s0:s0 + m] = np.asarray(sl)[:m]
        ht[s0:s0 + m] = np.asarray(t)[:m]
    print(f"simulated {n_launch} fleets in {time.perf_counter()-t0:.1f}s")

    ls = np.asarray(launch_step)
    lp = np.asarray(launch_player)
    is_planet = hk == int(bd.HitKind.PLANET)

    # --- owner category (0 enemy, 1 neutral, 2 friendly) + capture flag ---
    cat = np.full(n_launch, -1, np.int32)
    captured = np.zeros(n_launch, bool)
    n_pb = 0
    for n in range(n_launch):
        if not is_planet[n]:
            continue
        n_pb += 1
        gts, _slot, _a, _sh = flat[n]
        tslot = int(hslot[n])
        towner = int(OWN[gts][tslot])
        launcher = int(lp[n])
        if towner == launcher:
            cat[n] = 2
        elif towner == -1:
            cat[n] = 1
        else:
            cat[n] = 0
            # capture: does the target planet's owner flip to launcher near arrival?
            tpid = int(PID[gts][tslot])
            arr = int(ls[n]) + int(ht[n])
            ep = ep_of_state[gts]
            smap = ep_step_to_gts[ep]
            for st in range(arr, arr + args.capture_window + 1):
                g2 = smap.get(st)
                if g2 is None:
                    continue
                row = PID[g2]
                hit = np.where(row == tpid)[0]
                if hit.size and int(OWN[g2][hit[0]]) == launcher:
                    captured[n] = True
                    break

    print(f"\nplanet-bound launches: {n_pb} / {n_launch} "
          f"({100*n_pb/n_launch:.1f}%)")
    enemy_mask = is_planet & (cat == 0)
    print(f"  neutral={int((is_planet&(cat==1)).sum())}  "
          f"friendly={int((is_planet&(cat==2)).sum())}  "
          f"enemy={int(enemy_mask.sum())}  "
          f"enemy->capture={int((enemy_mask&captured).sum())} "
          f"({100*captured[enemy_mask].mean():.1f}% of enemy)")

    bw, mt = args.bin_width, args.max_ticks
    _print_dist("1. NEUTRAL target",  ht[is_planet & (cat == 1)], bw, mt)
    _print_dist("2. FRIENDLY target", ht[is_planet & (cat == 2)], bw, mt)
    _print_dist("3. ENEMY target",    ht[enemy_mask], bw, mt)
    _print_dist("4. ENEMY -> CAPTURE", ht[enemy_mask & captured], bw, mt)
    _print_dist("ALL planet-bound",   ht[is_planet], bw, mt)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
