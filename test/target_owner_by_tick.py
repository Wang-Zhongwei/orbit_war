"""Measure how the expert target-owner mix (enemy/neutral/friendly) shifts over a game.

For every launch we trajectory-simulate the fleet to find its true target planet,
then bucket by the game tick at which the launch happened and report the owner mix
per bucket. This tells us WHEN the early neutral (expansion) phase gives way to
enemy/friendly targeting — i.e. how fast the candidate-quota schedule should move
and where the transition should land.

No feature encoder is needed (owner mix only needs the true target + launcher),
so this runs light. Source slots are resolved directly from planet ids.

Usage:
    python -m test.target_owner_by_tick --episodes_dir episodes/2026-06-02 --n 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import src_jax.bc_data as bd

_GEOM = ("p_x", "p_y", "p_init_x", "p_init_y", "p_radius", "p_alive", "p_is_comet")
_EDGES = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-02")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    geom = {f: [] for f in _GEOM}
    pid_list, own_list, ang_list, step_list = [], [], [], []
    flat: list[tuple[int, int, float, float]] = []   # (gts, src_slot, angle, ships)
    launch_step: list[int] = []
    launch_player: list[int] = []

    t0 = time.perf_counter()
    n_ep = n_states = 0
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
        for i, acts in enumerate(actions):
            gts = base + i
            pid_to_slot = {int(p): k for k, p in enumerate(pid_list[gts]) if p >= 0}
            for act in acts:
                slot = pid_to_slot.get(int(act[0]), -1)
                if slot < 0:
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
          f"in {time.perf_counter()-t0:.1f}s")
    if not flat:
        print("no launches")
        return 1

    G = {f: np.stack(geom[f]) for f in _GEOM}
    OWN = np.stack(own_list)
    ang = np.asarray(ang_list, np.float32)
    step = np.asarray(step_list, np.int32)

    hk = np.empty(n_launch, np.int32)
    hslot = np.empty(n_launch, np.int32)
    C = args.chunk
    for s0 in range(0, n_launch, C):
        sub = flat[s0:s0 + C]
        pad = C - len(sub)
        if pad:
            sub = sub + [sub[-1]] * pad
        k, sl, _ = bd.simulate_episode_targets_jax(
            sub, p_x=G["p_x"], p_y=G["p_y"], p_init_x=G["p_init_x"],
            p_init_y=G["p_init_y"], p_radius=G["p_radius"], p_alive=G["p_alive"],
            p_is_comet=G["p_is_comet"], angular_velocity=ang, step=step)
        m = len(sub) - pad
        hk[s0:s0 + m] = np.asarray(k)[:m]
        hslot[s0:s0 + m] = np.asarray(sl)[:m]

    ls = np.asarray(launch_step)
    lp = np.asarray(launch_player)
    is_planet = hk == int(bd.HitKind.PLANET)

    # owner category per planet-bound launch
    cat = np.full(n_launch, -1, np.int32)   # 0 enemy, 1 neutral, 2 friendly
    for n in range(n_launch):
        if not is_planet[n]:
            continue
        gts, _slot, _a, _sh = flat[n]
        towner = int(OWN[gts][hslot[n]])
        if towner == lp[n]:
            cat[n] = 2
        elif towner == -1:
            cat[n] = 1
        else:
            cat[n] = 0

    print("\ntarget-owner mix by launch tick (planet-bound launches only):")
    print(f"  {'tick bucket':>14} {'n':>7}  {'enemy':>7} {'neutral':>8} {'friendly':>9}")
    for lo, hi in zip(_EDGES[:-1], _EDGES[1:]):
        sel = is_planet & (ls >= lo) & (ls < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        c = cat[sel]
        e = float(np.mean(c == 0)); nt = float(np.mean(c == 1)); fr = float(np.mean(c == 2))
        print(f"  [{lo:>3},{hi:>3}) {n:>9}  {100*e:6.1f}% {100*nt:7.1f}% {100*fr:8.1f}%")

    # cumulative neutral share up to tick T (to see when expansion tapers off)
    print("\nneutral share of planet-bound launches, early vs late:")
    for thr in (20, 30, 40, 50, 75, 100):
        early = is_planet & (ls < thr)
        late = is_planet & (ls >= thr)
        ne = 100 * float(np.mean(cat[early] == 1)) if early.any() else 0.0
        nl = 100 * float(np.mean(cat[late] == 1)) if late.any() else 0.0
        print(f"  tick<{thr:<4}: neutral={ne:5.1f}%   |  tick>={thr:<4}: neutral={nl:5.1f}%")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
