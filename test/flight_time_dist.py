"""Profile the flight-time (hit_tick) distribution of launched fleets.

For every launch in the dataset we forward-simulate the fleet (bc_data's batched
swept-collision sim) and record, for fleets that actually reach a planet, how many
ticks the flight took. The goal is to choose a sensible MAX_FLY_TICKS: large
enough not to truncate genuine arrivals, small enough to keep the scan cheap.

We deliberately skip the feature encoder — flight time only needs the source
planet slot (resolved directly from p_id) and planet geometry, so this runs light
(no padded-encoder memory blow-up).

Usage:
    python -m test.flight_time_dist --episodes_dir episodes/2026-06-02 --n 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import src_jax.bc_data as bd

_GEOM_FIELDS = ("p_x", "p_y", "p_init_x", "p_init_y", "p_radius", "p_alive", "p_is_comet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-02")
    ap.add_argument("--n", type=int, default=300, help="max episodes to scan")
    ap.add_argument("--max_ticks", type=int, default=bd.MAX_FLY_TICKS)
    ap.add_argument("--chunk", type=int, default=20000, help="fleets per sim call")
    ap.add_argument("--bin_width", type=int, default=5,
                    help="bin width (ticks) for the full distribution table")
    args = ap.parse_args()

    print(f"episodes_dir={args.episodes_dir}  n={args.n}  max_ticks={args.max_ticks}")

    geom = {f: [] for f in _GEOM_FIELDS}
    pid_list, ang_list, step_list = [], [], []
    flat: list[tuple[int, int, float, float]] = []   # (global_ts, src_slot, angle, ships)

    t0 = time.perf_counter()
    n_ep = n_states = n_launch = n_unresolved_src = 0
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        if not isinstance(ep, dict) or "steps" not in ep:
            continue
        states, _players, actions = bd._collect_states_actions(ep)
        if not states:
            continue
        base = n_states
        for s in states:
            for f in _GEOM_FIELDS:
                geom[f].append(np.asarray(getattr(s, f)))
            pid_list.append(np.asarray(s.p_id))
            ang_list.append(float(s.angular_velocity))
            step_list.append(int(s.step))
        for i, acts in enumerate(actions):
            gts = base + i
            pid_to_slot = {int(p): k for k, p in enumerate(pid_list[gts]) if p >= 0}
            for act in acts:
                slot = pid_to_slot.get(int(act[0]), -1)
                if slot < 0:
                    n_unresolved_src += 1
                    continue
                flat.append((gts, slot, float(act[1]), float(act[2])))
        n_states += len(states)
        n_ep += 1
        if n_ep >= args.n:
            break

    n_launch = len(flat)
    print(f"scanned {n_ep} episodes, {n_states} states, {n_launch} launches "
          f"({n_unresolved_src} with unresolved source) in {time.perf_counter()-t0:.1f}s")
    if not flat:
        print("no launches found")
        return 1

    G = {f: np.stack(geom[f]) for f in _GEOM_FIELDS}          # [T, MAX_P]
    ang = np.asarray(ang_list, np.float32)
    step = np.asarray(step_list, np.int32)

    # Run the sim in fixed-size chunks (one XLA compile; last chunk padded).
    t0 = time.perf_counter()
    hk = np.empty(n_launch, np.int32)
    ht = np.empty(n_launch, np.int32)
    C = args.chunk
    for start in range(0, n_launch, C):
        sub = flat[start:start + C]
        pad = C - len(sub)
        if pad:
            sub = sub + [sub[-1]] * pad
        k, _slot, t = bd.simulate_episode_targets_jax(
            sub, p_x=G["p_x"], p_y=G["p_y"], p_init_x=G["p_init_x"],
            p_init_y=G["p_init_y"], p_radius=G["p_radius"], p_alive=G["p_alive"],
            p_is_comet=G["p_is_comet"], angular_velocity=ang, step=step,
            max_ticks=args.max_ticks)
        m = len(sub) - pad
        hk[start:start + m] = k[:m]
        ht[start:start + m] = t[:m]
    print(f"simulated {n_launch} fleets in {time.perf_counter()-t0:.1f}s "
          f"(chunk={C}, max_ticks={args.max_ticks})")

    # --- outcome breakdown ---
    kinds = {int(k): n for k, n in zip(*np.unique(hk, return_counts=True))}
    label = {0: "NONE", 1: "PLANET", 2: "SUN", 3: "OUT"}
    print("\noutcome breakdown:")
    for code in (1, 3, 2, 0):
        c = kinds.get(code, 0)
        print(f"  {label[code]:7s}: {c:8d}  ({100*c/n_launch:5.1f}%)")

    # --- flight-time distribution for PLANET hits ---
    ticks = ht[hk == int(bd.HitKind.PLANET)]
    if ticks.size == 0:
        print("\nno PLANET hits")
        return 1
    pcts = [50, 75, 90, 95, 99, 99.9, 100]
    pv = np.percentile(ticks, pcts)
    print(f"\nflight time (ticks) for {ticks.size} fleets that reached a planet:")
    print(f"  mean={ticks.mean():.1f}  std={ticks.std():.1f}  min={ticks.min()}  max={ticks.max()}")
    for p, v in zip(pcts, pv):
        print(f"  p{p:<5}= {v:.1f}")

    print("\ncumulative coverage (fraction of planet-bound fleets that arrive by tick T):")
    for thr in (10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300):
        frac = float(np.mean(ticks <= thr))
        print(f"  T<={thr:<4}: {100*frac:6.2f}%")

    print("\nhistogram (tick bucket -> count):")
    edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 99999]
    hist, _ = np.histogram(ticks, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * int(60 * c / max(hist.max(), 1))
        print(f"  [{lo:>4},{hi if hi<99999 else 'inf':>5}): {c:8d} {bar}")

    # --- full distribution: fixed-width bins across the whole range (PMF + CDF) ---
    bw = args.bin_width
    top = int(ticks.max())
    fedges = list(range(0, top + bw, bw))
    fhist, _ = np.histogram(ticks, bins=fedges)
    fmax = max(fhist.max(), 1)
    cum = 0
    print(f"\nfull flight-time distribution (bin width={bw} ticks, "
          f"PMF=share in bin, CDF=cumulative):")
    print(f"  {'bin':>11} {'count':>8} {'PMF':>7} {'CDF':>7}")
    for lo, hi, c in zip(fedges[:-1], fedges[1:], fhist):
        cum += int(c)
        pmf = 100 * c / ticks.size
        cdf = 100 * cum / ticks.size
        bar = "#" * int(50 * c / fmax)
        print(f"  [{lo:>4},{hi:>4}) {int(c):>8} {pmf:6.2f}% {cdf:6.2f}% {bar}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
