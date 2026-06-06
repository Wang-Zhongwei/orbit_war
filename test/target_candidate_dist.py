"""Profile where fleets are actually sent, and how well K-candidate selection covers them.

For every real launch we trajectory-simulate the fleet (bc_data's batched swept sim)
to find its TRUE target planet slot. We then ask, for that target:

  * owner category relative to the launching player (enemy / neutral / friendly),
  * its distance-rank from the source planet (1 = nearest non-comet planet),
  * whether the production `category` candidate selector (features._make_env_encoder)
    actually includes it in the source's K-candidate pool — for a sweep of K.

A target outside the candidate pool is silently relabelled as a no-op by the BC
pipeline, so coverage-vs-K is exactly the fraction of expert launches the policy
can ever learn to reproduce. This tells us a good K cutoff.

We encode one episode at a time (batched over its states) to avoid the padded-encoder
memory blow-up — intended to run on a compute node via SLURM.

Usage:
    python -m test.target_candidate_dist --episodes_dir episodes/2026-06-02 --n 200 \
        --k_sweep 6,8,10,12,16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from types import SimpleNamespace

from src_jax.config import TrainConfig
from src_jax.features import _make_env_encoder, _select_sources_and_candidates
from src_jax.physics import MAX_PLANETS
import src_jax.bc_data as bd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-02")
    ap.add_argument("--n", type=int, default=200, help="max episodes to scan")
    ap.add_argument("--k_sweep", default="6,8,10,12,16",
                    help="comma-separated candidate_count values to evaluate")
    ap.add_argument("--fast", action="store_true",
                    help="use the shared selector only (skip the feature tensor & path "
                         "blocking). Reports PURE in-pool coverage — the K-cutoff signal — "
                         "and is ~50-100x faster. Without --fast, coverage also reflects "
                         "path-blocking (the full encoder's cand_mask).")
    args = ap.parse_args()

    Ks = [int(x) for x in args.k_sweep.split(",")]
    print(f"jax devices: {jax.devices()}")
    print(f"episodes_dir={args.episodes_dir}  n={args.n}  k_sweep={Ks}")

    cfg = TrainConfig()
    cand_select = getattr(cfg.env, "cand_select", "category")
    print(f"cand_select={cand_select!r}  fast={args.fast}")

    # Fast path: a jit+vmap of the shared selector — produces exactly the encoder's
    # source/candidate slots, but skips the 30-dim feature tensor AND path blocking.
    # Coverage from this is pure in-pool membership (the K-cutoff signal).
    def _make_selector(K):
        sel = jax.jit(jax.vmap(
            lambda st, pl: _select_sources_and_candidates(st, pl, K, cand_select)))

        def run(batched, players_j, b):
            ss, sv, cs, cv = jax.tree_util.tree_map(np.asarray, sel(batched, players_j))
            return SimpleNamespace(
                src_slots=ss, src_valid=sv,
                src_ids=np.take_along_axis(np.asarray(b.p_id), ss, axis=1),
                cand_slots_safe=np.clip(cs, 0, MAX_PLANETS - 1),
                cand_mask=cv)   # pre-blocking pool membership (slot 0 = True)
        return run

    # Accumulators -----------------------------------------------------------
    owner_counts = {"enemy": 0, "neutral": 0, "friendly": 0}
    target_is_comet = 0
    ranks: list[int] = []                 # distance-rank of target among non-comet planets
    captured = {K: 0 for K in Ks}         # target landed inside the K-candidate pool
    miss_pool = {K: 0 for K in Ks}        # real non-comet target but beyond K candidates
    miss_comet = {K: 0 for K in Ks}       # target is a comet (structurally excluded)

    n_ep = n_states = n_launch = n_planet = n_unresolved_src = 0
    t0 = time.perf_counter()

    # One K (one encoder) resident at a time — holding all encoders compiled at
    # once OOMs XLA. The trajectory sim + owner/rank stats are K-independent, so
    # we accumulate those only on the first pass (Ks[0]).
    for kpass, K in enumerate(Ks):
        first = kpass == 0
        if args.fast:
            encode_fn = _make_selector(K)
        else:
            _enc = _make_env_encoder(K, cfg.env)
            encode_fn = lambda batched, players_j, b: \
                jax.tree_util.tree_map(np.asarray, _enc(batched, players_j))
        ep_seen = 0
        for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
            if not isinstance(ep, dict) or "steps" not in ep:
                continue
            states, players, actions = bd._collect_states_actions(ep)
            if not states:
                continue
            N = len(states)
            batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *states)
            players_j = jnp.asarray(np.asarray(players, np.int32))
            b = jax.tree_util.tree_map(np.asarray, batched)
            e = encode_fn(batched, players_j, b)

            # Flat launch list + trajectory sim (src_* are K-independent).
            flat, meta = [], []   # meta: (state_i, src_idx, player)
            for i in range(N):
                for act in actions[i]:
                    sidx = bd._resolve_src_idx(e.src_ids[i], e.src_valid[i], int(act[0]))
                    if sidx == -1:
                        if first:
                            n_unresolved_src += 1
                        continue
                    flat.append((i, int(e.src_slots[i][sidx]), float(act[1]), float(act[2])))
                    meta.append((i, sidx, int(players[i])))
            ep_seen += 1
            if first:
                n_states += N
                n_launch += len(flat)
            if not flat:
                if ep_seen >= args.n:
                    break
                continue

            hk, hslot, _ = bd.simulate_episode_targets_jax(
                flat, p_x=b.p_x, p_y=b.p_y, p_init_x=b.p_init_x, p_init_y=b.p_init_y,
                p_radius=b.p_radius, p_alive=b.p_alive, p_is_comet=b.p_is_comet,
                angular_velocity=b.angular_velocity, step=b.step)
            hk = np.asarray(hk); hslot = np.asarray(hslot)

            for nidx, (i, sidx, player) in enumerate(meta):
                if int(hk[nidx]) != int(bd.HitKind.PLANET):
                    continue
                tslot = int(hslot[nidx])
                src_slot = int(e.src_slots[i][sidx])
                is_comet = bool(b.p_is_comet[i][tslot])

                if first:
                    n_planet += 1
                    # --- owner category (relative to launching player) ---
                    towner = int(b.p_owner[i][tslot])
                    if towner == player:
                        owner_counts["friendly"] += 1
                    elif towner == -1:
                        owner_counts["neutral"] += 1
                    else:
                        owner_counts["enemy"] += 1
                    # --- distance-rank among alive non-comet non-source planets ---
                    if is_comet:
                        target_is_comet += 1
                    else:
                        sx, sy = b.p_x[i][src_slot], b.p_y[i][src_slot]
                        sel = b.p_alive[i] & (~b.p_is_comet[i])
                        sel[src_slot] = False
                        d = np.hypot(b.p_x[i] - sx, b.p_y[i] - sy)
                        rank = int(np.sum(sel & (d < d[tslot]))) + 1
                        ranks.append(rank)

                # --- coverage under the real category selector for this K ---
                cs_row = e.cand_slots_safe[i][sidx]
                cm_row = e.cand_mask[i][sidx]
                in_pool = bool(np.any((cs_row[1:] == tslot) & cm_row[1:]))
                if in_pool:
                    captured[K] += 1
                elif is_comet:
                    miss_comet[K] += 1
                else:
                    miss_pool[K] += 1

            if ep_seen >= args.n:
                break
        del encode_fn
        jax.clear_caches()
        print(f"  K={K} pass done ({time.perf_counter()-t0:.1f}s elapsed)")

    dt = time.perf_counter() - t0
    print(f"\nscanned {n_ep or ep_seen} episodes, {n_states} states, {n_launch} launches "
          f"({n_unresolved_src} unresolved source) in {dt:.1f}s")
    print(f"planet-bound launches (the population below): {n_planet}")
    if n_planet == 0:
        print("no planet-bound launches")
        return 1

    # --- target owner distribution ---
    print("\ntarget owner category (where fleets were actually sent):")
    for k in ("enemy", "neutral", "friendly"):
        c = owner_counts[k]
        print(f"  {k:9s}: {c:8d}  ({100*c/n_planet:5.1f}%)")
    print(f"  {'(comet)':9s}: {target_is_comet:8d}  ({100*target_is_comet/n_planet:5.1f}%)"
          f"   <- structurally excluded from candidates")

    # --- distance-rank distribution (pure-knn view) ---
    r = np.asarray(ranks)
    print(f"\ndistance-rank of target among non-comet planets "
          f"({r.size} non-comet targets):")
    print(f"  mean={r.mean():.2f}  median={np.median(r):.0f}  "
          f"p90={np.percentile(r,90):.0f}  p99={np.percentile(r,99):.0f}  max={r.max()}")
    print("  cumulative: fraction of non-comet targets within the N nearest planets")
    for thr in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20):
        print(f"    rank<={thr:<3}: {100*np.mean(r<=thr):6.2f}%")

    # --- category-selector coverage vs K (the real answer) ---
    mode = ("PURE in-pool (selection only, no path blocking)" if args.fast
            else "after path-blocking (the full encoder cand_mask)")
    print(f"\ncoverage vs K under the production {cand_select!r} selector — {mode}")
    print("(fraction of ALL planet-bound launches whose true target the K-pool keeps;")
    print(" a miss is silently relabelled no-op, so this caps learnable launches):")
    print(f"  {'K':>4} {'capt%':>7} {'miss_pool%':>11} {'miss_comet%':>12}")
    for K in Ks:
        cap = 100 * captured[K] / n_planet
        mp = 100 * miss_pool[K] / n_planet
        mc = 100 * miss_comet[K] / n_planet
        print(f"  {K:>4} {cap:7.2f} {mp:11.2f} {mc:12.2f}")
    print("\n(K-1 real slots per source; slot 0 is no-op. 'miss_comet' is the hard"
          " ceiling — comets are never candidates regardless of K.)")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
