"""Correctness test for the will_be_mine candidate feature (plane 25).

will_be_mine is the exact counterfactual: if the source launches its full
garrison now (landing at ceil(eta)), is the target mine at the end of the
MAX_TICKS_IN outcome sim?

This test re-implements the whole outcome sim independently in numpy from the
raw EnvState spec — fleet flight, landing, per-tick arrival grouping,
production, combat resolution — and the per-pair counterfactual injection.
Validation is two-staged:

  1. The numpy baseline replay (no injection) must agree with the encoder's
     own would_be_mine_final plane [24] — this proves the reference itself
     mirrors the sim before we trust it for the counterfactual.
  2. The numpy counterfactual must agree with the will_be_mine plane [25].

The arrival tick is recovered from the normalized ETA feature (plane 7), which
roundtrips through float32 — when eta sits within EPS of an integer (always
true for rotating-intercept targets, whose eta is an exact tick), ceil() is
ambiguous by one tick. Those cells are checked against BOTH candidate arrival
ticks and pass if either reproduces the feature.

Run on a compute node (the padded encoder is heavy for the login node):
    python -m test.test_will_be_mine --episodes_dir episodes/2026-06-09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from src_jax.config import TrainConfig
from src_jax.constants import CENTER, ROTATION_RADIUS_LIMIT
from src_jax.physics import _fleet_speed
from src_jax.features import (
    MAX_TICKS_IN, _make_env_encoder, candidate_feature_dim,
)
import src_jax.bc_data as bd

ETA_EPS = 1e-3   # eta this close to an integer -> arrival tick ambiguous (check both)


def _np_arrivals(s, T: int) -> np.ndarray:
    """Per-tick arrival totals by player: [T, MAX_PLANETS, 4] (numpy ref)."""
    p_x, p_y = np.asarray(s.p_x), np.asarray(s.p_y)
    p_r      = np.asarray(s.p_radius)
    f_ships  = np.asarray(s.f_ships, dtype=np.float64)
    spd      = np.asarray(_fleet_speed(jnp.asarray(f_ships, jnp.float32)))
    f_vx     = spd * np.cos(np.asarray(s.f_angle))
    f_vy     = spd * np.sin(np.asarray(s.f_angle))
    f_alive  = np.asarray(s.f_alive)
    f_owner  = np.asarray(s.f_owner)
    w        = float(np.asarray(s.angular_velocity))

    orb_dx, orb_dy = p_x - CENTER, p_y - CENTER
    rotating = (np.hypot(orb_dx, orb_dy) + p_r) < ROTATION_RADIUS_LIMIT

    n_p, n_f = p_x.shape[0], f_ships.shape[0]
    cs = np.zeros((T, n_p, 4), dtype=np.float64)
    landed = np.zeros(n_f, dtype=bool)
    for t in range(1, T + 1):
        c, sn = np.cos(w * t), np.sin(w * t)
        px = np.where(rotating, CENTER + c * orb_dx - sn * orb_dy, p_x)
        py = np.where(rotating, CENTER + sn * orb_dx + c * orb_dy, p_y)
        fx = np.asarray(s.f_x) + t * f_vx
        fy = np.asarray(s.f_y) + t * f_vy
        d2 = (px[:, None] - fx[None, :]) ** 2 + (py[:, None] - fy[None, :]) ** 2
        hit = (d2 < (p_r ** 2)[:, None]) & ~landed[None, :]
        alive_hit = hit & f_alive[None, :]
        for pid in range(4):
            cs[t - 1, :, pid] = (alive_hit & (f_owner[None, :] == pid)) @ f_ships
        landed |= hit.any(axis=0)
    return cs


def _np_replay(g0, o0, prod, cs_p, inject_t=None, inject_ships=0.0, inject_pid=None):
    """Replay one planet's production+combat recurrence; returns final owner.

    cs_p: [T, 4] arrival totals at this planet. inject_t is 1-based.
    """
    g, o, po = float(g0), int(o0), int(o0)
    for t in range(1, cs_p.shape[0] + 1):
        cs = cs_p[t - 1].copy()
        if inject_t is not None and t == inject_t:
            cs[inject_pid] += inject_ships
        prod_owner = o if g > 0 else po
        g += prod * (prod_owner != -1)
        if (cs > 0).any():
            top = int(np.argmax(cs))
            srt = np.sort(cs)
            survivor = srt[-1] - srt[-2]
            if survivor > 0:
                g = g + survivor if top == o else g - survivor
                if top != o and g < 0:
                    o = top
                g = abs(g)
        po = o if g > 0 else po
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", default="episodes/2026-06-09")
    ap.add_argument("--n_states", type=int, default=24)
    args = ap.parse_args()

    assert candidate_feature_dim() == 26

    cfg = TrainConfig()
    K = cfg.env.candidate_count
    episode_steps = float(cfg.env.episode_steps)
    enc_fn = _make_env_encoder(K, cfg.env)

    states, players = [], []
    for _name, ep in bd._iter_episode_jsons(Path(args.episodes_dir)):
        st, pl, _ = bd._collect_states_actions(ep)
        # spread over the episode for early/mid/late coverage
        idx = np.linspace(0, len(st) - 1, min(8, len(st))).astype(int)
        states += [st[i] for i in idx]
        players += [pl[i] for i in idx]
        if len(states) >= args.n_states:
            break
    states, players = states[:args.n_states], players[:args.n_states]
    assert states, f"no states in {args.episodes_dir}"

    batched = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.stack(xs)), *states)
    enc = enc_fn(batched, jnp.array(players, dtype=jnp.int32))

    cf         = np.asarray(enc.cand_feats)            # [N, MAX_SRC, K, 26]
    cand_mask  = np.asarray(enc.cand_mask)
    src_valid  = np.asarray(enc.src_valid).astype(bool)
    src_slots  = np.asarray(enc.src_slots)
    cand_slots = np.asarray(enc.cand_slots_safe)

    n_base = n_wbm = n_skip = 0
    base_mism = wbm_mism = 0
    for n, (s, player) in enumerate(zip(states, players)):
        cs = _np_arrivals(s, MAX_TICKS_IN)
        p_ships = np.asarray(s.p_ships, dtype=np.float64)
        p_alive = np.asarray(s.p_alive)
        p_owner = np.asarray(s.p_owner)
        p_prod  = np.asarray(s.p_production, dtype=np.float64)
        src_g   = np.round(p_ships[src_slots[n]])

        # stage 1: baseline replay vs would_be_mine_final (plane 24)
        base_owner = {}
        for p in np.unique(cand_slots[n][src_valid[n]]):
            g0 = p_ships[p] if p_alive[p] else 0.0
            base_owner[p] = _np_replay(g0, p_owner[p], p_prod[p], cs[:, p])
        for i in np.where(src_valid[n])[0]:
            for k in range(1, K):
                if not cand_mask[n, i, k]:
                    continue
                p = cand_slots[n, i, k]
                ref = float(base_owner[p] == player)
                n_base += 1
                base_mism += int(ref != cf[n, i, k, 24])

        # stage 2: counterfactual vs will_be_mine (plane 25)
        for i in np.where(src_valid[n])[0]:
            for k in range(1, K):
                if not cand_mask[n, i, k]:
                    continue
                eta = float(cf[n, i, k, 7]) * episode_steps
                ambiguous = abs(eta - round(eta)) < ETA_EPS
                Ts = ({int(round(eta)), int(round(eta)) + 1} if ambiguous
                      else {int(np.ceil(eta))})
                Ts = {int(np.clip(T, 1, MAX_TICKS_IN)) for T in Ts}
                p = cand_slots[n, i, k]
                g0 = p_ships[p] if p_alive[p] else 0.0
                refs = {
                    float(_np_replay(g0, p_owner[p], p_prod[p], cs[:, p],
                                     inject_t=T, inject_ships=src_g[i],
                                     inject_pid=int(player)) == player)
                    for T in Ts
                }
                n_wbm += 1
                n_skip += int(ambiguous)
                wbm_mism += int(cf[n, i, k, 25] not in refs)

    print(f"baseline replay vs plane24: {base_mism}/{n_base} mismatches")
    print(f"will_be_mine  vs plane25:  {wbm_mism}/{n_wbm} mismatches ({n_skip} eta-boundary, either-tick accepted)")
    assert n_base > 200 and n_wbm > 200, "too few checked cells"
    assert base_mism / n_base < 0.005, "numpy reference disagrees with sim baseline"
    assert wbm_mism / n_wbm < 0.005, "will_be_mine disagrees with reference counterfactual"
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
