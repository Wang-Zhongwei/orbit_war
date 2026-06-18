"""Head-to-head: the SAME PPO checkpoint played deterministic vs stochastic.

Both sides are built from the live src_jax/jax_agent.build_agent (so both carry
the same code, incl. comet evac) — the only difference is `deterministic`. Seats
are alternated each round so neither sampling mode gets a fixed slot advantage.

    uv run python tools/det_vs_stoch.py \
        --ckpt artifacts/ppo/ppo_mixed_0615/ckpt_000900.pkl \
        --rounds 16 --replay results/h2h/det_vs_stoch_replay.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `src`/`src_jax`

from src.evaluation import run_match_raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("artifacts/ppo/ppo_mixed_0615/ckpt_000900.pkl"))
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--replay", type=Path, default=None,
                    help="Dump round-0's full env JSON here for viewing.")
    a = ap.parse_args()

    # Build BOTH agents from the live code — same params, differ only in sampling.
    from src_jax.jax_agent import build_agent
    ck = pickle.load(open(a.ckpt, "rb"))
    params = ck["params"]
    cfg = ck["config"]
    print(f"Building det + stoch agents from {a.ckpt} ...", flush=True)
    det   = build_agent(params, cfg, deterministic=True)
    stoch = build_agent(params, cfg, deterministic=False)

    # Aggregate win/draw/loss from DET's perspective.
    det_pts = 0.0
    n = 0
    print(f"Playing {a.rounds} rounds (seats alternate each round) ...", flush=True)
    print("round  seed  det_slot  det_reward  stoch_reward  outcome", flush=True)
    for r in range(a.rounds):
        seed = a.base_seed + r
        det_slot = r % 2                          # alternate seats for fairness
        agents = [det, stoch] if det_slot == 0 else [stoch, det]
        res = run_match_raw(agents, seed=seed, episode_steps=a.episode_steps)
        det_out = res["outcomes"][det_slot]       # 1=win, 0.5=draw, 0=loss
        det_pts += det_out
        n += 1
        label = {1.0: "DET win", 0.5: "draw", 0.0: "STOCH win"}.get(det_out, "?")
        print(f"{r:5d}  {seed:5d}  {det_slot:8d}  "
              f"{res['rewards'][det_slot]:10.2f}  "
              f"{res['rewards'][1 - det_slot]:12.2f}  {label}", flush=True)

        if a.replay is not None and r == 0:
            from kaggle_environments import make
            env = make("orbit_wars",
                       configuration={"seed": int(seed),
                                      "episodeSteps": int(a.episode_steps)},
                       debug=False)
            env.run(agents)
            a.replay.parent.mkdir(parents=True, exist_ok=True)
            a.replay.write_text(json.dumps(env.toJSON()))
            print(f"  (replay for round 0 -> {a.replay})", flush=True)

    wr = det_pts / max(n, 1)
    print("=" * 64)
    print(f"DETERMINISTIC win rate vs STOCHASTIC: {wr:.3f}  "
          f"(points {det_pts:.1f} / {n})")
    print(f"STOCHASTIC win rate: {1.0 - wr:.3f}")


if __name__ == "__main__":
    main()
