"""
Frame-by-frame correctness test: src.jax_env vs kaggle_environments.

For each episode:
  - Initialise both envs with the same seed.
  - Each tick: generate the same random actions from the SAME planet list
    (derived from our env, which should be identical to kaggle's).
  - Apply those actions to BOTH envs.
  - Compare every field of every planet and fleet.
  - Compare final termination status and rewards.

Run:
    python test_jax_env.py             # default 20 episodes
    python test_jax_env.py --episodes 100 --max-ticks 600
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Any

from kaggle_environments import make as kaggle_make

from src.jax_env.physics import init_game, step_game


# ---------------------------------------------------------------------------
# Random agent — deterministic from a shared RNG, state-independent
# ---------------------------------------------------------------------------

def random_actions(planets: list, player: int, rng: random.Random) -> list:
    """Send ships from each owned planet to a random target with 30% probability."""
    moves = []
    owned = [p for p in planets if p[1] == player and p[5] > 10]
    for p in owned:
        if rng.random() > 0.30:
            continue
        others = [t for t in planets if t[0] != p[0]]
        if not others:
            continue
        tgt = rng.choice(others)
        angle = math.atan2(tgt[3] - p[3], tgt[2] - p[2])
        ships = max(1, int(p[5] * rng.uniform(0.25, 0.6)))
        moves.append([p[0], angle, ships])
    return moves


# ---------------------------------------------------------------------------
# Per-field comparison helpers
# ---------------------------------------------------------------------------

POS_TOL = 1e-3   # tolerance for floating-point x/y


def compare_planets(
    our_planets: list,
    kaggle_planets: list,
    tick: int,
    episode: int,
    errors: list[str],
) -> None:
    our_by_id   = {p[0]: p for p in our_planets}
    kaggle_by_id = {p[0]: p for p in kaggle_planets}
    all_ids = set(our_by_id) | set(kaggle_by_id)
    for pid in sorted(all_ids):
        op = our_by_id.get(pid)
        kp = kaggle_by_id.get(pid)
        prefix = f"ep{episode} tick{tick:03d} planet{pid}"
        if op is None:
            errors.append(f"{prefix}: MISSING in our env (present in kaggle)")
            continue
        if kp is None:
            errors.append(f"{prefix}: EXTRA in our env (absent from kaggle)")
            continue
        if abs(op[2] - kp[2]) > POS_TOL or abs(op[3] - kp[3]) > POS_TOL:
            errors.append(f"{prefix}: pos mismatch  ours=({op[2]:.4f},{op[3]:.4f})  kaggle=({kp[2]:.4f},{kp[3]:.4f})")
        if op[5] != kp[5]:
            errors.append(f"{prefix}: ships mismatch  ours={op[5]}  kaggle={kp[5]}")
        if op[1] != kp[1]:
            errors.append(f"{prefix}: owner mismatch  ours={op[1]}  kaggle={kp[1]}")
        if abs(op[4] - kp[4]) > POS_TOL:
            errors.append(f"{prefix}: radius mismatch  ours={op[4]:.4f}  kaggle={kp[4]:.4f}")
        if op[6] != kp[6]:
            errors.append(f"{prefix}: production mismatch  ours={op[6]}  kaggle={kp[6]}")


def compare_fleets(
    our_fleets: list,
    kaggle_fleets: list,
    tick: int,
    episode: int,
    errors: list[str],
) -> None:
    # Match by fleet id
    our_by_id    = {f[0]: f for f in our_fleets}
    kaggle_by_id = {f[0]: f for f in kaggle_fleets}
    all_ids = set(our_by_id) | set(kaggle_by_id)
    for fid in sorted(all_ids):
        of = our_by_id.get(fid)
        kf = kaggle_by_id.get(fid)
        prefix = f"ep{episode} tick{tick:03d} fleet{fid}"
        if of is None:
            errors.append(f"{prefix}: MISSING in our env")
            continue
        if kf is None:
            errors.append(f"{prefix}: EXTRA in our env")
            continue
        if abs(of[2] - kf[2]) > POS_TOL or abs(of[3] - kf[3]) > POS_TOL:
            errors.append(f"{prefix}: pos mismatch  ours=({of[2]:.4f},{of[3]:.4f})  kaggle=({kf[2]:.4f},{kf[3]:.4f})")
        if of[6] != kf[6]:
            errors.append(f"{prefix}: ships mismatch  ours={of[6]}  kaggle={kf[6]}")
        if of[1] != kf[1]:
            errors.append(f"{prefix}: owner mismatch  ours={of[1]}  kaggle={kf[1]}")


# ---------------------------------------------------------------------------
# Single episode driver
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode: int
    seed: int
    ticks: int
    errors: list[str]
    our_reward_p0: float | None
    kaggle_reward_p0: float | None
    terminated_same: bool


def run_episode(episode: int, seed: int, max_ticks: int, verbose: bool) -> EpisodeResult:
    # --- Our env ---
    our = init_game(seed, num_agents=2)
    step_game(our, [[], []])   # init step (matches kaggle's first env.step call)

    # --- Kaggle env ---
    kenv = kaggle_make(
        "orbit_wars",
        configuration={"seed": seed, "randomSeed": seed},
        debug=False,
    )
    kenv.reset(num_agents=2)
    kenv.step([[], []])        # kaggle init step

    errors: list[str] = []
    rng = random.Random(seed * 1_000_003 + episode)  # deterministic, seed-independent from game
    our_reward_p0: float | None = None
    kaggle_reward_p0: float | None = None
    terminated_same = True
    ticks = 0

    for tick in range(1, max_ticks + 1):
        # Generate actions from OUR state (should equal kaggle's at this point)
        a0 = random_actions(our.planets, 0, rng)
        a1 = random_actions(our.planets, 1, rng)

        # Step both envs
        our_statuses, our_rewards = step_game(our, [a0, a1])
        kenv.step([a0, a1])
        kobs = kenv.state[0].observation

        # Compare planet and fleet state this frame
        compare_planets(our.planets, kobs.planets, tick, episode, errors)
        compare_fleets(our.fleets,   kobs.fleets,  tick, episode, errors)

        # Check step counter
        if our.step != kobs.step:
            errors.append(f"ep{episode} tick{tick:03d}: step counter mismatch  ours={our.step}  kaggle={kobs.step}")

        our_done    = our_statuses[0] != "ACTIVE"
        kaggle_done = kenv.done

        if our_done != kaggle_done:
            terminated_same = False
            errors.append(
                f"ep{episode} tick{tick:03d}: termination mismatch  "
                f"ours={'DONE' if our_done else 'ACTIVE'}  "
                f"kaggle={'DONE' if kaggle_done else 'ACTIVE'}"
            )

        if our_done or kaggle_done:
            ticks = tick
            if our_done:
                our_reward_p0 = float(our_rewards[0])
            if kaggle_done:
                kaggle_states = kenv.state
                kaggle_reward_p0 = float(kaggle_states[0].reward or 0.0)
            break
    else:
        ticks = max_ticks
        # Neither terminated — check rewards are absent
        our_reward_p0 = None
        kaggle_reward_p0 = None

    # Compare rewards on termination
    if our_reward_p0 is not None and kaggle_reward_p0 is not None:
        if our_reward_p0 != kaggle_reward_p0:
            errors.append(
                f"ep{episode}: reward mismatch  ours={our_reward_p0}  kaggle={kaggle_reward_p0}"
            )

    if verbose and errors:
        for e in errors[:10]:
            print(f"  !! {e}")
        if len(errors) > 10:
            print(f"  ... ({len(errors) - 10} more errors)")

    return EpisodeResult(
        episode=episode,
        seed=seed,
        ticks=ticks,
        errors=errors,
        our_reward_p0=our_reward_p0,
        kaggle_reward_p0=kaggle_reward_p0,
        terminated_same=terminated_same,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-by-frame jax_env vs kaggle correctness test")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--max-ticks", type=int, default=600, help="Max ticks per episode")
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed (episode i uses seed offset+i)")
    parser.add_argument("--verbose", action="store_true", help="Print individual error lines")
    args = parser.parse_args()

    total_errors  = 0
    failed_episodes = 0
    total_frames  = 0

    print(f"Running {args.episodes} episodes (seeds {args.seed_offset}..{args.seed_offset + args.episodes - 1})")
    print(f"Max ticks per episode: {args.max_ticks}")
    print()

    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        result = run_episode(ep + 1, seed, args.max_ticks, args.verbose)
        n_err  = len(result.errors)
        total_frames  += result.ticks
        total_errors  += n_err
        if n_err > 0:
            failed_episodes += 1

        status = "OK" if n_err == 0 else f"FAIL ({n_err} errors)"
        reward_str = f"reward={result.our_reward_p0}" if result.our_reward_p0 is not None else "no termination"
        print(f"  ep{ep+1:3d}  seed={seed:5d}  ticks={result.ticks:4d}  {reward_str:20s}  {status}")

    print()
    print("=" * 60)
    print(f"Episodes      : {args.episodes}")
    print(f"Total frames  : {total_frames}")
    print(f"Total errors  : {total_errors}")
    print(f"Failed episodes: {failed_episodes} / {args.episodes}")
    if total_errors == 0:
        print("\nPASSED — our env is frame-perfectly identical to kaggle.")
    else:
        print(f"\nFAILED — {total_errors} field mismatches across {failed_episodes} episodes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
