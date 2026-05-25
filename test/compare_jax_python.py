"""
Trajectory comparison: src_jax (JAX physics) vs kaggle_environments.

Two random agents play against each other.  Both environments are initialised
with the same seed and receive the same random action sequence every tick.
Every planet and fleet field is compared frame-by-frame.

Usage
-----
    python compare_jax_python.py                      # 5 episodes, 500 ticks max
    python compare_jax_python.py --episodes 20 --verbose
"""

from __future__ import annotations

import argparse
import jax
import jax.numpy as jnp
import math
import random
import sys

from kaggle_environments import make as kaggle_make

from src_jax.jax_physics import (
    MAX_ACTIONS_PA,
    MAX_FLEETS,
    MAX_PLANETS,
    init_jax_state,
    jax_step_jit,
)

# --------------------------------------------------------------------------- #
# Tolerances
# --------------------------------------------------------------------------- #

POS_TOL = 1e-2   # float32 accumulates small errors per tick

_EMPTY_ACTIONS = jnp.full((2, MAX_ACTIONS_PA, 3), -1.0, dtype=jnp.float32)


# --------------------------------------------------------------------------- #
# Helpers: extract state from JaxState
# --------------------------------------------------------------------------- #

def jax_planets(js) -> list[list]:
    """Return [[id, owner, x, y, radius, ships, production], ...] for alive planets."""
    js_np = jax.device_get(js)
    result = []
    for i in range(MAX_PLANETS):
        if not bool(js_np.p_alive[i]):
            continue
        result.append([
            int(js_np.p_id[i]),
            int(js_np.p_owner[i]),
            float(js_np.p_x[i]),
            float(js_np.p_y[i]),
            float(js_np.p_radius[i]),
            int(round(float(js_np.p_ships[i]))),
            int(round(float(js_np.p_production[i]))),
        ])
    return result


def jax_fleets(js) -> list[list]:
    """Return [[id, owner, x, y, angle, from_planet_id, ships], ...] for alive fleets.

    fleet id is assigned sequentially in insertion order (slot index).
    """
    js_np = jax.device_get(js)
    result = []
    fid = 0
    for i in range(MAX_FLEETS):
        if not bool(js_np.f_alive[i]):
            continue
        result.append([
            fid,
            int(js_np.f_owner[i]),
            float(js_np.f_x[i]),
            float(js_np.f_y[i]),
            float(js_np.f_angle[i]),
            -1,   # from_planet_id not tracked in JaxState
            int(round(float(js_np.f_ships[i]))),
        ])
        fid += 1
    return result


# --------------------------------------------------------------------------- #
# Random agent (deterministic from shared RNG)
# --------------------------------------------------------------------------- #

def random_actions(planets: list, player: int, rng: random.Random) -> list:
    """Send ships from each owned planet with 30% probability."""
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


def encode_actions_jax(actions_per_agent: list[list[list]]) -> jnp.ndarray:
    """Convert Python action lists → float32[num_agents, MAX_ACTIONS_PA, 3]."""
    rows = []
    for acts in actions_per_agent:
        arr = [[-1.0, -1.0, -1.0]] * MAX_ACTIONS_PA
        for i, move in enumerate(acts[:MAX_ACTIONS_PA]):
            arr[i] = [float(move[0]), float(move[1]), float(move[2])]
        rows.append(arr)
    return jnp.asarray(rows, dtype=jnp.float32)


# --------------------------------------------------------------------------- #
# Frame-level comparison helpers
# --------------------------------------------------------------------------- #

def compare_planets(
    jx_planets: list, kg_planets: list, tag: str, errors: list[str],
) -> None:
    jx_by_id = {p[0]: p for p in jx_planets}
    kg_by_id = {p[0]: p for p in kg_planets}
    all_ids  = sorted(set(jx_by_id) | set(kg_by_id))

    for pid in all_ids:
        jp = jx_by_id.get(pid)
        kp = kg_by_id.get(pid)
        pfx = f"{tag} planet{pid}"
        if jp is None:
            errors.append(f"{pfx}: MISSING in JAX (present in kaggle)")
            continue
        if kp is None:
            errors.append(f"{pfx}: EXTRA in JAX (absent from kaggle)")
            continue
        if abs(jp[2] - kp[2]) > POS_TOL or abs(jp[3] - kp[3]) > POS_TOL:
            errors.append(
                f"{pfx}: pos  jax=({jp[2]:.4f},{jp[3]:.4f})  "
                f"kaggle=({kp[2]:.4f},{kp[3]:.4f})"
            )
        if jp[5] != kp[5]:
            errors.append(f"{pfx}: ships  jax={jp[5]}  kaggle={kp[5]}")
        if jp[1] != kp[1]:
            errors.append(f"{pfx}: owner  jax={jp[1]}  kaggle={kp[1]}")
        if abs(jp[4] - kp[4]) > POS_TOL:
            errors.append(f"{pfx}: radius  jax={jp[4]:.4f}  kaggle={kp[4]:.4f}")
        if jp[6] != kp[6]:
            errors.append(f"{pfx}: production  jax={jp[6]}  kaggle={kp[6]}")


def compare_fleets(
    jx_fleets: list, kg_fleets: list, tag: str, errors: list[str],
) -> None:
    """Compare by position+owner sort (fleet IDs are not guaranteed to match)."""
    def sort_key(f):
        return (f[1], round(f[2], 2), round(f[3], 2))   # owner, x, y

    jx_s = sorted(jx_fleets, key=sort_key)
    kg_s = sorted(kg_fleets, key=sort_key)

    if len(jx_s) != len(kg_s):
        errors.append(
            f"{tag}: fleet count mismatch  jax={len(jx_s)}  kaggle={len(kg_s)}"
        )
        return

    for i, (jf, kf) in enumerate(zip(jx_s, kg_s)):
        pfx = f"{tag} fleet[{i}]"
        if jf[1] != kf[1]:
            errors.append(f"{pfx}: owner  jax={jf[1]}  kaggle={kf[1]}")
        if abs(jf[2] - kf[2]) > POS_TOL or abs(jf[3] - kf[3]) > POS_TOL:
            errors.append(
                f"{pfx}: pos  jax=({jf[2]:.4f},{jf[3]:.4f})  "
                f"kaggle=({kf[2]:.4f},{kf[3]:.4f})"
            )
        if jf[6] != kf[6]:
            errors.append(f"{pfx}: ships  jax={jf[6]}  kaggle={kf[6]}")


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #

def run_episode(ep: int, seed: int, max_ticks: int, verbose: bool) -> tuple[int, int]:
    """Run one episode; return (num_errors, num_ticks)."""

    # --- JAX env ---
    jx_state = init_jax_state(seed, num_agents=2)
    jx_state, _, _ = jax_step_jit(jx_state, _EMPTY_ACTIONS)   # init tick

    # --- Kaggle env ---
    kenv = kaggle_make(
        "orbit_wars",
        configuration={"seed": seed, "randomSeed": seed},
        debug=False,
    )
    kenv.reset(num_agents=2)
    kenv.step([[], []])   # kaggle init tick

    # Shared RNG (seeded separately from game seed for independence)
    rng = random.Random(seed * 1_000_003 + ep)

    errors: list[str] = []
    ticks_run = 0

    for tick in range(1, max_ticks + 1):
        # Generate actions from the JAX env's current planet list
        # (if envs are in sync, the planet lists are identical)
        jx_pl = jax_planets(jx_state)
        a0 = random_actions(jx_pl, 0, rng)
        a1 = random_actions(jx_pl, 1, rng)

        # --- Step JAX env ---
        jx_actions = encode_actions_jax([a0, a1])
        jx_state, jx_statuses, jx_rewards = jax_step_jit(jx_state, jx_actions)

        # --- Step kaggle env ---
        kenv.step([a0, a1])
        kobs = kenv.state[0].observation

        tag = f"ep{ep} tick{tick:03d}"

        # Compare planets
        compare_planets(jax_planets(jx_state), list(kobs.planets), tag, errors)

        # Compare fleets (kaggle fleets: [id, owner, x, y, angle, from_planet, ships])
        compare_fleets(jax_fleets(jx_state), list(kobs.fleets), tag, errors)

        # Step counter
        jx_step_val = int(jax.device_get(jx_state.step))
        if jx_step_val != kobs.step:
            errors.append(f"{tag}: step  jax={jx_step_val}  kaggle={kobs.step}")

        # Termination
        jx_done  = bool(jax.device_get(jx_statuses[0]) == 1)
        kag_done = kenv.done

        if jx_done != kag_done:
            errors.append(
                f"{tag}: done  jax={jx_done}  kaggle={kag_done}"
            )

        ticks_run = tick
        if jx_done or kag_done:
            # Reward comparison
            if jx_done and kag_done:
                jx_r  = float(jax.device_get(jx_rewards[0]))
                kag_r = float(kenv.state[0].reward or 0.0)
                if jx_r != kag_r:
                    errors.append(f"ep{ep}: reward  jax={jx_r}  kaggle={kag_r}")
            break

    if verbose and errors:
        for e in errors[:15]:
            print(f"    !! {e}")
        if len(errors) > 15:
            print(f"    ... ({len(errors) - 15} more errors)")

    return len(errors), ticks_run


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="src_jax vs kaggle trajectory check")
    ap.add_argument("--episodes",    type=int, default=5)
    ap.add_argument("--max-ticks",   type=int, default=500)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    total_errors = 0
    failed_eps   = 0

    print(f"Comparing src_jax (JAX) vs kaggle_environments — {args.episodes} episodes\n")

    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        n_err, ticks = run_episode(ep + 1, seed, args.max_ticks, args.verbose)
        total_errors += n_err
        if n_err:
            failed_eps += 1
        status = "OK" if n_err == 0 else f"FAIL ({n_err} errors)"
        print(f"  ep{ep+1:3d}  seed={seed:5d}  ticks={ticks:4d}  {status}")

    print()
    print("=" * 55)
    print(f"Episodes       : {args.episodes}")
    print(f"Total errors   : {total_errors}")
    print(f"Failed episodes: {failed_eps} / {args.episodes}")
    if total_errors == 0:
        print("\nPASSED — JAX env is frame-perfectly identical to kaggle.")
    else:
        print(f"\nFAILED — {total_errors} mismatches across {failed_eps} episodes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
