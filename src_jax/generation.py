"""
Procedural generation for Orbit Wars: planet layouts and comet paths.
"""
from __future__ import annotations

import math
import random as _stdlib_random

import numpy as np

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    MAX_PRODUCTION,
)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_planets(rng: _stdlib_random.Random | None = None) -> list[list]:
    if rng is None:
        rng = _stdlib_random
    planets: list[list] = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, MAX_PRODUCTION)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp: list[list] = [
            [id_counter,     -1, y,              x,              r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y,              r, ships, prod],
            [id_counter + 2, -1, x,              BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        valid = True
        for tp in temp:
            for p in planets:
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            planets.extend(temp)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)

        orbital_radius = _dist((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + r + 10:
            continue
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue

        ships = rng.randint(5, 30)
        temp = [
            [id_counter,     -1, y,              x,              r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y,              r, ships, prod],
            [id_counter + 2, -1, x,              BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        valid = True
        for tp in temp:
            tp_orbital = _dist((tp[2], tp[3]), (CENTER, CENTER))
            tp_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orbital = _dist((p[2], p[3]), (CENTER, CENTER))
                p_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                if tp_rotating != p_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp)
            id_counter += 4

    return planets


def generate_comet_paths(
    initial_planets: list[list],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: list[int] | None = None,
    comet_speed: float = 4.0,
    rng: _stdlib_random.Random | None = None,
) -> list[list[list[float]]] | None:
    if rng is None:
        rng = _stdlib_random
    comet_pid_set: set[int] = set(comet_planet_ids) if comet_planet_ids else set()

    static_planets = [p for p in initial_planets
                      if p[0] not in comet_pid_set
                      and _dist((p[2], p[3]), (CENTER, CENTER)) + p[4] >= ROTATION_RADIUS_LIMIT]
    orbiting_planets = [p for p in initial_planets
                        if p[0] not in comet_pid_set
                        and _dist((p[2], p[3]), (CENTER, CENTER)) + p[4] < ROTATION_RADIUS_LIMIT]

    # Precompute orbiting planet orbit params (static across attempts)
    if orbiting_planets:
        op_x0 = np.array([p[2] for p in orbiting_planets], dtype=np.float64)
        op_y0 = np.array([p[3] for p in orbiting_planets], dtype=np.float64)
        op_r_arr = np.array([p[4] for p in orbiting_planets], dtype=np.float64)
        dx_init = op_x0 - CENTER
        dy_init = op_y0 - CENTER
        op_orb_r = np.hypot(dx_init, dy_init)
        op_init_angle = np.arctan2(dy_init, dx_init)
    else:
        op_r_arr = op_orb_r = op_init_angle = None

    if static_planets:
        sp_x = np.array([p[2] for p in static_planets], dtype=np.float64)
        sp_y = np.array([p[3] for p in static_planets], dtype=np.float64)
        sp_r = np.array([p[4] for p in static_planets], dtype=np.float64)
    else:
        sp_x = sp_y = sp_r = None

    buf = COMET_RADIUS + 0.5
    num = 5000
    t_arr = np.linspace(0.3 * math.pi, 0.3 * math.pi + 1.4 * math.pi, num)

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        # Vectorized dense ellipse generation (replaces 5000-iteration Python loop)
        ex = c_val + a * np.cos(t_arr)
        ey = b * np.sin(t_arr)
        x_arr = CENTER + ex * cos_phi - ey * sin_phi
        y_arr = CENTER + ex * sin_phi + ey * cos_phi
        dense = np.stack([x_arr, y_arr], axis=1)  # [5000, 2]

        # Vectorized cumulative-distance subsampling (replaces second Python loop)
        diffs = np.diff(dense, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
        targets = np.arange(comet_speed, float(cum_dist[-1]) + comet_speed, comet_speed)
        indices = np.searchsorted(cum_dist, targets)
        indices = indices[indices < num]
        path = dense[np.concatenate([[0], indices])]  # [N, 2]

        # Board bounds filtering
        in_bounds = ((path[:, 0] >= 0) & (path[:, 0] <= BOARD_SIZE) &
                     (path[:, 1] >= 0) & (path[:, 1] <= BOARD_SIZE))
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) == 0:
            continue
        visible = path[int(valid_idx[0]):int(valid_idx[-1]) + 1]  # [V, 2]
        if not (5 <= len(visible) <= 40):
            continue

        vx = visible[:, 0]  # x coords (cx in original)
        vy = visible[:, 1]  # y coords (cy in original)

        # 4 symmetric path variants — same layout as original [[coord0, coord1], ...]
        paths = [
            np.stack([vy, vx], axis=1).tolist(),
            np.stack([BOARD_SIZE - vx, vy], axis=1).tolist(),
            np.stack([vx, BOARD_SIZE - vy], axis=1).tolist(),
            np.stack([BOARD_SIZE - vy, BOARD_SIZE - vx], axis=1).tolist(),
        ]

        # sym_pt[k, j] = j-th symmetric position for visible point k
        # sym_pt 0 = (cy, cx), 1 = (BOARD-cx, cy), 2 = (cx, BOARD-cy), 3 = (BOARD-cy, BOARD-cx)
        sym0 = np.stack([vy, BOARD_SIZE - vx, vx, BOARD_SIZE - vy], axis=1)  # [V, 4] first coord
        sym1 = np.stack([vx, vy, BOARD_SIZE - vy, BOARD_SIZE - vx], axis=1)  # [V, 4] second coord

        # Sun collision (vectorized)
        if np.any(np.hypot(vx - CENTER, vy - CENTER) < SUN_RADIUS + COMET_RADIUS):
            continue

        # Static planet collision (vectorized, all sym-pts × all planets at once)
        valid = True
        if sp_x is not None:
            dx = sym0[:, :, None] - sp_x[None, None, :]  # [V, 4, P]
            dy = sym1[:, :, None] - sp_y[None, None, :]
            if np.any(np.hypot(dx, dy) < (sp_r[None, None, :] + buf)):
                valid = False

        # Orbiting planet collision (vectorized, planet position varies per step)
        if valid and op_orb_r is not None:
            V = len(visible)
            game_steps = (spawn_step - 1) + np.arange(V, dtype=np.float64)  # [V]
            cur_angles = op_init_angle[None, :] + angular_velocity * game_steps[:, None]  # [V, Q]
            px = CENTER + op_orb_r[None, :] * np.cos(cur_angles)  # [V, Q]
            py = CENTER + op_orb_r[None, :] * np.sin(cur_angles)
            dx = sym0[:, :, None] - px[:, None, :]  # [V, 4, Q]
            dy = sym1[:, :, None] - py[:, None, :]
            if np.any(np.hypot(dx, dy) < (op_r_arr[None, None, :] + COMET_RADIUS)):
                valid = False

        if valid:
            return paths
    return None
