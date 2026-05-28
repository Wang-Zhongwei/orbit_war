"""
Procedural generation for Orbit Wars: planet layouts and comet paths.
"""
from __future__ import annotations

import math
import random as _stdlib_random

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

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense: list[tuple[float, float]] = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path: list[tuple[float, float]] = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _dist(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start: int | None = None
        board_end: int | None = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i
        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets = [p for p in initial_planets
                          if p[0] not in comet_pid_set
                          and _dist((p[2], p[3]), (CENTER, CENTER)) + p[4] >= ROTATION_RADIUS_LIMIT]
        orbiting_planets = [p for p in initial_planets
                            if p[0] not in comet_pid_set
                            and _dist((p[2], p[3]), (CENTER, CENTER)) + p[4] < ROTATION_RADIUS_LIMIT]

        buf = COMET_RADIUS + 0.5
        valid = True
        for k, (cx, cy) in enumerate(visible):
            if _dist((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _dist(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None
