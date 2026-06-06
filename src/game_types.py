
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]
    angular_velocity: float = 0.0
    comet_paths: dict[int, list] = None  # planet_id -> [[x,y], ...] remaining waypoints

    def __post_init__(self):
        if self.comet_paths is None:
            self.comet_paths = {}


def parse_observation(observation: Any) -> GameState:
    def obs_get(key: str, default: Any) -> Any:
        if isinstance(observation, dict):
            return observation.get(key, default)
        return getattr(observation, key, default)

    planets = [
        PlanetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )
        for row in obs_get("planets", [])
    ]
    fleets = [
        FleetState(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )
        for row in obs_get("fleets", [])
    ]
    comet_paths: dict[int, list] = {}
    for group in obs_get("comets", []):
        g = group if isinstance(group, dict) else vars(group)
        idx = g.get("path_index", -1)
        for pid, path in zip(g.get("planet_ids", []), g.get("paths", [])):
            comet_paths[int(pid)] = path[idx + 1:]

    return GameState(
        step=int(obs_get("step", 0)),
        player=int(obs_get("player", 0)),
        planets=planets,
        fleets=fleets,
        angular_velocity=float(obs_get("angular_velocity", 0.0)),
        comet_paths=comet_paths,
    )
