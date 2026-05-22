from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection

import torch

from .config import TrainConfig
from .env import OrbitWarsEnv, StepResult
from .evaluation import SelfPlayOpponent, build_opponent
from .features import TurnBatch


def _worker_fn(pipe: Connection, env_idx: int, cfg: TrainConfig, opponent_name: str, device_str: str) -> None:
    """Runs in a child process. Owns one OrbitWarsEnv and blocks on pipe commands."""
    device = torch.device(device_str)
    opponent = build_opponent(opponent_name, cfg=cfg, device=device)
    env = OrbitWarsEnv(cfg, opponent, env_index=env_idx)
    try:
        while True:
            msg = pipe.recv() # wait for command from parent process
            cmd = msg[0]
            if cmd == "reset":
                pipe.send(env.reset(seed=msg[1])) # send back to the parent process
            elif cmd == "step":
                pipe.send(env.step(msg[1]))
            elif cmd == "sync":
                if isinstance(env.opponent, SelfPlayOpponent) and env.opponent.is_live:
                    env.opponent.sync_from(msg[1])
                pipe.send(None)
            elif cmd == "close":
                break
    finally:
        pipe.close()


class SubprocVecEnv:
    """N OrbitWarsEnv instances each in its own subprocess, stepped in parallel.

    ``opponent_names`` is cycled across workers: env 0 gets opponent_names[0],
    env 1 gets opponent_names[1], env 2 gets opponent_names[0], and so on.

    Usage pattern in the rollout loop:
        vec_env.step_async(moves_per_env)   # dispatch all steps simultaneously
        results = vec_env.step_wait()        # collect when all workers are done
    """

    def __init__(
        self,
        num_envs: int,
        cfg: TrainConfig,
        opponent_names: list[str],
        worker_device: str = "cpu",
    ) -> None:
        ctx = mp.get_context("spawn")
        self._pipes: list[Connection] = []
        self._procs: list[mp.Process] = []
        for idx in range(num_envs):
            name = opponent_names[idx % len(opponent_names)]
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_fn,
                args=(child, idx, cfg, name, worker_device),
                daemon=True,
            )
            proc.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(proc)

    def __len__(self) -> int:
        return len(self._pipes)

    def reset_all(self, seeds: list[int]) -> list[TurnBatch]:
        for pipe, seed in zip(self._pipes, seeds):
            pipe.send(("reset", seed))
        return [pipe.recv() for pipe in self._pipes]

    def reset_one(self, env_idx: int, seed: int) -> TurnBatch:
        self._pipes[env_idx].send(("reset", seed))
        return self._pipes[env_idx].recv()

    def step_async(self, moves_list: list[list]) -> None:
        for pipe, moves in zip(self._pipes, moves_list):
            pipe.send(("step", moves))

    def step_wait(self) -> list[StepResult]:
        return [pipe.recv() for pipe in self._pipes] # receive StepResult from each child process

    def sync_opponent(self, state_dict: dict) -> None:
        """Push updated policy weights to live self-play workers (no-op for others)."""
        cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
        for pipe in self._pipes:
            pipe.send(("sync", cpu_sd))
        for pipe in self._pipes:
            pipe.recv()

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
        for pipe in self._pipes:
            pipe.close()
