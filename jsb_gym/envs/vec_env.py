"""
vec_env.py
==========
ChunkSubprocVecEnv — a high-throughput vectorised environment wrapper.

Instead of 1 env per worker process (naive SubprocVecEnv = N-way IPC barrier
every step), this runs W worker processes each stepping M envs in-series per
message. One IPC round-trip covers M env-steps, amortising the pickle/pipe
overhead that kills naive SubprocVecEnv at scale.

Sweet spot for this machine (20 logical cores, WSL2):
    W = 12 workers,  M = 30 envs/worker  →  360 total envs

Usage (drop-in replacement for SubprocVecEnv in main.py):

    from jsb_gym.envs.vec_env import ChunkSubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    vec_env = make_vec_env(
        BVRBase,
        n_envs      = 360,
        vec_env_cls = ChunkSubprocVecEnv,
        env_kwargs  = {'conf': baseEnv_conf},
    )
"""

import multiprocessing as mp
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs, VecEnvStepReturn, VecEnvIndices,
)
import gymnasium as gym
from typing import Callable, List, Optional, Sequence, Type, Union


def _worker(
    remote: mp.connection.Connection,
    env_fns: List[Callable[[], gym.Env]],
) -> None:
    """Worker process: owns a DummyVecEnv of M envs, steps them all per message."""
    import torch
    torch.set_num_threads(1)   # CRITICAL — prevents thread thrash across workers

    # Under 'spawn' (Windows/macOS) the env factories arrive cloudpickled as
    # bytes (plain pickle can't serialise the closures); under 'fork' they come
    # through directly. Handle both.
    import cloudpickle
    if isinstance(env_fns, (bytes, bytearray)):
        env_fns = cloudpickle.loads(env_fns)
    chunk_env = DummyVecEnv(env_fns)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rewards, dones, infos = chunk_env.step(data)
                remote.send((obs, rewards, dones, infos))
            elif cmd == "reset":
                obs = chunk_env.reset()
                remote.send(obs)
            elif cmd == "reset_one":
                idx = data
                obs, info = chunk_env.envs[idx].reset()
                remote.send((obs, info))
            elif cmd == "get_attr":
                attr, idx = data
                remote.send(getattr(chunk_env, attr) if idx is None
                             else getattr(chunk_env.envs[idx], attr))
            elif cmd == "set_attr":
                attr, value, idx = data
                if idx is None:
                    setattr(chunk_env, attr, value)
                else:
                    setattr(chunk_env.envs[idx], attr, value)
                remote.send(None)
            elif cmd == "env_method":
                method, args, kwargs, idx = data
                if idx is None:
                    remote.send(getattr(chunk_env, method)(*args, **kwargs))
                else:
                    remote.send(getattr(chunk_env.envs[idx], method)(*args, **kwargs))
            elif cmd == "get_spaces":
                remote.send((chunk_env.observation_space, chunk_env.action_space))
            elif cmd == "close":
                chunk_env.close()
                remote.close()
                break
    except Exception as e:
        remote.send(RuntimeError(f"Worker error: {e}"))
        raise


class ChunkSubprocVecEnv(VecEnv):
    """
    W worker processes, each running M envs in-series via DummyVecEnv.
    Total envs = W * M.  One IPC barrier per W*M env-steps.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], n_workers: int = None):
        n_total  = len(env_fns)
        # infer spaces from one env
        dummy = env_fns[0]()
        obs_space = dummy.observation_space
        act_space = dummy.action_space
        dummy.close()
        del dummy

        super().__init__(n_total, obs_space, act_space)

        # Worker count: empirically tuned to ~0.8 x logical cores.
        # On hybrid P+E CPUs (e.g. i7-14650HX) throughput plateaus near
        # logical*0.8 and dips beyond — measured sweet spot = 16 on a
        # 20-logical machine. Override via n_workers kwarg for other CPUs.
        if n_workers is None:
            n_workers = max(1, (mp.cpu_count() * 4) // 5)
        n_workers   = min(n_total, max(1, n_workers))
        chunks      = [env_fns[i::n_workers] for i in range(n_workers)]
        self._chunk_sizes = [len(c) for c in chunks]

        # 'fork' (Linux) inherits memory cheaply; Windows/macOS only have
        # 'spawn', which pickles the worker args -> cloudpickle the env factory
        # chunks so the closures survive.
        import cloudpickle
        _methods = mp.get_all_start_methods()
        _use_fork = "fork" in _methods
        ctx = mp.get_context("fork" if _use_fork else "spawn")
        self._remotes, self._work_remotes = zip(*[ctx.Pipe() for _ in chunks])
        self._processes = []
        for remote, work_remote, chunk in zip(
            self._remotes, self._work_remotes, chunks
        ):
            payload = chunk if _use_fork else cloudpickle.dumps(chunk)
            p = ctx.Process(
                target=_worker,
                args=(work_remote, payload),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            work_remote.close()

        # ask each worker for their spaces (sanity check)
        for remote in self._remotes:
            remote.send(("get_spaces", None))
        for remote in self._remotes:
            remote.recv()

        self._buf_obs = None  # allocated on first reset

    # ------------------------------------------------------------------ #
    #  VecEnv interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> VecEnvObs:
        for remote in self._remotes:
            remote.send(("reset", None))
        obs_chunks    = [remote.recv() for remote in self._remotes]
        self._buf_obs = np.concatenate(obs_chunks, axis=0)
        return self._buf_obs

    def step_async(self, actions: np.ndarray) -> None:
        idx = 0
        for remote, size in zip(self._remotes, self._chunk_sizes):
            remote.send(("step", actions[idx: idx + size]))
            idx += size

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self._remotes]
        obs_l, rew_l, done_l, info_l = zip(*results)
        obs   = np.concatenate(obs_l,  axis=0)
        rews  = np.concatenate(rew_l,  axis=0)
        dones = np.concatenate(done_l, axis=0)
        infos = [i for chunk in info_l for i in chunk]
        return obs, rews, dones, infos

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self._processes:
            p.join()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        self._remotes[0].send(("get_attr", (attr_name, None)))
        return [self._remotes[0].recv()] * self.num_envs

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices = None) -> None:
        for remote in self._remotes:
            remote.send(("set_attr", (attr_name, value, None)))
        for remote in self._remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args,
                   indices: VecEnvIndices = None, **method_kwargs):
        self._remotes[0].send(
            ("env_method", (method_name, method_args, method_kwargs, None))
        )
        return [self._remotes[0].recv()] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        pass
